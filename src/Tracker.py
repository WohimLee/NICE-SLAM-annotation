import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ): # Tracker类负责跟踪相机的位姿
        # 存储配置和SLAM系统参数
        self.cfg = cfg
        self.args = args

        # 从配置中提取跟scale、coarse level、occupancy、同步方法相关的设置
        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        # 与 mapper 使用的一致
        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr'] # 相机学习率
        self.device = cfg['tracking']['device'] # 相机设备
        self.num_cam_iters = cfg['tracking']['iters']   # 迭代次数
        self.gt_camera = cfg['tracking']['gt_camera']   # 真实相机位姿
        self.tracking_pixels = cfg['tracking']['pixels']    # 跟踪使用的像素数量
        self.seperate_LR = cfg['tracking']['seperate_LR']   # 是否分离学习率
        self.w_color_loss = cfg['tracking']['w_color_loss'] # 颜色损失权重
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']   # 忽略图像边缘的宽度和高度
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']   
        self.handle_dynamic = cfg['tracking']['handle_dynamic'] # 是否处理动态物体
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']   # 是否在跟踪中使用颜色
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption'] # 是否假设恒定速度

        # 从 mapping 配置中提取每帧处理、第一帧不可视化的设置
        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        # 初始化前一次映射索引为-1，准备数据集读取器和帧加载器
        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        
        # 初始化可视化器，用于跟踪过程的可视化
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device
        self.c = {}
        
        # 如果 verbose 详细模式开启，则直接使用frame_loader进行迭代，否则使用tqdm生成进度条
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)
            
        # 遍历数据集的每一帧，idx是当前帧索引，gt_*分别是真实的颜色、深度和相机位姿
        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}") # 设置进度条的描述

            # 将批处理的数据维度降为单帧数据
            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            # 如果同步方法为"strict"，则在 mapping 之后进行跟踪
            if self.sync_method == 'strict':    
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    # 每隔self.every_frame帧进行一次 mapping，等待直到 mapping 索引追上当前帧索引
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            
            # 如果同步方法为"loose"，则允许一定范围内的 mapping 延迟
            elif self.sync_method == 'loose':   
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            
            # 如果同步方法为"free"，则完全并行运行映射和跟踪，可能导致不平衡
            elif self.sync_method == 'free':    
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()

            if self.verbose: # 在详细模式下打印跟踪信息
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)
            
            # 对于第一帧, 如果是第一帧或者使用真实相机位姿，直接使用真实相机位姿
            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    # 如果不是在第一帧禁止可视化，则进行可视化
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)
                    
            # 对于后续帧, 根据上一帧位姿和当前帧深度进行相机位姿估计
            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx-2 >= 0:
                    # 如果假设恒定速度，根据前两帧的位姿变化预测新的相机位姿
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w

                # 从估计的新的相机位姿获取相机的 tensor 表示
                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR: 
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    # 如果分别优化旋转和平移，则使用不同的学习率
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else: # 如果没有设置分别优化 R 和 t
                    camera_tensor = Variable( # 将摄像机张量转移到计算设备上，并设置为可优化
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor] # 创建包含摄像机张量的参数列表
                    optimizer_camera = torch.optim.Adam( # 创建优化器，为摄像机参数设置学习率
                        cam_para_list, lr=self.cam_lr)

                # 计算初始损失：这一步计算的是估计的摄像机位姿张量与真实摄像机位姿张量之间的差的绝对值的平均值
                # 这里的损失用于评估初始的估计精度，并在优化过程中作为对比基准
                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                # 初始化候选摄像机张量变量，这个变量用于存储在优化过程中发现的损失最小的摄像机位姿
                candidate_cam_tensor = None
                # 设置一个非常大的初始最小损失值，这个值会在接下来的优化迭代中更新
                # 如果在迭代过程中找到一个损失更小的摄像机位姿，该值将被更新为较小的损失值
                current_min_loss = 10000000000.
                
                # 优化相机位姿
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR: # 分开 R 和 t 优化
                        camera_tensor = torch.cat([quad, T], 0).to(self.device) # 将旋转和平移参数合并成一个张量

                    # 使用visualizer工具可视化当前迭代的结果，包括当前帧索引、迭代次数、深度图、颜色图和摄像机张量
                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    # 调用optimize_cam_in_batch函数来优化摄像机参数，计算此次迭代的损失
                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

                    if cam_iter == 0: # 如果是第一次迭代，记录下初始的损失值
                        initial_loss = loss

                    # 计算当前相机 tensor 与 gt 之间的误差
                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    
                    # 如果开启了详细输出，打印最后一次迭代的重渲染损失和相机张量误差
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    # 如果当前损失小于当前记录的最小损失，则更新最小损失和候选相机 tensor
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                # 构建齐次坐标的最后一行，用于从相机 tensor 中得到完整的4x4相机矩阵
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                
                # 更新相机位姿
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
                
            # 更新估计的相机位姿列表和真实相机位姿列表
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()   # 保存当前帧的位姿作为下一帧的预估位姿
            self.idx[0] = idx       # 更新处理的最新帧索引
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
