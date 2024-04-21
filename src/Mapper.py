import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer




class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False
                 ):

        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper # 是否启用 coarse_mapper

        self.idx = slam.idx
        self.nice = slam.nice   # NICE模型
        self.c = slam.shared_c  # 共享特征网格
        self.bound = slam.bound # 场景边界
        self.logger = slam.logger   # 日志记录器
        self.mesher = slam.mesher   # mesh生成器
        self.output = slam.output   # 输出目录
        self.verbose = slam.verbose 
        self.renderer = slam.renderer           # 渲染器
        self.low_gpu_mem = slam.low_gpu_mem     # 是否低GPU内存模式
        self.mapping_idx = slam.mapping_idx     # 当前mapping的索引
        self.mapping_cnt = slam.mapping_cnt     # mapping 计数器
        self.decoders = slam.shared_decoders    # 共享解码器
        self.estimate_c2w_list = slam.estimate_c2w_list     # 估计的相机位姿列表
        self.mapping_first_frame = slam.mapping_first_frame # mapping的第一个帧索引

        # 从配置文件中获取与 mapping 和 mesh 生成相关的参数
        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.device = cfg['mapping']['device']  # 使用的设备
        self.fix_fine = cfg['mapping']['fix_fine']  # 是否固定 fine level
        self.eval_rec = cfg['meshing']['eval_rec']  # 是否评估重建质量
        self.BA = False  # 即使BA被启用，也只至少有4个关键帧时才会进行 BA
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']    
        self.mesh_freq = cfg['mapping']['mesh_freq']    # mesh 生成频率
        self.ckpt_freq = cfg['mapping']['ckpt_freq']    # checkpoint 保存频率
        self.fix_color = cfg['mapping']['fix_color']    # 是否固定 color level
        self.mapping_pixels = cfg['mapping']['pixels']  # mapping 中使用的像素数量
        self.num_joint_iters = cfg['mapping']['iters']  # 联合迭代次数
        self.clean_mesh = cfg['meshing']['clean_mesh']  # 是否清理 mesh
        self.every_frame = cfg['mapping']['every_frame']    # 每帧进行映射
        self.color_refine = cfg['mapping']['color_refine']  # 对 color 进行细化
        self.w_color_loss = cfg['mapping']['w_color_loss']  # color 损失权重
        self.keyframe_every = cfg['mapping']['keyframe_every']          # 关键帧间隔
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']        # fine level 迭代比例
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']    # middle level 迭代比例
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level']    # 是否在 coarse level 生成 mesh
        self.mapping_window_size = cfg['mapping']['mapping_window_size']        # mapping 窗口大小
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']    # 第一帧是否可视化
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']    # 第一帧是否记录日志
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']  # 第一帧是否生成mesh
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']    # 视锥体特征选择
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']    # 关键帧选择方法
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']  # 是否保存选定关键帧信息
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        # 如果是NICE模型且启用 coarse level，则设置关键帧选择方法为全局
        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = 'global'

        # 初始化关键帧字典和列表，数据集读取器，以及获取数据集中的图片数量
        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        
        # 如果输出目录不是 Demo，初始化 visualizer，用于 mapping 过程的可视化
        if 'Demo' not in self.output:  # disable this visualization in demo
            self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                         vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                         verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        # 为每个维度生成坐标点
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))
        # 将坐标点组合成3D点
        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        # 如果处理的是 grid_coarse，直接返回全部为True的掩码
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool)
            return mask
        # 备份points以便后续计算距离
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w) # 计算 w2c 矩阵
        
        # 添加齐次坐标
        ones = np.ones_like(points[:, 0]).reshape(-1, 1) 
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        
        # 转换到相机坐标系
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        # 构建内参矩阵
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        
        # 变换坐标系后计算像素坐标
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        # 利用opencv函数重新映射深度图获取每个3D点对应的深度
        remap_chunk = int(3e4) # 定义了每次处理的点的数量。可以管理内存使用，避免一次性处理过多数据导致内存不足
        depths = [] # 初始化一个列表，用来存储从深度图中提取的深度值
        for i in range(0, uv.shape[0], remap_chunk): # 循环, 通过每个块处理uv坐标
            depths += [cv2.remap(depth_np, # 原始的深度图
                                 uv[i:i+remap_chunk, 0], # 当前块内所有点的x坐标
                                 uv[i:i+remap_chunk, 1], # 当前块内所有点的y坐标
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)
        
        # 计算有效范围内的掩码
        edge = 0
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        # 处理深度为零的情况
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        # 深度测试，确定哪些点在有效深度范围内
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        # 添加靠近摄像机中心的特征网格
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak-ray_o
        dist = torch.sum(dist*dist, axis=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        # 重塑掩码以适应原始网格形状
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100):
        """
        选择与当前相机观测重叠的关键帧

        参数:
            gt_color (tensor): 当前帧的 gt_color
            gt_depth (tensor): 当前帧的 gt_depth
            c2w (tensor): 相机到世界坐标系的变换矩阵
            keyframe_dict (list): 包含每个关键帧信息的列表
            k (int): 要选择的重叠关键帧数量
            N_samples (int, optional): 每条射线的样本点数, 默认为16。
            pixels (int, optional): 从当前相机图像中稀疏采样的像素数, 默认为100
        返回:
            selected_keyframe_list (list): 选定的关键帧ID列表
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        # 从当前相机位置生成 rays_o rays_d
        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        # 将深度图像转换成列向量，并复制N_samples次，为每个射线样本点准备深度值
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        
        # 生成介于0和1之间的N_samples个样本点的t值
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        
        # 计算每个样本点的近处和远处深度界限
        near = gt_depth*0.8 # 将实际深度缩放0.8倍作为起始深度
        far = gt_depth+0.5  # 在实际深度上增加0.5作为结束深度
        
        # 每个样本点的深度值，是near到far之间的线性插值
        z_vals = near * (1.-t_vals) + far * (t_vals)
        
        # 计算每个样本点的3D坐标，通过射线的起点加上射线方向乘以深度值
        # rays_o[..., None, :] 的形状是 [N_rays, 1, 3]
        # rays_d[..., None, :] 的形状是 [N_rays, 1, 3]
        # z_vals[..., :, None] 的形状是 [N_rays, N_samples, 1]
        # 所以 pts 的形状将是 [N_rays, N_samples, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
            
        # 将3D点位置转换为一个长的列表，方便后续处理
        vertices = pts.reshape(-1, 3).cpu().numpy()
        
        # 初始化关键帧列表
        list_keyframe = []
        
        # 遍历所有关键帧
        for keyframeid, keyframe in enumerate(keyframe_dict):
            
            c2w = keyframe['est_c2w'].cpu().numpy() # 从关键帧中获取估计的世界到相机的变换矩阵，并转换为numpy数组
            w2c = np.linalg.inv(c2w)    # 计算相机到世界坐标的逆矩阵，即世界到相机的变换矩阵
            
            # 为顶点数组添加齐次坐标
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1) 
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            
            # 使用世界到相机的变换矩阵将顶点从世界坐标变换到相机坐标
            cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            
            # 提取变换后的x, y, z坐标
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            
            # 构建相机内参矩阵
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            
            # 将x坐标取反，因为相机坐标系中x通常指向右，而图像坐标系中x指向左
            cam_cord[:, 0] *= -1
            
            # 将3D点转换为图像平面的2D点
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            
            # 设置图像边界，以确保点位于图像的有效区域内
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
                
            # 进一步确保z值（深度）为负，即点位于相机前方
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            
            # 计算在图像有效区域内的点的比例
            percent_inside = mask.sum()/uv.shape[0]
            # 将每个关键帧及其对应点在当前视角下的占比添加到列表中
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})
            
        # 对关键帧列表按照每个关键帧的重叠比例进行降序排序
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # 从排序后的列表中选取重叠比例大于0的关键帧
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        # 对选取的关键帧列表进行随机打乱，以避免总是选择同一组关键帧，增加算法的鲁棒性
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        # 返回最终选取的关键帧列表，数量为 k 个
        return selected_keyframe_list

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w):
        """
        Mapping 优化迭代过程。从选定的关键帧中采样像素点, 然后优化场景表示和相机位姿 (如果启用了局部 BA)
        
        Args:
            num_joint_iters (int): Mapping 迭代次数
            lr_factor (float): 当前学习率的倍数因子
            idx (int): 当前帧的索引
            cur_gt_color (tensor): 当前 gt color
            cur_gt_depth (tensor): 当前 gt depth
            gt_cur_c2w (tensor): 当前帧 gt c2w 矩阵
            keyframe_dict (list): 关键帧信息字典列表
            keyframe_list (list): 关键帧索引列表
            cur_c2w (tensor): 当前帧的 c2w 估计

        Returns:
            cur_c2w/None (tensor/None): 返回更新后的 cur_c2w, 如果没有进行 BA 则返回 None
        """
        
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy # 获取图像的高度、宽度和相机内参
        c = self.c # 获取当前场景的特征, 也就是 grid
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device) # 创建一个4x4变换矩阵的底部行，常用于表示齐次坐标下的变换矩阵

        if len(keyframe_dict) == 0: # 检查关键帧字典是否为空，若为空，则设置优化帧列表为空
            optimize_frame = []
        else: # 如果关键帧字典非空，根据选择策略确定哪些帧将被用于优化
            if self.keyframe_selection_method == 'global': # iMAP: 'global'策略: 从所有关键帧中随机选择特定数量的帧进行优化
                num = self.mapping_window_size-2 # 计算需要选择的帧数，减2是因为最后两个位置留给最新的关键帧和当前帧
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
            elif self.keyframe_selection_method == 'overlap': # NICE-SLAM: 'overlap'策略: 选择与当前帧有最大重叠区域的关键帧
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num) # 根据颜色、深度、当前帧的变换矩阵来选择帧

        # add the last keyframe and the current frame(use -1 to denote)
        # 添加最后一个关键帧和当前帧（使用 -1 表示）
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1] # 添加列表中最新的关键帧索引
            oldest_frame = min(optimize_frame) # 找到最旧的帧索引，用于在后续步骤中固定，以避免优化漂移
        optimize_frame += [-1] # 将当前帧索引（-1）添加到优化帧列表中

        # 保存选定关键帧的信息
        if self.save_selected_keyframes_info:
            keyframes_info = [] # 初始化一个空列表用于存储关键帧的信息
            # 遍历优化帧列表中的每个帧索引
            for id, frame in enumerate(optimize_frame):
                # 如果帧索引不是 -1（表示这是一个有效的关键帧而不是当前帧）
                if frame != -1:
                    frame_idx = keyframe_list[frame] # 获取实际的帧索引
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w'] # 获取该帧的 gt c2w矩阵
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w'] # 获取该帧的估计 c2w矩阵
                    
                else: # 如果帧索引是 -1，表示当前帧
                    frame_idx = idx # 当前帧的索引
                    tmp_gt_c2w = gt_cur_c2w # 当前帧的 gt c2w 矩阵
                    tmp_est_c2w = cur_c2w # 当前帧的估计 c2w 矩阵
                    
                # 将当前帧的 idx 和 c2w 矩阵信息添加到关键帧信息列表
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info # 将当前帧的关键帧信息列表保存到一个字典中，用当前帧的索引作为键

        # 计算每个图像应分配多少像素点进行处理，这是根据 mapping 像素总数除以优化帧的数量得出
        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        # 初始化 decoder 和 grid 参数列表
        decoders_para_list = [] # 用于存放所有 decoder 的参数
        coarse_grid_para = [] # coarse grid 参数
        middle_grid_para = [] # middle grid 参数
        fine_grid_para = [] # fine grid 参数
        color_grid_para = [] # color grid 参数
        gt_depth_np = cur_gt_depth.cpu().numpy()
        if self.nice:
            if self.frustum_feature_selection:
                masked_c_grad = {} # 用于存储视锥体内部可优化的网格梯度
                mask_c2w = cur_c2w # 当前帧的 c2w 矩阵，用于确定视锥体范围
            for key, val in c.items(): # 这里包含 coarse mid fine color 的 grid
                
                # 如果没有启用 frustum_feature_selection，则对所有 grid 参数进行常规的梯度优化设置
                if not self.frustum_feature_selection: 
                    val = Variable(val.to(device), requires_grad=True) # 将网格数据转移到计算设备并设置为可求导
                    c[key] = val # 更新网格数据
                    # 根据网格类型将其添加到相应的参数列表中
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val)
                    elif key == 'grid_color':
                        color_grid_para.append(val)
                        
                # 如果启用了 frustum_feature_selection，只优化当前相机视锥体内的grid部分
                else: 
                    mask = self.get_mask_from_c2w(
                        mask_c2w, key, val.shape[2:], gt_depth_np)       # 计算当前网格与视锥体的交集掩码
                    mask = torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(
                        0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1) # 将掩码调整到合适的维度并复制到每个通道
                    val = val.to(device)
                    # val_grad is the optimizable part, other parameters will be fixed
                    # val_grad 是可优化部分，其他参数将保持固定
                    val_grad = val[mask].clone() # 克隆视锥体内的网格部分，并设置为可优化
                    val_grad = Variable(val_grad.to(
                        device), requires_grad=True) # 设置为可求导
                    masked_c_grad[key] = val_grad # 保存可优化的网格梯度
                    masked_c_grad[key+'mask'] = mask # 保存掩码
                    
                    # 将可优化的网格梯度添加到相应的参数列表中
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val_grad)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val_grad)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val_grad)
                    elif key == 'grid_color':
                        color_grid_para.append(val_grad)
        # 根据使用的模型选择 decoder 参数
        if self.nice:
            # 如果没有固定 fine  decoder，将其参数添加到 decoder 参数列表中
            if not self.fix_fine:
                decoders_para_list += list(
                    self.decoders.fine_decoder.parameters())
            # 如果没有固定颜色 decoder，将其参数添加到 decoder 参数列表中
            if not self.fix_color:
                decoders_para_list += list(
                    self.decoders.color_decoder.parameters())
        else:
            # imap*, single MLP
            # 如果使用的是imap，将所有 decoder 参数添加到 decoder 参数列表中
            decoders_para_list += list(self.decoders.parameters())

        # 如果启用了 BA，则进行相机位姿的优化
        if self.BA: 
            camera_tensor_list = [] # 初始化一个空列表用于存放可优化的相机位姿张量
            gt_camera_tensor_list = [] # 初始化一个空列表用于存放地面真实的相机位姿张量
            
            # 遍历所有选择的优化帧
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                # 最旧的帧应该被固定以避免漂移
                if frame != oldest_frame:
                    if frame != -1: # 如果帧索引不是-1，说明不是当前帧，从关键帧字典中获取估计的和真实的 c2w
                        c2w = keyframe_dict[frame]['est_c2w']
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else: # 如果帧索引是-1，说明是当前帧，直接使用当前帧的 c2w
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                        
                    # c2w 矩阵转换为 tensor，并设置为可优化
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    
                    # 将gt c2w 矩阵也转换为 tensor
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

        if self.nice:
            if self.BA:
                # The corresponding lr will be set according to which stage the optimization is in
                # 如果启用了 BA, 为 decoder、grid 参数和相机参数设置优化器
                # 此处学习率初始设为0, 将在后续代码中根据优化阶段动态调整
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                # 如果没有启用BA, 仅为 decoder 和 grid 参数设置优化器
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0}])
        else: # imap
            # imap*, single MLP
            if self.BA:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                optimizer = torch.optim.Adam(
                    [{'params': decoders_para_list, 'lr': 0}])
            # 设置学习率调度器，用于在优化过程中调整学习率
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

        # 联合迭代优化
        for joint_iter in range(num_joint_iters):
            if self.nice:
                # 如果启用 frustum_feature_selection
                if self.frustum_feature_selection:
                    for key, val in c.items():
                        # 根据 mapping 类型（coarse 或 fine）选择对应的 grid
                        if (self.coarse_mapper and 'coarse' in key) or \
                                ((not self.coarse_mapper) and ('coarse' not in key)):
                            val_grad = masked_c_grad[key]
                            mask = masked_c_grad[key+'mask']
                            val = val.to(device)
                            val[mask] = val_grad
                            c[key] = val
                # 根据ratio计算出来的迭代次数, 设置当前迭代的 mapping 阶段
                if self.coarse_mapper:
                    self.stage = 'coarse'
                elif joint_iter <= int(num_joint_iters*self.middle_iter_ratio):
                    self.stage = 'middle'
                elif joint_iter <= int(num_joint_iters*self.fine_iter_ratio):
                    self.stage = 'fine'
                else:
                    self.stage = 'color'
                # 根据当前阶段调整优化器的学习率
                optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor
                optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['coarse_lr']*lr_factor
                optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['middle_lr']*lr_factor
                optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['fine_lr']*lr_factor
                optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr']*lr_factor
                if self.BA:
                    if self.stage == 'color':
                        optimizer.param_groups[5]['lr'] = self.BA_cam_lr
            else: # imap
                self.stage = 'color'
                optimizer.param_groups[0]['lr'] = cfg['mapping']['imap_decoders_lr']
                if self.BA:
                    optimizer.param_groups[1]['lr'] = self.BA_cam_lr
                    
            # 进行可视化
            if (not (idx == 0 and self.no_vis_on_first_frame)) and ('Demo' not in self.output):
                self.visualizer.vis(
                    idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.c, self.decoders)
                
            # 在每个联合迭代过程的开始，重置优化器的梯度
            optimizer.zero_grad()
            
            # 初始化批处理射线方向、原点、深度和颜色数据的列表
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []

            # 设置相机 tensor 索引，用于从相机 tensor 列表中取得当前帧的相机参数
            camera_tensor_id = 0
            
            # 处理每一帧的数据，进行射线和颜色采样
            for frame in optimize_frame:
                if frame != -1: # 非当前帧（即关键帧）
                    gt_depth = keyframe_dict[frame]['depth'].to(device) # 获取当前关键帧的 depth 并移至设备
                    gt_color = keyframe_dict[frame]['color'].to(device) # 获取当前关键帧的 color 并移至设备
                    
                    # 如果开启了 BA 且当前帧不是最旧的帧，从相机 tensor 列表中获取对应的相机参数
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor) # 通过相机 tensor 计算 c2w 矩阵
                    else:
                        c2w = keyframe_dict[frame]['est_c2w'] # 否则直接使用估计的相机到世界坐标系的转换矩阵

                else: # 当前帧
                    gt_depth = cur_gt_depth.to(device) # 获取当前帧的 depth
                    gt_color = cur_gt_color.to(device) # 获取当前帧的 color
                    if self.BA:
                        # 如果开启了 BA，从相机 tensor 列表中获取当前帧的相机参数
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = cur_c2w # 否则使用当前帧的估计的 c2w 矩阵
                        
                # 使用指定参数获取射线原点、方向以及对应的 gt depth 和 color
                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                    0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

            # 合并
            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)

            if self.nice:
                # should pre-filter those out of bounding box depth value
                # 预筛选那些超出 bbox 深度值的射线
                with torch.no_grad():
                    # 将射线原点和方向增加一个维度，以便后续处理
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    # 计算射线与场景边界的交点，用于过滤不在边界内的射线
                    t = (self.bound.unsqueeze(0).to(
                        device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth # 根据计算的t值筛选在边界内的射线
                # 应用过滤掩码，仅保留有效的射线和相关数据
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
            # 使用渲染器处理过滤后的射线数据，计算深度、不确定性和颜色
            ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                 batch_rays_o, device, self.stage,
                                                 gt_depth=None if self.coarse_mapper else batch_gt_depth)
            depth, uncertainty, color = ret

            # 计算深度损失，仅在深度大于0的区域计算
            depth_mask = (batch_gt_depth > 0)
            loss = torch.abs(
                batch_gt_depth[depth_mask]-depth[depth_mask]).sum()

            # 如果是 color 阶段或非NICE模式，计算颜色损失，并加权到总损失中
            if ((not self.nice) or (self.stage == 'color')):
                color_loss = torch.abs(batch_gt_color - color).sum()
                weighted_color_loss = self.w_color_loss*color_loss
                loss += weighted_color_loss

            # for imap*, it uses volume density
            # 如果使用体积密度调整，即imap*模式，计算调整损失并加入总损失
            regulation = (not self.occupancy)
            if regulation:
                point_sigma = self.renderer.regulation(
                    c, self.decoders, batch_rays_d, batch_rays_o, batch_gt_depth, device, self.stage)
                regulation_loss = torch.abs(point_sigma).sum()
                loss += 0.0005*regulation_loss # 调整损失加权

            # 反向传播优化
            loss.backward(retain_graph=False)
            optimizer.step()
            if not self.nice:
                # for imap* 使用调度器进行学习率调整
                scheduler.step()
            optimizer.zero_grad() # 清空梯度

            # put selected and updated features back to the grid
            # 如果使用NICE-SLAM并且启用了视锥体特征选择，进行特征回填
            if self.nice and self.frustum_feature_selection:
                for key, val in c.items():
                    # 确定当前键是否适用于当前的优化 mapper 设置
                    if (self.coarse_mapper and 'coarse' in key) or \
                            ((not self.coarse_mapper) and ('coarse' not in key)):
                        val_grad = masked_c_grad[key] # 获取对应的优化梯度值
                        mask = masked_c_grad[key+'mask'] # 获取对应的掩码，掩码用于指示哪些特征是需要更新的
                        val = val.detach() # 从计算图中分离变量，防止梯度信息回传
                        val[mask] = val_grad.clone().detach() # 使用掩码更新特征网格中的值
                        c[key] = val # 更新特征字典中的值

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0 # 初始化用于追踪当前相机 tensor 索引的变量
            # 遍历优化帧列表
            for id, frame in enumerate(optimize_frame):
                if frame != -1: # 如果当前帧不是特殊标记（-1），即实际存在的帧
                    if frame != oldest_frame: # 如果当前帧不是最旧的帧（旧帧可能被固定不进行优化以防漂移）
                        # 从相机 tensor 列表中获取对应的相机位姿 tensor，并将其从计算图中分离
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        # 将3x4的相机矩阵转换为4x4的齐次坐标形式，添加一个[0, 0, 0, 1]的行
                        c2w = torch.cat([c2w, bottom], dim=0)
                        # 更新相机 tensor 索引
                        camera_tensor_id += 1
                        # 将更新后的相机世界矩阵保存回关键帧字典
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    # 获取当前帧的相机位姿 tensor 并分离
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0) # 转换为齐次坐标形式
                    cur_c2w = c2w.clone() # 保存当前帧的更新后相机姿态
        if self.BA: # 如果使用BA，返回更新后的当前帧相机姿态
            return cur_c2w
        else: # 如果没有使用BA，返回None
            return None

    def run(self):
        cfg = self.cfg
        # 读取第一帧数据并初始化估计的相机位姿列表的第0个
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        self.estimate_c2w_list[0] = gt_c2w.cpu()
        
        init = True     # 初始化标志位，用于标记是否是第一次迭代
        prev_idx = -1   # 前一次处理的帧索引, 初始化为 -1
        
        # 主循环, 直到处理完所有帧或满足退出条件
        while (1):
            # 预先判断, 根据几种不同的同步方法确定是否处理当前帧
            while True:
                idx = self.idx[0].clone() # 当前帧索引
                if idx == self.n_img-1: # 如果当前是最后一帧，则退出循环
                    break
                if self.sync_method == 'strict':
                    if idx % self.every_frame == 0 and idx != prev_idx: # every_frame 默认设置 5
                        break
                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1) # 短暂延迟，以减少在等待过程中对 CPU 资源的过度占用，特别是在等待条件尚未满足时
            prev_idx = idx      # 更新前一次处理的帧索引

            # 如果开启了详细输出，打印当前正在 mapping 的帧信息
            if self.verbose:           
                print(Fore.GREEN)
                prefix = 'Coarse ' if self.coarse_mapper else ''
                print(prefix+"Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]

            # 如果不是初始化阶段，则设置 mapping 相关的参数
            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']

                # here provides a color refinement postprocess
                # 如果是最后一帧且启用了 color_refine，则调整迭代次数和其他参数
                if idx == self.n_img-1 and self.color_refine and not self.coarse_mapper:
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else: # 根据是否使用NICE模型，设置外部迭代次数
                    if self.nice:
                        outer_joint_iters = 1
                    else:
                        outer_joint_iters = 3
                        
            # 如果是初始化阶段，设置特定的学习率因子和迭代次数
            else:
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_first']

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)   # 获取当前帧的相机位姿
            num_joint_iters = num_joint_iters//outer_joint_iters    # 调整迭代次数
            
            # 外部迭代循环，可能包括 Bundle Adjustment 或其他 mapping 优化过程
            for outer_joint_iter in range(outer_joint_iters):
                # 根据keyframe_list数量、BA配置选项、非coarse mapper 决定是否进行 Bundle Adjustment
                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA'] and (
                    not self.coarse_mapper)
                
                # 执行 mapping 优化过程
                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
                
                # 如果进行了Bundle Adjustment，更新相机位姿
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

                # 最后一次循环，将当前帧添加到关键帧集合中
                if outer_joint_iter == outer_joint_iters-1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) \
                            and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                        ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})
                        
            # 如果开启了低GPU内存模式，清空缓存
            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False # 标记初始化阶段结束
            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1 # 更新状态，表明第一帧已完成 mapping，可以开始跟踪。

            # 如果不是 coarse_mapper, 只在 mapper 执行记录日志、生成 mesh等后续操作
            if not self.coarse_mapper:
                # 保存日志和生成 mesh
                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img-1:
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                                    selected_keyframes=self.selected_keyframes
                                    if self.save_selected_keyframes_info else None)

                self.mapping_idx[0] = idx   # 更新当前 mapping 索引
                self.mapping_cnt[0] += 1    # 更新 mapping 计数器

                # 按照设定频率生成 mesh
                if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)

                # 如果处理到最后一帧, 生成最终的 mesh 文件
                if idx == self.n_img-1:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    os.system(
                        f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    
                    # 如果需要评估重建质量, 生成专用的 mesh 文件
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)
                    break
            
            # 如果已处理到最后一帧，退出循环
            if idx == self.n_img-1:
                break
