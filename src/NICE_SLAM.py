import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')


class NICE_SLAM():
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process. 
    """

    def __init__(self, cfg, args):
        # Stores the configuration settings, 
        # command-line arguments, 
        # and a specific argument (nice) for further use within the class.
        self.cfg = cfg
        self.args = args
        self.nice = args.nice

        # Initializes various properties from the configuration settings, 
        # including flags for coarse mapping, occupancy mapping, low GPU memory mode, verbosity, dataset details, 
        # and parameters for bounding enlargement in coarse mapping.
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']
        
        # Determines the output directory for storing results, 
        # preferring the command-line specification over the configuration setting.
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        
        # Sets the directory path for saving checkpoints.
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        
        # Ensures the creation of the output, checkpoints, and mesh directories, 
        # avoiding errors if they already exist.
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        
        # Extracts camera parameters from the configuration, 
        # including image height, width, and intrinsic parameters.
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        
        # Calls a method to update camera settings, presumably based on the parameters just set.
        self.update_cam()

        # Loads a model based on the configuration and the nice argument.
        model = config.get_model(cfg,  nice=self.nice)
        
        # Stores the loaded model for shared access across different components of the system.
        self.shared_decoders = model

        # Sets the scaling factor from the configuration.
        self.scale = cfg['scale']

        # Calls a method to load boundary conditions for mapping.
        self.load_bound(cfg)
        
        # Loads pretrained models and initializes the grid if the nice argument is True; 
        # otherwise, initializes an empty dictionary for shared content.
        if self.nice:
            self.load_pretrain(cfg)
            self.grid_init(cfg)
        else:
            self.shared_c = {}

        # need to use spawn
        # Attempts to set the multiprocessing start method to 'spawn', 
        # ignoring any RuntimeErrors if this setting cannot be changed.
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Initializes a dataset reader based on the configuration, command-line arguments, and scale factor.
        self.frame_reader = get_dataset(cfg, args, self.scale)
        
        # Stores the number of images or frames in the dataset.
        self.n_img = len(self.frame_reader)
        
        # Initializes a shared memory tensor for storing estimated camera-to-world transformation matrices for each image.
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()

        # Initializes a shared memory tensor for ground truth camera-to-world transformations, if available.
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        
        # Initializes shared memory tensors for indexing and mapping control, 
        # including current index, first frame to map, current frame being mapped, 
        # and a counter for mapping operations.
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()
        
        # Transfers any shared content to the specified device for mapping and enables shared memory access.
        for key, val in self.shared_c.items():
            val = val.to(self.cfg['mapping']['device'])
            val.share_memory_()
            self.shared_c[key] = val
            
        # Moves the shared decoders model to the specified device for mapping.
        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        
        # Enables shared memory for the decoders model.
        self.shared_decoders.share_memory()
        
        # Initializes the renderer, mesher, logger, and mapper components with the current configuration and class instance, 
        # setting the mapper for fine mapping.
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)
        
        # If coarse mapping is enabled, initializes a separate mapper for coarse mapping.
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)
        
        # Initializes a tracker component for tracking the camera's position and orientation.
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound'])*self.scale)
        bound_divisible = cfg['grid_len']['bound_divisible']
        # enlarge the bound a bit to allow it divisible by bound_divisible
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisible).int()+1)*bound_divisible+self.bound[:, 0]
        if self.nice:
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        if self.coarse:
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'],
                              map_location=cfg['mapping']['device'])
            coarse_dict = {}
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids. 初始化 hierarchical feature grids

        Args:
            cfg (dict): parsed config dict.
        """

        # 检查是否指启用 coarse grid（self.coarse为True）
        # 如果是，则从配置中读取 coarse grid 的长度并存储为属性以供后续使用
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
            
        # 从配置中读取并存储 mid、fine 和 color grid 的长度到相应的属性
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        # 初始化一个字典c来保存网格。从配置中读取通道维度（c_dim）
        # 通过从上界减去下界来计算每个网格的域长度（xyz_len）
        c = {}
        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        # If you have questions regarding the swap of axis 0 and 2,
        # please refer to https://github.com/cvg/nice-slam/issues/24
        '''The swapping of axes in the shape calculation 
        (coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]) 
        is an interesting aspect. 
        According to the given comment and the linked GitHub issue, 
        this swap addresses a specific alignment or ordering issue in how the data is structured or processed, 
        ensuring the grid aligns correctly with the spatial dimensions it represents.'''

        # 如果启用了 coarse grid，根据域长度和网格长度计算 shape,
        # 使用正态分布初始化网格值，并将其存储在字典c中
        if self.coarse:
            coarse_key = 'grid_coarse'
            coarse_val_shape = list(
                map(int, (xyz_len*self.coarse_bound_enlarge/coarse_grid_len).tolist()))
            coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0] # 调整维度, 适应数据结构和处理流程的约定，确保网格的空间排列与预期的一致
            self.coarse_val_shape = coarse_val_shape  # 每个维度的voxel数量
            val_shape = [1, c_dim, *coarse_val_shape] # c_dim表示每个网格点的特征维度，后面是voxel的shape
            coarse_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
            c[coarse_key] = coarse_val

        # 类似的过程用于 mid grid: 计算 shape，初始化其值，并将其存储在c中
        middle_key = 'grid_middle'
        middle_val_shape = list(map(int, (xyz_len/middle_grid_len).tolist()))
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        self.middle_val_shape = middle_val_shape
        val_shape = [1, c_dim, *middle_val_shape]
        middle_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[middle_key] = middle_val

        # 类似的过程用于 fine grid: 计算 shape，初始化其值，并将其存储在c中
        fine_key = 'grid_fine'
        fine_val_shape = list(map(int, (xyz_len/fine_grid_len).tolist()))
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001)
        c[fine_key] = fine_val

        # 类似的过程用于 color grid：计算 shape，初始化其值，并将其存储在c中
        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len/color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[color_key] = color_val
        
        # 将初始化的网格字典存储在属性shared_c中，以便共享访问
        self.shared_c = c

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)
        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def coarse_mapping(self, rank):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(3):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            elif rank == 2:
                if self.coarse:
                    p = mp.Process(target=self.coarse_mapping, args=(rank, ))
                else:
                    continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
