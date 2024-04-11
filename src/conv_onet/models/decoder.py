import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale) # 初始化矩阵_B, 元素从高斯分布中随机采样
        else: # 如果learnable为False，则_B只是一个普通的tensor，不会在训练过程中更新
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0) 
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    # DenseLayer类继承自nn.Linear, 是一个带有激活函数的全连接层
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs):
        # 初始化函数，接收输入维度in_dim，输出维度out_dim，激活函数类型activation，默认为"relu"
        self.activation = activation 
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self):
        # 重置层参数的方法，用于初始化或重新初始化权重和偏置
        # 使用Xavier均匀初始化方法初始化权重，该方法可以根据激活函数的类型自动调整初始化的scale
        # torch.nn.init.calculate_gain(self.activation)根据激活函数类型计算适当的scale
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            # 如果存在偏置，则将其初始化为零
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP(nn.Module):
    """
    解码器。点的坐标不仅用于特征网格采样, 还作为 MLP 的输入
    
    参数:
        name (str): 解码器的名称
        dim (int): 输入维度
        c_dim (int): 特征维度
        hidden_size (int): 解码器网络的隐藏层大小
        n_blocks (int): 层数
        leaky (bool): 是否使用leaky ReLU激活函数
        sample_mode (str): 采样特征策略, bilinear|nearest
        color (bool): 是否输出颜色
        skips (list): 有跳跃连接的层的列表
        grid_len (float): 对应特征网格的体素长度
        pos_embedding_method (str): 位置嵌入方法
        concat_feature (bool): 是否从中层获取特征并与当前特征进行拼接
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        if c_dim != 0:
            # 如果特征维度不为0，则为每一层初始化一个全连接层，用于处理特征向量
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])
            
        # 根据位置编码方法初始化相应的编码模块
        if pos_embedding_method == 'fourier':   # 使用高斯傅里叶特征变换
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':    # 使用相同的嵌入方法，即不改变
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':    # 使用Nerf中的位置编码方法
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu': # 使用带有ReLU激活函数的全连接层进行编码
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')
            
        # 初始化点坐标的线性层，包括跳跃连接
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])
        
        # 初始化输出层
        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        # 设置激活函数
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        # 采样网格特征函数。将点坐标归一化，然后在给定的特征网格上采样特征
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        # 实际上如果 mode = 'bilinear', 则为三线性插值
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None):
        # 前向传播函数。根据 level 选择不同的解码路径，合并特征和/或颜色信息
        if self.c_dim != 0: # 如果特征维度不为0，从特征网格中采样特征
            c = self.sample_grid_feature(
                p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

            if self.concat_feature:
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                # 如果需要拼接特征（仅在 fine 解码器中），则从 mid level 采样特征并与当前特征拼接
                with torch.no_grad():
                    c_middle = self.sample_grid_feature(
                        p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)

        p = p.float()

        # 应用位置编码，并通过 MLP 处理编码后的点坐标
        embedded_pts = self.embedder(p)
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class MLP_no_xyz(nn.Module):
    """
    解码器。点坐标仅用于特征网格采样, 不作为MLP的输入

    参数:
        name (str): 解码器的名称
        dim (int): 输入维度
        c_dim (int): 特征维度
        hidden_size (int): 解码器网络的隐藏层大小
        n_blocks (int): 层数
        leaky (bool): 是否使用leaky ReLUs激活函数
        sample_mode (str): 采样特征策略, bilinear|nearest
        color (bool): 是否输出颜色
        skips (list): 拥有跳跃连接的层的列表
        grid_len (float): 对应特征网格的体素长度
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False,
                 sample_mode='bilinear', color=False, skips=[2], grid_len=0.16):
        super().__init__()
        
        # 初始化解码器的基本属性和模块列表
        self.name = name
        self.no_grad_feature = False
        self.color = color
        self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        # 初始化线性层模块列表，包括普通的线性层和带有跳跃连接的线性层
        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color: # 如果需要输出 color，则输出层的维度为4（RGB+透明度）
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else: # 否则，输出层的维度为1（例如，用于 occupancy 或 distance 的单个值）
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky: # 设置激活函数为ReLU，除非指定使用leaky ReLU
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode # 设置采样模式（双线性或最近邻）

    def sample_grid_feature(self, p, grid_feature):
        # 定义一个函数用于在给定的特征网格上采样特征
        # 将输入的点坐标归一化
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        
        # 创建一个虚拟网格，用于特征采样, 使用grid_sample函数根据虚拟网格在特征网格上采样特征，然后去掉多余的维度
        c = F.grid_sample(grid_feature, vgrid, padding_mode='border',
                          align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c # 返回采样得到的特征
 
    def forward(self, p, c_grid, **kwargs):
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)
        h = c
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([c, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class NICE(nn.Module):
    """    
    Neural Implicit Scalable Encoding.

    参数:
        dim (int): 输入维度
        c_dim (int): 特征维度
        coarse_grid_len (float): coarse grid 的体素长度
        middle_grid_len (float): middle grid 的体素长度
        fine_grid_len (float)  : fine grid 的体素长度
        color_grid_len (float) : color grid 的体素长度
        hidden_size (int): 解码器 MLP 的隐藏层大小
        coarse (bool): 是否使用 coarse 网络
        pos_embedding_method (str): 位置编码方法
    """

    def __init__(self, dim=3, c_dim=32,
                 coarse_grid_len=2.0,  middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier'):
        super().__init__()

        if coarse: # 如果使用 coarse 层级, 初始化一个没有XYZ输入的多层感知机(MLP)作为 coarse 解码器
            self.coarse_decoder = MLP_no_xyz(
                name='coarse', dim=dim, c_dim=c_dim, color=False, hidden_size=hidden_size, grid_len=coarse_grid_len)
            
        # 分别为mid、fine 和 color level初始化 MLP 解码器
        self.middle_decoder = MLP(name='middle', dim=dim, c_dim=c_dim, color=False,
                                  skips=[2], n_blocks=5, hidden_size=hidden_size,
                                  grid_len=middle_grid_len, pos_embedding_method=pos_embedding_method)
        self.fine_decoder = MLP(name='fine', dim=dim, c_dim=c_dim*2, color=False,
                                skips=[2], n_blocks=5, hidden_size=hidden_size,
                                grid_len=fine_grid_len, concat_feature=True, pos_embedding_method=pos_embedding_method)
        self.color_decoder = MLP(name='color', dim=dim, c_dim=c_dim, color=True,
                                 skips=[2], n_blocks=5, hidden_size=hidden_size,
                                 grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)

    def forward(self, p, c_grid, stage='middle', **kwargs):
        """
            不同 Level 输出 occupancy 或者 color
        """
        device = f'cuda:{p.get_device()}'
        if stage == 'coarse':
            # coarse level，只计算 occupancy 并返回扩展后的输出
            occ = self.coarse_decoder(p, c_grid)
            occ = occ.squeeze(0)
            raw = torch.zeros(occ.shape[0], 4).to(device).float()
            raw[..., -1] = occ
            return raw
        elif stage == 'middle':
            # mid level，同样只计算 occupancy 并返回扩展后的输出
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            raw[..., -1] = middle_occ
            return raw
        elif stage == 'fine':
            # fine level，计算 fine 和 mid level 的 occupancy 之和
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ+middle_occ
            return raw
        elif stage == 'color':
            # color level, 计算颜色, 并将 fine 和 mid level 的 occupancy 之和用于最后一通道
            fine_occ = self.fine_decoder(p, c_grid)
            raw = self.color_decoder(p, c_grid)
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ+middle_occ
            return raw
