from src.conv_onet import models


def get_model(cfg,  nice=True):
    """
    Return the network model.

    Args:
        cfg (dict): 导入的yaml配置文件
        nice (bool, optional): 是否使用Neural Implicit Scalable Encoding(神经隐式可扩展编码)。默认为False。

    Returns:
        decoder (nn.module): 网络模型
    """

    dim = cfg['data']['dim'] # 数据的维度
    coarse_grid_len = cfg['grid_len']['coarse'] # coarse-level grid 的长度
    middle_grid_len = cfg['grid_len']['middle'] # mid-level grid 的长度
    fine_grid_len = cfg['grid_len']['fine']     # fine-level grid 的长度
    color_grid_len = cfg['grid_len']['color']   # color grid 的长度
    c_dim = cfg['model']['c_dim']  # feature dimensions 特征维度
    pos_embedding_method = cfg['model']['pos_embedding_method'] # 位置编码方法
    if nice: # NICE-SLAM
        decoder = models.decoder_dict['nice'](
            dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
            middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
            color_grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)
    else: # iMAP
        decoder = models.decoder_dict['imap'](
            dim=dim, c_dim=0, color=True,
            hidden_size=256, skips=[], n_blocks=4, pos_embedding_method=pos_embedding_method
        )
    return decoder
