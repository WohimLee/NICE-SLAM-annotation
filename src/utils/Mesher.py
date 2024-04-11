import numpy as np
import open3d as o3d
import skimage
import torch
import torch.nn.functional as F
import trimesh
from packaging import version
from src.utils.datasets import get_dataset


class Mesher(object):

    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        """
        Mesher类, 给定场景表示, mesher从中提取网格
        
        参数:
            cfg (dict): 解析后的配置字典
            args (class 'argparse.Namespace'): argparse解析的参数
            slam (class NICE-SLAM): NICE-SLAM主类
            points_batch_size (int): 单批次查询中的最大点数
                                     用于减轻GPU内存使用, 默认为500000
            ray_batch_size (int): 单批次查询中的最大光线数
                                  用于减轻GPU内存使用, 默认为100000
        """
        # 设置点和光线的批处理大小，用于控制内存使用
        self.points_batch_size = points_batch_size
        self.ray_batch_size = ray_batch_size
        
        # 从SLAM系统获取渲染器实例
        self.renderer = slam.renderer
        
        # 从配置中获取是否使用粗糙级别、缩放比例和是否考虑占据性等设置
        self.coarse = cfg['coarse']
        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        
        # 从配置中获取网格生成相关的参数，包括分辨率、等值面设置、清理网格的边界缩放因子、
        # 移除小几何体的阈值、提取颜色网格的方法、是否获取最大组件以及是否进行深度测试
        self.resolution = cfg['meshing']['resolution']
        self.level_set = cfg['meshing']['level_set']
        self.clean_mesh_bound_scale = cfg['meshing']['clean_mesh_bound_scale']
        self.remove_small_geometry_threshold = cfg['meshing']['remove_small_geometry_threshold']
        self.color_mesh_extraction_method = cfg['meshing']['color_mesh_extraction_method']
        self.get_largest_components = cfg['meshing']['get_largest_components']
        self.depth_test = cfg['meshing']['depth_test']
        
        # 从SLAM系统中获取场景边界、是否使用NICE模型和是否打印详细信息的设置
        self.bound = slam.bound
        self.nice = slam.nice
        self.verbose = slam.verbose

        # 根据配置和缩放比例设置Marching Cubes算法的边界
        self.marching_cubes_bound = torch.from_numpy(
            np.array(cfg['mapping']['marching_cubes_bound']) * self.scale)

        # 初始化数据集读取器，用于读取输入数据
        self.frame_reader = get_dataset(cfg, args, self.scale, device='cpu')
        self.n_img = len(self.frame_reader) # 计算数据集中的图片数量
        
        # 从SLAM系统中获取相机参数
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def point_masks(self, input_points, keyframe_dict, estimate_c2w_list,
                    idx, device, get_mask_use_all_frames=False):
        """
        Split the input points into seen, unseen, and forcast,
        according to the estimated camera pose and depth image.

        Args:
            input_points (tensor): input points.
            keyframe_dict (list): list of keyframe info dictionary.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current frame index.
            device (str): device name to compute on.

        Returns:
            seen_mask (tensor): the mask for seen area.
            forecast_mask (tensor): the mask for forecast area.
            unseen_mask (tensor): the mask for unseen area.
        """
        # Retrieves the image dimensions and camera intrinsic parameters stored in the class.
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        # Ensures the input points are in a PyTorch tensor format.
        if not isinstance(input_points, torch.Tensor):
            input_points = torch.from_numpy(input_points)
            
        # Prepares the input points by detaching them from any previous computation graph.
        input_points = input_points.clone().detach()
        
        # Initializes lists to hold binary masks for seen, forecast, and unseen points.
        seen_mask_list = []
        forecast_mask_list = []
        unseen_mask_list = []
        
        # Iterates over batches of input points, splitting them according to points_batch_size.
        for i, pnts in enumerate(
                torch.split(input_points, self.points_batch_size, dim=0)):
            
            # Converts points to the specified device and data type.
            points = pnts.to(device).float()
            # should divide the points into three parts, seen and forecast and unseen
            # - seen: union of all the points in the viewing frustum of keyframes
            # - forecast: union of all the points in the extended edge of the viewing frustum of keyframes
            # - unseen: all the other points

            # Initializes seen and forecast masks as tensors of False values on the specified device.
            seen_mask = torch.zeros((points.shape[0])).bool().to(device)
            forecast_mask = torch.zeros((points.shape[0])).bool().to(device)
            
            # Conditionally processes all frames up to the current index if get_mask_use_all_frames is True.
            if get_mask_use_all_frames:
                # Iterates over all frames up to the current one.
                for i in range(0, idx + 1, 1):
                    
                    # Converts the camera-to-world matrix to world-to-camera by inversion.
                    c2w = estimate_c2w_list[i].cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    # Transfers the world-to-camera matrix to the specified device as a float tensor.
                    w2c = torch.from_numpy(w2c).to(device).float()
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    
                    # Homogenizes the points by adding a column of ones, making them compatible for transformation.
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()  # (N, 4)
                    # (N, 4, 1)=(4,4)*(N, 4, 1)
                    
                    # Transforms the points from world to camera coordinates.
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)

                    # Constructs the camera intrinsic matrix and transfers it to the specified device.
                    K = torch.from_numpy(
                        np.array([[fx, .0, cx], [.0, fy, cy],
                                  [.0, .0, 1.0]]).reshape(3, 3)).to(device)
                    cam_cord[:, 0] *= -1
                    
                    # Projects the camera coordinates to image coordinates and normalizes by the depth (z) component.
                    uv = K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    
                    # Creates a mask for points within the image boundaries, marking them as seen.
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    # Adjusts the edge value to expand the boundaries artificially for forecast points.
                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    # forecast
                    cur_mask_forecast = cur_mask_forecast.reshape(-1)
                    # seen
                    cur_mask_seen = cur_mask_seen.reshape(-1)

                    # Updates the global masks with the current batch's results using logical OR.
                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast
            else:
                for keyframe in keyframe_dict:
                    c2w = keyframe['est_c2w'].cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    w2c = torch.from_numpy(w2c).to(device).float()
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]

                    K = torch.from_numpy(
                        np.array([[fx, .0, cx], [.0, fy, cy],
                                  [.0, .0, 1.0]]).reshape(3, 3)).to(device)
                    cam_cord[:, 0] *= -1
                    uv = K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    if self.depth_test:
                        gt_depth = keyframe['depth'].to(
                            device).reshape(1, 1, H, W)
                        vgrid = uv.reshape(1, 1, -1, 2)
                        # normalized to [-1, 1]
                        vgrid[..., 0] = (vgrid[..., 0] / (W-1) * 2.0 - 1.0)
                        vgrid[..., 1] = (vgrid[..., 1] / (H-1) * 2.0 - 1.0)
                        depth_sample = F.grid_sample(
                            gt_depth, vgrid, padding_mode='zeros', align_corners=True)
                        depth_sample = depth_sample.reshape(-1)
                        max_depth = torch.max(depth_sample)
                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                                                        2].reshape(-1)
                        cur_mask_forecast[cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth
                        # seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - cam_cord[cur_mask_seen, 2].reshape(-1)
                        cur_mask_seen[cur_mask_seen.clone()] &= \
                            (proj_depth_seen < depth_sample[cur_mask_seen]+2.4) \
                            & (depth_sample[cur_mask_seen]-2.4 < proj_depth_seen)
                    else:
                        max_depth = torch.max(keyframe['depth'])*1.1

                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                                                        2].reshape(-1)
                        cur_mask_forecast[
                            cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth

                        # seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - \
                            cam_cord[cur_mask_seen, 2].reshape(-1)
                        cur_mask_seen[cur_mask_seen.clone(
                        )] &= proj_depth_seen < max_depth

                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast
                    
            # Adjusts forecast mask to exclude seen points, and calculates unseen mask by excluding both seen and forecast points.
            forecast_mask &= ~seen_mask
            unseen_mask = ~(seen_mask | forecast_mask)

            # Transfers the final masks back to the CPU and converts them to NumPy arrays for further processing.
            seen_mask = seen_mask.cpu().numpy()
            forecast_mask = forecast_mask.cpu().numpy()
            unseen_mask = unseen_mask.cpu().numpy()

            seen_mask_list.append(seen_mask)
            forecast_mask_list.append(forecast_mask)
            unseen_mask_list.append(unseen_mask)

        # Concatenates the batch-wise masks into complete arrays for the entire set of points.
        seen_mask = np.concatenate(seen_mask_list, axis=0)
        forecast_mask = np.concatenate(forecast_mask_list, axis=0)
        unseen_mask = np.concatenate(unseen_mask_list, axis=0)
        
        # Returns the binary masks indicating whether each point is 
        # seen, forecast, or unseen relative to the keyframes' viewing frustums and the camera poses.
        return seen_mask, forecast_mask, unseen_mask

    def get_bound_from_frames(self, keyframe_dict, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """
        # Retrieves image dimensions and camera intrinsic parameters from the instance.
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        # Checks the version of the Open3D library to ensure compatibility with the method calls that follow.
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            # Initializes a scalable TSDF (Truncated Signed Distance Function) volume using parameters suitable for the current scene scale. 
            # This is for the newer versions of Open3D.
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        # Initializes the TSDF volume for older versions of Open3D.
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
            
        # Initializes an empty list to store camera positions from each keyframe.
        cam_points = []
        
        # Iterates over each keyframe in the provided dictionary.
        for keyframe in keyframe_dict:
            
            # Retrieves the estimated camera-to-world transformation matrix for the keyframe.
            c2w = keyframe['est_c2w'].cpu().numpy()
            
            # convert to open3d camera pose
            # Adjusts the transformation matrix for use with Open3D conventions 
            # and calculates the inverse (world-to-camera).
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            
            # Stores the camera position extracted from the transformation matrix.
            cam_points.append(c2w[:3, 3])
            
            depth = keyframe['depth'].cpu().numpy()
            color = keyframe['color'].cpu().numpy()

            # Converts depth and color images to Open3D Image objects, with color images normalized appropriately.
            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            # Creates an Open3D PinholeCameraIntrinsic object using the extracted camera parameters.
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            
            # Creates an RGBD image from the color and depth images.
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            
            # Integrates the RGBD image into the TSDF volume using the world-to-camera transformation.
            volume.integrate(rgbd, intrinsic, w2c)
            
        # Stacks the camera positions into a NumPy array.
        cam_points = np.stack(cam_points, axis=0)
        
        # Extracts a mesh from the TSDF volume.
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        
        # Concatenates camera positions with mesh vertices to include in the convex hull computation.
        points = np.concatenate([cam_points, mesh_points], axis=0)
        
        # Converts the concatenated points into an Open3D PointCloud, then computes its convex hull.
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        
        # Computes vertex normals for the convex hull mesh.
        mesh.compute_vertex_normals()
        
        # Checks Open3D version for compatibility with scaling operations.
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # Scales the convex hull mesh based on a predefined scale factor around its center. 
            # This is conditional on Open3D version.
            mesh = mesh.scale(self.clean_mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.clean_mesh_bound_scale, center=True)
            
        # Extracts vertices and faces from the scaled mesh.
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        
        # Creates a trimesh mesh from the vertices and faces for external use.
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        
        # Returns the computed convex hull as a trimesh object.
        return return_mesh

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): point coordinates.
            decoders (nn.module decoders): decoders.
            c (dicts, optional): feature grids. Defaults to None.
            stage (str, optional): query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """
        # Splits the input points into smaller batches based on self.points_batch_size to manage GPU memory usage effectively.
        p_split = torch.split(p, self.points_batch_size)
        
        # Retrieves the scene bounds stored in self.bound and 
        # initializes an empty list rets to collect the results from processing each batch of points.
        bound = self.bound
        rets = []
        
        # Iterates over each batch of points.
        for pi in p_split:
            # mask for points out of bound
            # For each batch, computes masks to identify points that lie within the predefined scene bounds along each axis.
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            # Adds a batch dimension to the points tensor to match the input shape expected by the decoders.
            pi = pi.unsqueeze(0)
            
            # Passes the points through the decoders. 
            # If self.nice is True, it uses the provided feature grids c and the evaluation stage. 
            # If self.nice is False, it evaluates the points without any feature grids.
            if self.nice:
                ret = decoders(pi, c_grid=c, stage=stage)
            else:
                ret = decoders(pi, c_grid=None)
            
            # Removes the batch dimension from the output tensor.
            ret = ret.squeeze(0)
            
            # Ensures the output tensor has the correct shape, 
            # especially for cases where the output might represent a single point with properties (e.g., RGB and alpha or occupancy).
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            # Sets the value of the fourth property (presumably occupancy or an alpha channel) to 100 for points outside the scene bounds, 
            # indicating they are not part of the scene or are invalid.
            ret[~mask, 3] = 100
            
            # Collects the results from each batch into the rets list.
            rets.append(ret)
            
        # Concatenates the results from all batches along the first dimension 
        # to form a single tensor representing the evaluated properties of all input points.
        ret = torch.cat(rets, dim=0)
        
        # Returns the final tensor containing the evaluated properties (occupancy and/or color) for the input points.
        return ret

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        # Retrieves the bounding box (bound) for the marching cubes operation, stored as a class attribute. 
        # This bounding box defines the spatial extent within which the grid points will be generated.
        bound = self.marching_cubes_bound

        # Sets a padding value that will be used to slightly expand the bounding box to ensure the surface is fully captured, even at the edges.
        padding = 0.05
        
        # Generates linearly spaced coordinates along the X, Y, and Z axes within the padded bounds. 
        # The number of points along each axis is determined by resolution.
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding,
                        resolution)
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding,
                        resolution)
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding,
                        resolution)

        # Creates a 3D meshgrid from the X, Y, and Z coordinates. 
        # This meshgrid represents the Cartesian product of the three coordinate vectors, 
        # producing a grid of points spread throughout the volume defined by bound.
        xx, yy, zz = np.meshgrid(x, y, z)
        
        # Flattens the meshgrid arrays and stacks them vertically, 
        # then transposes the result to get a list of 3D coordinates. 
        # Each row of grid_points represents the coordinates of a point in the grid.
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        
        # Converts the NumPy array of grid points to a PyTorch tensor with floating-point data type. 
        # This step is repeated unnecessarily and could be optimized 
        # by removing the previous np.vstack operation that was assigned to grid_points.
        grid_points = torch.tensor(np.vstack(
            [xx.ravel(), yy.ravel(), zz.ravel()]).T,
            dtype=torch.float)
        
        # Returns a dictionary containing the tensor of grid points (grid_points) 
        # and a list of the original coordinate arrays for each axis (xyz). 
        # This data structure provides both the individual coordinates along each axis 
        # and the flattened grid points for use in surface reconstruction algorithms.
        return {"grid_points": grid_points, "xyz": [x, y, z]}

    def get_mesh(self,
                 mesh_out_file,
                 c,
                 decoders,
                 keyframe_dict,
                 estimate_c2w_list,
                 idx,
                 device='cuda:0',
                 show_forecast=False,
                 color=True,
                 clean_mesh=True,
                 get_mask_use_all_frames=False):
        """
        Extract mesh from scene representation and save mesh to file.

        Args:
            mesh_out_file (str): output mesh filename.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
            keyframe_dict (list):  list of keyframe info.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current processed camera ID.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.
            show_forecast (bool, optional): show forecast. Defaults to False.
            color (bool, optional): whether to extract colored mesh. Defaults to True.
            clean_mesh (bool, optional): whether to clean the output mesh 
                                        (remove outliers outside the convexhull and small geometry noise). 
                                        Defaults to True.
            get_mask_use_all_frames (bool, optional): 
                whether to use all frames or just keyframes when getting the seen/unseen mask. Defaults to False.
        """
        with torch.no_grad():
            # Creates a uniform grid within the scene, which will be used to sample points for the marching cubes algorithm.
            grid = self.get_grid_uniform(self.resolution)
            # Moves the grid points to the specified computing device.
            points = grid['grid_points']
            points = points.to(device)
            
            # Conditionally executes if the forecast region should be shown in the mesh.
            if show_forecast:
                # Determines which points in the grid are seen, forecast, and unseen based on the camera poses.
                seen_mask, forecast_mask, unseen_mask = self.point_masks(
                    points, keyframe_dict, estimate_c2w_list, idx, device=device, 
                    get_mask_use_all_frames=get_mask_use_all_frames)
                # Filters points that are in the forecast and seen categories.
                forecast_points = points[forecast_mask]
                seen_points = points[seen_mask]

                # Initializes a list to store occupancy values for forecast points.
                z_forecast = []
                
                # Iterates over batches of forecast points.
                for i, pnts in enumerate(
                        torch.split(forecast_points,
                                    self.points_batch_size,
                                    dim=0)):
                    # Evaluates and stores the occupancy values of forecast points, using a coarser level of detail.
                    z_forecast.append(
                        self.eval_points(pnts, decoders, c, 'coarse',
                                         device).cpu().numpy()[:, -1])
                # Concatenates the occupancy values into a single array.
                z_forecast = np.concatenate(z_forecast, axis=0)
                z_forecast += 0.2

                z_seen = []
                for i, pnts in enumerate(
                        torch.split(seen_points, self.points_batch_size,
                                    dim=0)):
                    z_seen.append(
                        self.eval_points(pnts, decoders, c, 'fine',
                                         device).cpu().numpy()[:, -1])
                z_seen = np.concatenate(z_seen, axis=0)
                
                # Initializes an array to hold occupancy values for all points, defaulting to zero.
                z = np.zeros(points.shape[0])
                z[seen_mask] = z_seen
                z[forecast_mask] = z_forecast
                z[unseen_mask] = -100
            # Executes if the forecast region should not be shown.
            else:
                # Computes the scene bound (convex hull) from keyframe information to filter points outside the scene.
                mesh_bound = self.get_bound_from_frames(
                    keyframe_dict, self.scale)
                z = []
                mask = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    # Checks which points are contained within the scene bounds.
                    mask.append(mesh_bound.contains(pnts.cpu().numpy()))
                mask = np.concatenate(mask, axis=0)
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    # Evaluates the occupancy values for points within the scene using a finer level of detail.
                    z.append(self.eval_points(pnts, decoders, c, 'fine',
                                              device).cpu().numpy()[:, -1])

                z = np.concatenate(z, axis=0)
                z[~mask] = 100

            z = z.astype(np.float32)

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print(
                    'marching_cubes error. Possibly no surface extracted from the level set.'
                )
                return

            # convert back to world coordinates
            vertices = verts + np.array(
                [grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if clean_mesh:
                if show_forecast:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices,
                                           faces=faces,
                                           process=False)
                    mesh_bound = self.get_bound_from_frames(
                        keyframe_dict, self.scale)
                    contain_mask = []
                    for i, pnts in enumerate(
                            np.array_split(points, self.points_batch_size,
                                           axis=0)):
                        contain_mask.append(mesh_bound.contains(pnts))
                    contain_mask = np.concatenate(contain_mask, axis=0)
                    not_contain_mask = ~contain_mask
                    face_mask = not_contain_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)
                else:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices,
                                           faces=faces,
                                           process=False)
                    seen_mask, forecast_mask, unseen_mask = self.point_masks(
                        points, keyframe_dict, estimate_c2w_list, idx, device=device, 
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    unseen_mask = ~seen_mask
                    face_mask = unseen_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)

                # get connected components
                components = mesh.split(only_watertight=False)
                if self.get_largest_components:
                    areas = np.array([c.area for c in components], dtype=np.float)
                    mesh = components[areas.argmax()]
                else:
                    new_components = []
                    for comp in components:
                        if comp.area > self.remove_small_geometry_threshold * self.scale * self.scale:
                            new_components.append(comp)
                    mesh = trimesh.util.concatenate(new_components)
                vertices = mesh.vertices
                faces = mesh.faces

            if color:
                if self.color_mesh_extraction_method == 'direct_point_query':
                    # color is extracted by passing the coordinates of mesh vertices through the network
                    points = torch.from_numpy(vertices)
                    z = []
                    for i, pnts in enumerate(
                            torch.split(points, self.points_batch_size, dim=0)):
                        z_color = self.eval_points(
                            pnts.to(device).float(), decoders, c, 'color',
                            device).cpu()[..., :3]
                        z.append(z_color)
                    z = torch.cat(z, axis=0)
                    vertex_colors = z.numpy()

                elif self.color_mesh_extraction_method == 'render_ray_along_normal':
                    # for imap*
                    # render out the color of the ray along vertex normal, and assign it to vertex color
                    import open3d as o3d
                    mesh = o3d.geometry.TriangleMesh(
                        vertices=o3d.utility.Vector3dVector(vertices),
                        triangles=o3d.utility.Vector3iVector(faces))
                    mesh.compute_vertex_normals()
                    vertex_normals = np.asarray(mesh.vertex_normals)
                    rays_d = torch.from_numpy(vertex_normals).to(device)
                    sign = -1.0
                    length = 0.1
                    rays_o = torch.from_numpy(
                        vertices+sign*length*vertex_normals).to(device)
                    color_list = []
                    batch_size = self.ray_batch_size
                    gt_depth = torch.zeros(vertices.shape[0]).to(device)
                    gt_depth[:] = length
                    for i in range(0, rays_d.shape[0], batch_size):
                        rays_d_batch = rays_d[i:i+batch_size]
                        rays_o_batch = rays_o[i:i+batch_size]
                        gt_depth_batch = gt_depth[i:i+batch_size]
                        depth, uncertainty, color = self.renderer.render_batch_ray(
                            c, decoders, rays_d_batch, rays_o_batch, device, 
                            stage='color', gt_depth=gt_depth_batch)
                        color_list.append(color)
                    color = torch.cat(color_list, dim=0)
                    vertex_colors = color.cpu().numpy()

                vertex_colors = np.clip(vertex_colors, 0, 1) * 255
                vertex_colors = vertex_colors.astype(np.uint8)

                # cyan color for forecast region
                if show_forecast:
                    seen_mask, forecast_mask, unseen_mask = self.point_masks(
                        vertices, keyframe_dict, estimate_c2w_list, idx, device=device,
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    vertex_colors[forecast_mask, 0] = 0
                    vertex_colors[forecast_mask, 1] = 255
                    vertex_colors[forecast_mask, 2] = 255

            else:
                vertex_colors = None

            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh.export(mesh_out_file)
            if self.verbose:
                print('Saved mesh at', mesh_out_file)
                