import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):

        # 初始化 Renderer 对象, 并设置点和光线的批处理大小
        # 这些大小控制了在渲染过程中一次处理多少光线或点，影响性能和内存使用
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        # 从配置中提取特定于渲染的参数，存储为属性
        # 包括lindisp（线性视差）、perturb（用于抗锯齿的随机扰动）、
        # N_samples（每条光线的采样数）、N_surface（表面采样数）和N_importance（重要性采样的采样数）
        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance']

        # 存储额外的配置参数：
        # scale（影响渲染图像的大小或分辨率）和
        # occupancy（控制如何在渲染中使用占用网格）
        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        
        # 从slam对象检索并存储属性：
        # nice（可能是指NICE-SLAM模型或其神经隐式表示的引用）和
        # bound（正在渲染的场景的空间边界）
        self.nice = slam.nice
        self.bound = slam.bound

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """
        # Splits the input points into batches based on the points_batch_size attribute to manage memory usage during computation.
        p_split = torch.split(p, self.points_batch_size)
        
        # Retrieves the spatial boundaries of the scene from the bound attribute and 
        # initializes an empty list rets to store results.
        bound = self.bound
        rets = []
        
        # Iterates over each batch of points.
        for pi in p_split:
            # mask for points out of bound
            # Computes boolean masks for points within the spatial boundaries (bound) in each dimension (X, Y, Z). 
            # These masks are used to filter out points outside the boundaries.
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            # Adds a batch dimension to the points tensor for compatibility with the decoders.
            pi = pi.unsqueeze(0)
            
            # If the nice attribute is true, evaluates the points using the decoders and the feature grids c for the specified stage. 
            # If not, it evaluates without feature grids.
            if self.nice:
                ret = decoders(pi, c_grid=c, stage=stage)
            else:
                ret = decoders(pi, c_grid=None)
                
            # Removes the batch dimension from the results.
            ret = ret.squeeze(0)
            
            # If the result is a single point (shape [4]), it adds a dimension to maintain consistency in tensor shapes.
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            # For points outside the boundaries (mask), 
            # sets a default value (e.g., 100) for a specific channel (possibly indicating invalid points).
            ret[~mask, 3] = 100
            # Appends the results of the current batch to the list rets.
            rets.append(ret)
            
        # Concatenates all batch results into a single tensor.
        ret = torch.cat(rets, dim=0)
        return ret

    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """
        # Initializes variables for the number of samples per ray (N_samples), 
        # the number of surface samples (N_surface), 
        # and the number of samples for importance sampling (N_importance).
        N_samples = self.N_samples
        N_surface = self.N_surface
        N_importance = self.N_importance

        # Determines the number of rays by checking the shape of the rays' origins tensor.
        N_rays = rays_o.shape[0]

        # If rendering at the 'coarse' stage or if no ground truth depth is provided, 
        # disables surface sampling and sets a near clipping distance.
        if stage == 'coarse':
            gt_depth = None
        if gt_depth is None:
            N_surface = 0
            near = 0.01
            
        # For given ground truth depths, repeats and reshapes them for sampling, 
        # adjusting the near clipping distance based on these depths.
        else:
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples*0.01

        # Disables gradient computation to improve performance during rendering, as gradients are not needed.
        with torch.no_grad():
            # Prepares ray origins and directions for computation by detaching and reshaping them.
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            
            # Computes intersection times t of rays with the bounding box.
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2)
            
            # Determines the far clipping distance based on the bounding box intersection, 
            # ensuring rays do not extend beyond the scene bounds.
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01
            
        # If ground truth depth is provided, adjusts the far clipping distance based on it to ensure accurate depth rendering.
        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        else:
            far = far_bb
            
        # Checks if surface samples are to be used, indicating detailed near-surface rendering.
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(
                    gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask,
                               :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * \
                    (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(
                    0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask,
                               :] = z_vals_surface_depth_zero

        # Generates linearly spaced values between 0 and 1 for each sample along the rays.
        t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

        # If not using linear disparity, computes depth values z_vals linearly interpolated between near and far.
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        # If perturbation is enabled, adjusts sample positions randomly within their intervals for anti-aliasing.
        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        # Combines and sorts all depth values (from volume and surface sampling) for final sampling along rays.
        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        # Calculates 3D positions for all samples along each ray.
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        
        # Reshapes these positions into a two-dimensional tensor for processing.
        pointsf = pts.reshape(-1, 3)

        # Evaluates properties (color, occupancy) of points using the eval_points method.
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        # Reshapes the results to match the number of rays and samples.
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)

        # Converts raw network outputs to depth, uncertainty, and color, 
        # using a function (not defined here) that likely applies volume rendering formulas.
        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
        
        # If importance sampling is enabled, further refines depth and color estimation by sampling based on the distribution of learned weights.
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples+N_importance+N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
            return depth, uncertainty, color
        
        # Returns the final rendered depth, uncertainty, and color for the batch of rays.
        return depth, uncertainty, color

    def render_img(self, c, decoders, c2w, device, stage, gt_depth=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            # Retrieves the height (H) and width (W) of the image to be rendered from the class attributes.
            H = self.H
            W = self.W
            
            # Generates ray origins (rays_o) and directions (rays_d) for each pixel in the image using camera intrinsics, 
            # the camera-to-world transformation, and the device.
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)

            # Reshapes the rays to a two-dimensional array where each row represents a ray.
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # Initializes empty lists to store depth, uncertainty, and color values for each batch of rays.
            depth_list = []
            uncertainty_list = []
            color_list = []

            # Retrieves the batch size for processing rays to manage computational resources efficiently.
            ray_batch_size = self.ray_batch_size
            
            # Reshapes the optional ground truth depth map to a one-dimensional array, aligning with the ray dimensions.
            gt_depth = gt_depth.reshape(-1)

            # Iterates over rays in batches to render them incrementally.
            for i in range(0, rays_d.shape[0], ray_batch_size):
                # Extracts a batch of ray directions and origins for processing.
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                
                # Calls render_batch_ray for the current batch, passing ground truth depth if available.
                if gt_depth is None:
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=gt_depth_batch)

                # Extracts depth, uncertainty, and color from the returned values.
                depth, uncertainty, color = ret
                
                # Appends the batch's results to the respective lists, 
                # converting depth and uncertainty to double precision for numerical stability.
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)
                
            # Concatenates the lists into tensors to form complete images.
            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            # Reshapes the tensors to the original image dimensions. 
            # The color tensor gets an additional dimension for the RGB channels.
            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color

    # this is only for imap*
    def regulation(self, c, decoders, rays_d, rays_o, gt_depth, device, stage='color'):
        """
        Regulation that discourage any geometry from the camera center to 0.85*depth.
        For imap, the geometry will not be as good if this loss is not added.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            gt_depth (tensor): sensor depth image
            device (str): device name to compute on.
            stage (str, optional):  query stage. Defaults to 'color'.

        Returns:
            sigma (tensor, N): volume density of sampled points.
        """
        
        # Reshapes the ground truth depth to ensure it is a two-dimensional tensor 
        # and then repeats it across a new dimension to match the number of samples per ray.
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, self.N_samples)
        
        # Creates linearly spaced values between 0 and 1 for sampling along the rays, moved to the specified device.
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        
        # Sets the near clipping plane to 0 and 
        # calculates the far clipping plane as 85% of the ground truth depth 
        # to focus on the region before obstructing geometry.
        near = 0.0
        far = gt_depth*0.85
        
        # Linearly interpolates between near and far values 
        # to get depth values (z_vals) for each sample along the rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
        
        # Sets a perturb flag to introduce randomness in sampling 
        # and checks if it's greater than 0 to apply perturbation.
        perturb = 1.0
        if perturb > 0.:
            # get intervals between samples
            # Computes midpoints between adjacent depth values, 
            # and determines the upper and lower bounds for each interval for perturbation.
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            
            # stratified samples in those intervals
            # Generates random values within [0, 1] for each interval, 
            # scaled and shifted to lie within each interval, effectively perturbing the depth values.
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand
            
        # Computes the 3D positions of all samples by extending ray origins along their directions by the perturbed depth values.
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # (N_rays, N_samples, 3)
            
        # Reshapes these 3D positions into a two-dimensional array for processing.
        pointsf = pts.reshape(-1, 3)
        
        # Evaluates these points using the eval_points method, 
        # which likely queries the neural implicit model for properties like occupancy or color at these points.
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        sigma = raw[:, -1]
        
        # Returns the computed volume densities, which indicate how much geometry is present at each sampled point along the rays.
        return sigma
