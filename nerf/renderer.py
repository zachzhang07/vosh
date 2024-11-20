import os
import cv2
import math
import json
import tqdm
import trimesh
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_efficient_distloss import eff_distloss

import nvdiffrast.torch as dr
import xatlas
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion

from meshutils import *
from nerf import math
from nerf import coord
from nerf import schedule
from nerf import grid_utils
from nerf import quantize

TORCH_SCATTER = None  # lazy import
import aspose.threed as a3d


def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[
        1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]


def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]


# @torch.cuda.amp.autocast(enabled=False)
# def sparsity_loss_vol_loss(alphas, xyzs, voxel_error_grid):
#     cube_idx = (torch.floor((xyzs + 2.0) / (2 * 2.0) * voxel_error_grid.shape[0]).long()
#                 .clamp(0, voxel_error_grid.shape[0] - 1))
#     pix_mult = voxel_error_grid[cube_idx[:, 0], cube_idx[:, 1], cube_idx[:, 2]]
#     loss = (torch.log(1 + 2 * (alphas ** 2)) * pix_mult.unsqueeze(-1)).mean()
#     return loss


@torch.cuda.amp.autocast(enabled=False)
def yu_sparsity_loss(random_positions, random_viewdirs, density, voxel_size, contract):
    step_size = coord.stepsize_in_squash(
        random_positions, random_viewdirs, voxel_size, contract
    )
    return 1.0 - torch.exp(-step_size * density).mean()


# def sparsity_loss(alphas, mask_mesh, beta=0.8):
#     pix_mult = mask_mesh * beta + (~mask_mesh) * (1 - beta)
#     loss = (torch.log(1 + 2 * (alphas ** 2)) * pix_mult.unsqueeze(-1)).mean()
#     return loss


@torch.cuda.amp.autocast(enabled=False)
def distort_loss(bins, weights):
    # bins: [N, T+1]
    # weights: [N, T]

    intervals = bins[..., 1:] - bins[..., :-1]
    mid_points = bins[..., :-1] + intervals / 2

    loss = eff_distloss(weights, mid_points, intervals)

    return loss


@torch.cuda.amp.autocast(enabled=False)
def proposal_loss(all_bins, all_weights):
    # all_bins: list of [N, T+1]
    # all_weights: list of [N, T]

    def loss_interlevel(t0, w0, t1, w1):
        # t0, t1: [N, T+1]
        # w0, w1: [N, T]
        cw1 = torch.cat([torch.zeros_like(w1[..., :1]), torch.cumsum(w1, dim=-1)], dim=-1)
        inds_lo = (torch.searchsorted(t1[..., :-1].contiguous(),
                                      t0[..., :-1].contiguous(), right=True) - 1).clamp(0, w1.shape[-1] - 1)
        inds_hi = (torch.searchsorted(t1[..., 1:].contiguous(), t0[..., 1:].contiguous(), right=True)
                   .clamp(0, w1.shape[-1] - 1))

        cw1_lo = torch.take_along_dim(cw1[..., :-1], inds_lo, dim=-1)
        cw1_hi = torch.take_along_dim(cw1[..., 1:], inds_hi, dim=-1)
        w = cw1_hi - cw1_lo

        return (w0 - w).clamp(min=0) ** 2 / (w0 + 1e-8)

    bins_ref = all_bins[-1].detach()
    weights_ref = all_weights[-1].detach()
    loss = 0
    for bins, weights in zip(all_bins[:-1], all_weights[:-1]):
        loss += loss_interlevel(bins_ref, weights_ref, bins, weights).mean()

    return loss


# MeRF-like contraction
@torch.cuda.amp.autocast(enabled=False)
def contract(x):
    # x: [..., C]
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = x.abs().max(1, keepdim=True)  # [N, 1], [N, 1]
    scale = 1 / mag.repeat(1, C)
    scale.scatter_(1, idx, (2 - 1 / mag) / mag)
    z = torch.where(mag < 1, x, x * scale)
    return z.view(*shape, C)


@torch.cuda.amp.autocast(enabled=False)
def uncontract(z):
    # z: [..., C]
    shape, C = z.shape[:-1], z.shape[-1]
    z = z.view(-1, C)
    mag, idx = z.abs().max(1, keepdim=True)  # [N, 1], [N, 1]
    scale = 1 / (2 - mag.repeat(1, C)).clamp(min=1e-8)
    scale.scatter_(1, idx, 1 / (2 * mag - mag * mag).clamp(min=1e-8))
    x = torch.where(mag < 1, z, z * scale)
    return x.view(*shape, C)


# @torch.cuda.amp.autocast(enabled=False)
# def contract(xyzs):
#     if isinstance(xyzs, np.ndarray):
#         mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
#         xyzs = np.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
#     else:
#         mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
#         xyzs = torch.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
#     return xyzs
#
#
# @torch.cuda.amp.autocast(enabled=False)
# def uncontract(xyzs):
#     if isinstance(xyzs, np.ndarray):
#         mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
#         xyzs = np.where(mag <= 1, xyzs, xyzs * (1 / (2 * mag - mag * mag)))
#     else:
#         mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
#         xyzs = torch.where(mag <= 1, xyzs, xyzs * (1 / (2 * mag - mag * mag)))
#     return xyzs


@torch.cuda.amp.autocast(enabled=False)
def sample_pdf(bins, weights, T, perturb=False):
    # bins: [N, T0+1]
    # weights: [N, T0]
    # return: [N, T]

    N, T0 = weights.shape
    weights = weights + 0.01  # prevent NaNs
    weights_sum = torch.sum(weights, -1, keepdim=True)  # [N, 1]
    pdf = weights / weights_sum
    cdf = torch.cumsum(pdf, -1).clamp(max=1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [N, T+1]

    u = torch.linspace(0.5 / T, 1 - 0.5 / T, steps=T).to(weights.device)
    u = u.expand(N, T)

    if perturb:
        u = u + (torch.rand_like(u) - 0.5) / T

    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)  # [N, t]

    below = torch.clamp(inds - 1, 0, T0)
    above = torch.clamp(inds, 0, T0)

    cdf_g0 = torch.gather(cdf, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g0 = torch.gather(bins, -1, below)
    bins_g1 = torch.gather(bins, -1, above)

    bins_t = torch.clamp(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0)), 0, 1)  # [N, t]
    bins = bins_g0 + bins_t * (bins_g1 - bins_g0)  # [N, t]

    return bins


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.05):
    # rays: [N, 3], [N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [N, 1], far [N, 1]

    tmin = (aabb[:3] - rays_o) / (rays_d + 1e-15)  # [N, 3]
    tmax = (aabb[3:] - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).amax(dim=-1, keepdim=True)
    far = torch.where(tmin > tmax, tmin, tmax).amin(dim=-1, keepdim=True)
    # if far < near, means no intersection, set both near and far to inf (1e9 here)
    mask = far < near
    near[mask] = 1e9
    far[mask] = 1e9
    # restrict near to a minimal value
    near = torch.clamp(near, min=min_near)

    return near, far


def erode(img, ksize=5):
    pad = (ksize - 1) // 2
    img = F.pad(img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    out = F.max_pool2d(1 - img, kernel_size=ksize, stride=1, padding=0)
    return 1 - out


def dilate(img, ksize=5):
    pad = (ksize - 1) // 2
    img = F.pad(img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    out = F.max_pool2d(img, kernel_size=ksize, stride=1, padding=0)
    return out


def simulate_alpha_culling(density, positions, viewdirs, alpha_threshold, voxel_size_to_use):
    """Computes the alpha value based on a constant step size."""

    # During real-time rendering, a constant step size (i.e., voxel size) is used.
    # During training, a variable step size is used that can be vastly different
    # from the voxel size.
    #
    # When baking, we discard voxels that would only contribute negligible
    # alpha values in the real-time renderer.
    #
    # To make this lossless, we already simulate the behavior of the real-time renderer during
    # training by ignoring alpha values below the threshold.

    def zero_density_below_threshold(_density):
        viewdirs_b = torch.broadcast_to(
            viewdirs[..., None, :], positions.shape
        ).reshape(-1, 3)
        positions_b = positions.reshape(-1, 3)
        step_size_uncontracted = coord.stepsize_in_squash(
            positions_b, viewdirs_b, voxel_size_to_use, contract
        )
        step_size_uncontracted = step_size_uncontracted.reshape(_density.shape)
        alpha = math.density_to_alpha(_density, step_size_uncontracted)
        return torch.where(alpha >= alpha_threshold, _density,
                           torch.zeros_like(_density))  # density = 0 <=> alpha = 0

    if alpha_threshold and alpha_threshold > 0.0:
        return zero_density_below_threshold(density)
    else:
        return density


class NeRFRenderer(nn.Module):
    def __init__(self, opt, device=None):
        super().__init__()

        self.opt = opt

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # bound for ray marching (world space)
        self.real_bound = opt.bound

        # bound for grid querying
        if self.opt.contract:
            self.bound = min(2, opt.bound)
        else:
            self.bound = opt.bound

        self.cascade_list = self.opt.cascade_list if 'cascade_list' in self.opt else None

        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb_train = torch.FloatTensor(
            [-self.real_bound, -self.real_bound, -self.real_bound, self.real_bound, self.real_bound, self.real_bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = opt.cuda_ray

        if opt.render == 'grid':
            if self.opt.alpha_thres == 0:
                self.alpha_threshold = None
            else:
                self.alpha_threshold = schedule.LogLerpSchedule(
                    start=10000, end=20000, v0=self.opt.alpha_thres * 0.1, v1=self.opt.alpha_thres,
                    zero_before_start=True
                )  # Multiplier for alpha-culling-simulation loss.
        else:
            self.glctx = dr.RasterizeGLContext()

            # if opt.render == 'mesh':
            #     print('[INFO] Clean env_mesh for mesh refine...')
            #     _updated_mesh_path = os.path.join(self.opt.vol_path, 'mesh', f'mesh_2.0_updated.ply')
            #     if not os.path.exists(_updated_mesh_path):
            #         mesh = trimesh.load(os.path.join(self.opt.vol_path, 'mesh', f'mesh_2.0.ply'), force='mesh',
            #                             skip_material=True, process=False)
            #
            #         v, t = mesh.vertices, mesh.faces
            #         # remove the faces which have the length over self.opt.max_edge_len
            #         v, t = remove_selected_vt_by_edge_length(v, t, self.opt.max_edge_len)
            #
            #         # remove the isolated component composed by a limited number of triangles
            #         v, t = remove_selected_isolated_faces(v, t, self.opt.min_iso_size)
            #
            #         mesh = trimesh.Trimesh(v, t, process=False)
            #         mesh.export(os.path.join(self.opt.vol_path, 'mesh', f'mesh_2.0_updated.ply'))

            # sequentially load cascaded meshes
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]

            # if 'all_mesh' in self.opt and self.opt.all_mesh:
            #     name_list = ['_refined', '']
            # else:
            #     name_list = ['_selected', '_refined', '']
            name_list = ['_refined', '']

            path = None
            for name in name_list:
                if os.path.exists(os.path.join(opt.workspace, 'mesh', f'mesh_all{name}.ply')):
                    path = os.path.join(opt.workspace, 'mesh', f'mesh_all{name}.ply')
                    break
            if path is None:
                for name in name_list:
                    if os.path.exists(os.path.join(opt.vol_path, 'mesh', f'mesh_all{name}.ply')):
                        path = os.path.join(opt.vol_path, 'mesh', f'mesh_all{name}.ply')
                        os.makedirs(os.path.join(opt.workspace, 'mesh'), exist_ok=True)
                        shutil.copy(os.path.join(opt.vol_path, 'mesh', f'mesh_all{name}.ply'),
                                    os.path.join(opt.workspace, 'mesh', f'mesh_all{name}.ply'))
                        break
            assert path is not None, f'Cannot find mesh_all*.ply in {opt.workspace} and {opt.vol_path}'

            mesh = trimesh.load(path, force='mesh', skip_material=True, process=False)
            v, t = mesh.vertices, mesh.faces
            print(f'[INFO] loader mesh from {path}: {v.shape}, {t.shape}')
            m = pml.Mesh(v, t)
            ms = pml.MeshSet()
            ms.add_mesh(m, 'mesh')

            print("dataset format:{}".format(self.opt.data_format))
            if (self.opt.data_format != 'nerf'):
                ms.compute_selection_by_condition_per_vertex(
                    condselect='(x < 1.0) && (x > -1.0) && (y < 1.0) && (y > -1.0) && (z < 1.0) && (z > -1.0)')
            else:
                ms.compute_selection_by_condition_per_vertex(
                    condselect='(x < 2.0) && (x > -2.0) && (y < 2.0) && (y > -2.0) && (z < 2.0) && (z > -2.0)')
            ms.compute_selection_transfer_vertex_to_face()
            ms.generate_from_selected_faces()

            ms.set_current_mesh(1)
            m = ms.current_mesh()
            inner_v, inner_f = m.vertex_matrix(), m.face_matrix()
            vertices.append(inner_v)
            triangles.append(inner_f + v_cumsum[-1])
            v_cumsum.append(v_cumsum[-1] + inner_v.shape[0])
            f_cumsum.append(f_cumsum[-1] + inner_f.shape[0])
            print(f'[INFO] inner mesh: {inner_v.shape}, {inner_f.shape}')

            ms.set_current_mesh(0)
            m = ms.current_mesh()
            outer_v, outer_f = m.vertex_matrix(), m.face_matrix()
            vertices.append(outer_v)
            triangles.append(outer_f + v_cumsum[-1])
            v_cumsum.append(v_cumsum[-1] + outer_v.shape[0])
            f_cumsum.append(f_cumsum[-1] + outer_f.shape[0])
            print(f'[INFO] outer mesh: {outer_v.shape}, {outer_f.shape}')

            # name_list = ['_selected', '_final', '_updated', '']
            #
            # for cas in self.cascade_list:
            #     mesh_name = None
            #     for name in name_list:
            #         if os.path.exists(os.path.join(opt.workspace, 'mesh', f'mesh_{cas}{name}.ply')):
            #             mesh_name = f'mesh_{cas}{name}.ply'
            #             break
            #     # if os.path.exists(os.path.join(opt.workspace, 'mesh', f'mesh_{cas}_final.ply')):
            #     #     mesh_name = f'mesh_{cas}_final.ply'
            #     # elif os.path.exists(os.path.join(opt.workspace, 'mesh', f'mesh_{cas}_updated.ply')):
            #     #     mesh_name = f'mesh_{cas}_updated.ply'
            #     # elif os.path.exists(os.path.join(opt.workspace, 'mesh', f'mesh_{cas}.ply')):
            #     #     mesh_name = f'mesh_{cas}.ply'
            #     # else:
            #     #     if os.path.exists(os.path.join(opt.vol_path, 'mesh', f'mesh_{cas}_final.ply')):
            #     #         mesh_name = f'mesh_{cas}_final.ply'
            #     #     elif os.path.exists(os.path.join(opt.vol_path, 'mesh', f'mesh_{cas}_updated.ply')):
            #     #         mesh_name = f'mesh_{cas}_updated.ply'
            #     #     else:  # base (not updated)
            #     #         mesh_name = f'mesh_{cas}.ply'
            #     if mesh_name is None:
            #         for name in name_list:
            #             if os.path.exists(os.path.join(opt.vol_path, 'mesh', f'mesh_{cas}{name}.ply')):
            #                 mesh_name = f'mesh_{cas}{name}.ply'
            #                 break
            #         os.makedirs(os.path.join(self.opt.workspace, 'mesh'), exist_ok=True)
            #         shutil.copy(os.path.join(self.opt.vol_path, 'mesh', mesh_name),
            #                     os.path.join(self.opt.workspace, 'mesh', mesh_name))
            #     mesh = trimesh.load(os.path.join(self.opt.workspace, 'mesh', mesh_name),
            #                         force='mesh', skip_material=True, process=False)
            #     v, t = mesh.vertices, mesh.faces
            #
            #     if opt.render == 'mesh':
            #         # decimation
            #         decimate_target = self.opt.decimate_target[len(vertices)]
            #         if 0 < decimate_target < t.shape[0]:
            #             v, t = decimate_mesh(v, t, decimate_target, remesh=False)
            #             m = trimesh.Trimesh(v, t, process=False)
            #             os.makedirs(os.path.join(self.opt.workspace, f'mesh_{int(decimate_target)}'), exist_ok=True)
            #             s_path = os.path.join(self.opt.workspace, f'mesh_{int(decimate_target)}', mesh_name)
            #             m.export(s_path)
            #
            #     print(f'[INFO] loaded {mesh_name}: {v.shape}, {t.shape}')
            #
            #     vertices.append(v)
            #     triangles.append(t + v_cumsum[-1])
            #
            #     v_cumsum.append(v_cumsum[-1] + v.shape[0])
            #     f_cumsum.append(f_cumsum[-1] + t.shape[0])

            vertices = np.concatenate(vertices, axis=0)
            triangles = np.concatenate(triangles, axis=0)

            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)

            self.vertices = torch.from_numpy(vertices).float().contiguous().to(self.device)
            self.triangles = torch.from_numpy(triangles).int().contiguous().to(self.device)
            print(f'[INFO] total mesh: {self.vertices.shape}, {self.triangles.shape}')

            # learnable offsets for mesh vertex
            self.vertices_offsets = None
            if self.opt.vert_offset:
                self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

            # accumulate error for mesh face
            if self.opt.render == 'mesh' and self.opt.refine:
                self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).to(self.device)
                self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).to(self.device)

            self.error_grid = None
            self.occ_grid = None

            if self.opt.render == 'mesh' and self.opt.vis_error:
                self.error_grid = torch.load(os.path.join(self.opt.workspace, 'error_grid.pt')).to(self.device)
                self.error_grid.requires_grad = False

            if self.opt.render == 'mixed':
                # self.mesh_occ_mask = torch.zeros([opt.grid_resolution] * 3, dtype=torch.bool,
                #                                  device=self.device, requires_grad=False)
                self.mesh_occ_mask = None

                if self.opt.alpha_thres == 0:
                    self.alpha_threshold = None
                else:
                    self.alpha_threshold = schedule.LogLerpSchedule(
                        start=0, end=self.opt.iters // 2, v0=0.005, v1=self.opt.alpha_thres,
                        zero_before_start=True)  # Multiplier for alpha-culling-simulation loss.

                # if self.opt.density_scale > 0:
                #     self.density_scale = nn.Parameter(
                #         self.opt.density_scale * torch.ones([512] * 3, dtype=torch.float32, device=self.device))

    def get_params(self):
        params = []
        if self.opt.render != 'grid' and self.opt.vert_offset:
            params.append({'params': self.vertices_offsets, 'lr': self.opt.lr_vert, 'weight_decay': 0})

        return params

    def set_training(self, flag):
        self.training = flag

    def forward(self, x, d, **kwargs):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x, **kwargs):
        raise NotImplementedError()

    # @torch.no_grad()
    # def update_mesh_occ_mask(self, loader, resolution=None):
    #     self.set_training(False)
    #     if resolution is None:
    #         resolution = self.opt.grid_resolution
    #     mesh_occ_grid = torch.zeros([resolution] * 3, dtype=torch.bool, device=self.device)
    #
    #     # mesh_occ_grid = self.pos_to_cube_cnt(mesh_occ_grid, contract(self.triangles_center), 1, accumulate=False)
    #
    #     print('Update mesh_occ_mask using rasterization...')
    #     for i, data in enumerate(tqdm.tqdm(loader)):
    #         results_mesh = self.render_mesh(rays_o=None, rays_d=data['rays_d_all'],
    #                                         h0=data['H'], w0=data['W'], mvp=data['mvp'], only_xyzs=True)
    #         if results_mesh['mask'].any():
    #             # [the number of 1 in mask, 3] in (-2, 2)
    #             xyzs = contract(results_mesh['xyzs'][results_mesh['mask']])
    #             mesh_occ_grid = self.pos_to_cube_cnt(mesh_occ_grid, xyzs, 1, accumulate=False)
    #
    #     self.mesh_occ_mask = mesh_occ_grid > 0
    #     self.set_training(True)

    def set_error_grid(self, error_grid):
        self.error_grid = error_grid

    @torch.no_grad()
    def update_mesh_occ_mask(self, loader):
        self.set_training(False)
        resolution = self.opt.grid_resolution // self.opt.mesh_check_ratio
        self.mesh_occ_mask = torch.zeros([resolution] * 3, dtype=torch.bool, device=self.device)

        print(f'Update mesh_occ_mask using rasterization using resolution {resolution}...')
        for i, data in enumerate(tqdm.tqdm(loader)):
            results_mesh = self.render_mesh(rays_o=None, rays_d=data['rays_d_all'],
                                            h0=data['H'], w0=data['W'], mvp=data['mvp'], only_xyzs=True)
            if results_mesh['mask'].any():
                # [the number of 1 in mask, 3] in (-2, 2)
                if self.opt.contract:
                    xyzs = contract(results_mesh['xyzs'][results_mesh['mask']])
                else:
                    xyzs = results_mesh['xyzs'][results_mesh['mask']]
                coords = (torch.floor((xyzs + self.bound) / (2 * self.bound) * resolution)
                          .long().clamp(0, resolution - 1))  # [N*T, 3]
                self.mesh_occ_mask[tuple(coords.T)] = 1

        self.set_training(True)

    # @torch.no_grad()
    # def pos_to_cube_cnt(self, cube_cnt, pos, value, accumulate=True):
    #     # pos: contracted position in [-2, 2]
    #     assert pos.max() <= self.bound and pos.min() >= -self.bound, (
    #         print(f'pos.max(): {pos.max()}, pos.min():{pos.min()}'))
    #     if not isinstance(value, torch.Tensor):
    #         value = torch.ones(pos.shape[0], dtype=cube_cnt.dtype, device=cube_cnt.device) * value
    #     # cube_cnt = torch.ones([self.opt.mcubes_reso] * 3, dtype=torch.float32, device=pos.device)
    #     cube_idx = (torch.floor((pos + self.bound) / (2 * self.bound) *
    #                             cube_cnt.shape[0]).long().clamp(0, cube_cnt.shape[0] - 1))
    #     cube_cnt = cube_cnt.index_put((cube_idx[:, 0], cube_idx[:, 1], cube_idx[:, 2]), value, accumulate)
    #     return cube_cnt

    @torch.no_grad()
    def export_mesh_after_refine(self):
        if self.vertices_offsets is not None:
            v = (self.vertices + self.vertices_offsets).detach().cpu().numpy()
        else:
            v = self.vertices.detach().cpu().numpy()
        f = self.triangles.detach().cpu().numpy()

        mesh = trimesh.Trimesh(v, f, process=False)
        path = os.path.join(self.opt.workspace, 'mesh', f'mesh_all_refined.ply')
        mesh.export(path)

    @torch.no_grad()
    def refine_and_decimate(self):
        device = self.vertices.device

        if self.vertices_offsets is not None:
            v = (self.vertices + self.vertices_offsets).detach().cpu().numpy()
        else:
            v = self.vertices.detach().cpu().numpy()
        f = self.triangles.detach().cpu().numpy()

        errors = self.triangles_errors.cpu().numpy()

        cnt = self.triangles_errors_cnt.cpu().numpy()
        cnt_mask = cnt > 0
        errors[cnt_mask] = errors[cnt_mask] / cnt[cnt_mask]

        # only care about the inner mesh
        errors = errors[:self.f_cumsum[1]]
        cnt_mask = cnt_mask[:self.f_cumsum[1]]

        # find a threshold to decide whether we perform subdivision / decimation.
        thresh_refine = np.percentile(errors[cnt_mask], 90)
        thresh_decimate = np.percentile(errors[cnt_mask], 50)

        mask = np.zeros_like(errors)
        sharp_faces_mask, flat_faces_mask = None, None

        if not self.opt.no_normal:
            # assign mask values according to the shape(normal, folded?)
            # mesh_tmp = trimesh.Trimesh(v[:self.v_cumsum[1]], f[:self.f_cumsum[1]], process=False)
            # face_adjacency = mesh_tmp.face_adjacency
            # face_adjacency_angles = mesh_tmp.face_adjacency_angles
            # normal_change = torch.zeros_like(self.triangles_errors)
            # face_adjacency_cnt = torch.zeros_like(self.triangles_errors)
            # for i in range(face_adjacency.shape[0]):
            #     # print('face_adjacency[i]: ', face_adjacency[i])
            #     # print('face_adjacency_angles[i]: ', face_adjacency_angles[i])
            #     normal_change[face_adjacency[i, 0]] += face_adjacency_angles[i]
            #     normal_change[face_adjacency[i, 1]] += face_adjacency_angles[i]
            #     face_adjacency_cnt[face_adjacency[i, 0]] += 1
            #     face_adjacency_cnt[face_adjacency[i, 1]] += 1
            # face_adjacency_cnt[face_adjacency_cnt == 0] = 1.0
            # normal_change = (normal_change / face_adjacency_cnt).cpu().numpy()[:self.f_cumsum[1]]
            #
            # thresh_refine_normal = np.percentile(normal_change[cnt_mask], 90)
            # thresh_decimate_normal = np.percentile(normal_change[cnt_mask], 10)
            # mask[(errors > thresh_refine) | (normal_change > thresh_refine_normal) & cnt_mask] = 2
            # # mask[(errors < thresh_decimate) | (normal_change < thresh_decimate_normal) & cnt_mask] = 1
            # mask[(errors < thresh_decimate) & cnt_mask] = 1

            # torch-vosh-new 使用的版本
            # sharp_faces_mask, flat_faces_mask = select_bad_and_flat_faces_by_normal(
            # v[:self.v_cumsum[1]], f[:self.f_cumsum[1]])

            sharp_faces_mask, flat_faces_mask = select_sharp_and_flat_faces_by_normal_using_ratio(
                v[:self.v_cumsum[1]], f[:self.f_cumsum[1]], sharp_ratio=self.opt.sharp_ratio,
                flat_ratio=self.opt.flat_ratio)
            mask[(errors > thresh_refine) | sharp_faces_mask & cnt_mask] = 2
            mask[(errors < thresh_decimate) | flat_faces_mask & cnt_mask] = 1

        else:
            mask[(errors > thresh_refine) & cnt_mask] = 2
            mask[(errors < thresh_decimate) & cnt_mask] = 1

        if self.opt.no_mesh_refine:
            mask[mask == 2] = 0
        
        if self.opt.no_mesh_decimate:
            mask[mask == 1] = 0

        # print(f'[INFO] faces to decimate {(mask == 1).sum()}, faces to refine {(mask == 2).sum()}')
        # if sharp_faces_mask is not None:
        #     print(f'[INFO] faces to refine in error {((errors > thresh_refine) & cnt_mask).sum()}')
        #     print(f'[INFO] faces to refine in normal {(sharp_faces_mask & cnt_mask).sum()}')
        #     print(f'[INFO] faces to decimate in error {((errors < thresh_decimate) & cnt_mask).sum()}')
        #     print(f'[INFO] faces to decimate in normal {(flat_faces_mask & cnt_mask).sum()}')

        vertices = []
        triangles = []
        v_cumsum = [0]
        f_cumsum = [0]

        inner_v, inner_f = v[self.v_cumsum[0]:self.v_cumsum[1]], f[self.f_cumsum[0]:self.f_cumsum[1]] - self.v_cumsum[0]
        outer_v, outer_f = v[self.v_cumsum[1]:self.v_cumsum[2]], f[self.f_cumsum[1]:self.f_cumsum[2]] - self.v_cumsum[1]
        _mesh = trimesh.Trimesh(inner_v, inner_f, process=False)
        inner_v, inner_f = decimate_and_refine_mesh(inner_v, inner_f, mask,
                                                    decimate_ratio=self.opt.refine_decimate_ratio,
                                                    refine_size=self.opt.refine_size,
                                                    refine_remesh_size=self.opt.refine_remesh_size)
        vertices.append(inner_v)
        triangles.append(inner_f + v_cumsum[-1])
        v_cumsum.append(v_cumsum[-1] + inner_v.shape[0])
        f_cumsum.append(f_cumsum[-1] + inner_f.shape[0])
        vertices.append(outer_v)
        triangles.append(outer_f + v_cumsum[-1])
        v_cumsum.append(v_cumsum[-1] + outer_v.shape[0])
        f_cumsum.append(f_cumsum[-1] + outer_f.shape[0])

        # for cas in range(len(self.cascade_list)):
        #
        #     cur_v = v[self.v_cumsum[cas]:self.v_cumsum[cas + 1]]
        #     cur_f = f[self.f_cumsum[cas]:self.f_cumsum[cas + 1]] - self.v_cumsum[cas]
        #
        #     if cas == 0:
        #         _mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
        #         cur_v, cur_f = decimate_and_refine_mesh(cur_v, cur_f, mask,
        #                                                 decimate_ratio=self.opt.refine_decimate_ratio,
        #                                                 refine_size=self.opt.refine_size,
        #                                                 refine_remesh_size=self.opt.refine_remesh_size)
        #
        #     # cur_v, cur_f = close_holes_meshfix(cur_v, cur_f)  # close holes in mesh
        #     mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
        #     if cas == 0:
        #         mesh.export(os.path.join(self.opt.workspace, 'mesh', f'mesh_{self.cascade_list[cas]}_updated.ply'))
        #
        #         # cur_v, cur_f = close_holes_meshfix(cur_v, cur_f)  # close holes in mesh
        #         # _mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
        #         # _mesh.export(os.path.join(self.opt.workspace, 'mesh', f'mesh_'
        #         #                                                       f'{self.cascade_list[cas]}_updated_fix.ply'))
        #
        #     vertices.append(mesh.vertices)
        #     triangles.append(mesh.faces + v_cumsum[-1])
        #
        #     v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
        #     f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])

        v = np.concatenate(vertices, axis=0)
        f = np.concatenate(triangles, axis=0)
        self.v_cumsum = np.array(v_cumsum)
        self.f_cumsum = np.array(f_cumsum)

        self.vertices = torch.from_numpy(v).float().contiguous().to(device)  # [N, 3]
        self.triangles = torch.from_numpy(f).int().contiguous().to(device)

        if self.vertices_offsets is not None:
            self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))
        else:
            self.vertices_offsets = None

        self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)
        self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)

        print(f'[INFO] update mesh: {self.vertices.shape}, {self.triangles.shape}')

    def render(self, rays_o, rays_d, cam_near_far=None, shading='full', update_proposal=False, **kwargs):
        N = rays_o.shape[0]
        device = rays_o.device

        if self.training:
            return self.run(rays_o, rays_d, shading=shading, update_proposal=update_proposal, **kwargs)
        else:  # staged inference
            head = 0
            results = {}
            while head < N:
                # if self.opt.render != 'mesh':
                #     tail = min(head + self.opt.max_ray_batch, N)
                # else:
                #     tail = N
                tail = min(head + self.opt.max_ray_batch, N)

                if rays_o.shape[0] != self.opt.num_rays and self.opt.render == 'mixed':
                    assert kwargs['H'] * kwargs['W'] == rays_o.shape[0]
                    idx = torch.linspace(head, tail - 1, tail - head, device=device).long()
                    kwargs['rays_j'] = torch.div(idx, kwargs['W'], rounding_mode='trunc')
                    kwargs['rays_i'] = idx % kwargs['W']

                if cam_near_far is None:
                    results_ = self.run(rays_o[head:tail], rays_d[head:tail], cam_near_far=None,
                                        update_proposal=False, **kwargs)
                elif cam_near_far.shape[0] == 1:
                    results_ = self.run(rays_o[head:tail], rays_d[head:tail], cam_near_far=cam_near_far,
                                        update_proposal=False, **kwargs)
                else:
                    results_ = self.run(rays_o[head:tail], rays_d[head:tail], cam_near_far=cam_near_far[head:tail],
                                        update_proposal=False, **kwargs)

                for k, v in results_.items():
                    if k not in results:
                        results[k] = torch.empty(N, *v.shape[1:], device=device)
                    results[k][head:tail] = v
                head += self.opt.max_ray_batch

            return results

    def render_mesh(self, rays_o, rays_d, mvp, h0, w0, shading='full', rays_j=None, rays_i=None,
                    bg_color=None, **kwargs):

        # do super-sampling
        h = int(h0 * self.opt.ssaa)
        w = int(w0 * self.opt.ssaa)

        # mix background color
        if bg_color is None:
            bg_color = 1

        results = {}

        if self.vertices_offsets is not None:
            vertices = self.vertices + self.vertices_offsets
        else:
            vertices = self.vertices

        triangles = self.triangles

        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0),
                                     torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]

        # do rasterization in original resolution to get xyzs and mask
        rast, _ = dr.rasterize(self.glctx, vertices_clip, triangles, (h0, w0))
        xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, triangles)  # [1, h0, w0, 3]
        mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, triangles)  # [1, h0, w0, 1]
        results['xyzs'] = xyzs[:, rays_j, rays_i].view(-1, 3) if rays_j is not None else xyzs.view(-1, 3)
        results['mask'] = mask[:, rays_j, rays_i].view(-1) > 0 if rays_j is not None else mask.view(-1) > 0
        if 'only_xyzs' in kwargs.keys() and kwargs['only_xyzs']:
            return results

        mask_flatten = None
        if self.opt.ssaa > 1:
            rast, _ = dr.rasterize(self.glctx, vertices_clip, triangles, (h, w))
            xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, triangles)  # [1, H, W, 3]
            mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, triangles)  # [1, H, W, 1]
            if rays_j is not None:
                mask_i_j = torch.zeros((1, h0, w0, 1)).to(mask)
                mask_i_j[0, rays_j, rays_i] = 1.0
                mask_i_j = dilate((scale_img_nhwc(mask_i_j, (h, w)) > 0).float())
                mask_flatten = ((mask_i_j > 0) * (mask > 0)).view(-1).detach()

        mask_flatten = (mask > 0).view(-1).detach() if mask_flatten is None else mask_flatten
        xyzs = xyzs.view(-1, 3)

        # random noise to make appearance more robust
        if self.training:
            xyzs = xyzs + torch.randn_like(xyzs) * 1e-3

        assert shading == 'full'
        features = torch.zeros(h * w, 7, dtype=torch.float32).to(mvp)

        if mask_flatten.any():
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                # mask_output = self(contract(xyzs[mask_flatten].detach()), None, shading)
                # mask_features, _ = self.forward(contract(xyzs[mask_flatten].detach()), only_feat=True)
                # if self.opt.contract:
                #     mask_features, _ = self.DensityAndFeaturesMLP(contract(xyzs[mask_flatten].detach()),
                #                                                   bound=self.bound)
                # else:
                #     mask_features, _ = self.DensityAndFeaturesMLP(xyzs[mask_flatten].detach(), bound=self.bound)
                if self.opt.contract:
                    xyzs = contract(xyzs[mask_flatten].detach())
                else:
                    xyzs = xyzs[mask_flatten].detach()
                # if 'mesh_encoder' in kwargs.keys() and kwargs['mesh_encoder']:
                if 'mesh_encoder' in self.opt and self.opt.mesh_encoder:
                    mask_features, _ = self.DensityAndFeaturesMLP_mesh(xyzs, bound=self.bound)
                else:
                    mask_features, _ = self.DensityAndFeaturesMLP(xyzs, bound=self.bound)

                mask_features = quantize.simulate_quantization(
                    mask_features, self.range_features[0], self.range_features[1]
                )
                mask_features = torch.sigmoid(mask_features)

            features[mask_flatten] = mask_features.float()

        features = features.view(1, h, w, 7)
        diffuse, specular = features[..., :3].contiguous(), features[..., 3:].squeeze(0).contiguous()
        alphas = mask.float()

        alphas = dr.antialias(alphas, rast, vertices_clip, triangles,
                              pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        # alphas = alphas.squeeze(0).clamp(0, 1)
        diffuse = dr.antialias(diffuse, rast, vertices_clip, triangles,
                               pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        # diffuse = alphas * diffuse
        # diffuse = diffuse.squeeze(0).clamp(0, 1)

        depth = alphas * rast[0, :, :, [2]]
        T = 1 - alphas
        # diffuse = diffuse + T * bg_color

        # trig_id for updating trig errors
        trig_id = (rast[0, :, :, -1] - 1).float()  # [h, w]

        # ssaa
        if self.opt.ssaa:
            # image = scale_img_hwc(image, (h0, w0))
            diffuse = scale_img_hwc(diffuse, (h0, w0))
            specular = scale_img_hwc(specular, (h0, w0))
            alphas = scale_img_hwc(alphas, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            T = scale_img_hwc(T, (h0, w0))
            trig_id = scale_img_hw(trig_id, (h0, w0), mag='nearest', min='nearest')

        features = torch.cat([diffuse, specular], dim=-1)

        if rays_j is not None:
            trig_id = trig_id[rays_j, rays_i]
            # image = image[rays_j, rays_i].view(-1, 3)
            features = features[rays_j, rays_i]
            alphas = alphas[rays_j, rays_i]
            depth = depth[rays_j, rays_i]
            T = T[rays_j, rays_i].view(-1)
            # prefix = rays_j.shape

        if self.training:
            self.triangles_errors_id = trig_id.view(-1)

        # image = image.view(*prefix, 3)
        # depth = depth.view(*prefix)

        results['depth'] = depth.view(-1)
        results['features'] = features.view(-1, 7)
        results['alphas'] = alphas.view(-1)

        # results['image'] = image
        results['weights_sum'] = 1 - T.view(-1)

        # tmp: visualize accumulated triangle error by abusing depth
        # error_val = self.triangles_errors[trig_id.view(-1)].view(*prefix)
        # error_cnt = self.triangles_errors_cnt[trig_id.view(-1)].view(*prefix)
        # cnt_mask = error_cnt > 0
        # error_val[cnt_mask] = error_val[cnt_mask] / error_cnt[cnt_mask].float()
        # results['depth'] = error_val

        return results

    @torch.no_grad()
    def remove_faces_in_selected_voxels(self, error_grid, target_resolution=32, mesh_select=0.7, keep_center=0.25):
        error_grid = F.max_pool3d(error_grid.float().unsqueeze(0).unsqueeze(0),
                                  error_grid.shape[0] // target_resolution,
                                  stride=error_grid.shape[0] // target_resolution).squeeze(0).squeeze(0)
        if keep_center > 0:
            half_resolution = target_resolution // 2
            half_itv = int(keep_center * half_resolution)
            error_grid[half_resolution - half_itv:half_resolution + half_itv,
            half_resolution - half_itv:half_resolution + half_itv,
            half_resolution - half_itv:half_resolution + half_itv, ] = 0
        threshold = torch.quantile(error_grid[error_grid > 0].flatten(), mesh_select)
        print(f'voxels selected ratio: '
              f'{((error_grid > threshold).float().sum() / (error_grid > 0).float().sum()).item() * 100:.2f}%')
        error_grid = error_grid > threshold

        if self.vertices_offsets is not None:
            verts = self.vertices + self.vertices_offsets
        else:
            verts = self.vertices
        verts = verts.detach().cpu().numpy()
        faces = self.triangles.detach().cpu().numpy()
        triangles_center = (torch.tensor(trimesh.Trimesh(verts, faces, process=False).triangles_center)
                            .to(self.device))
        if self.opt.contract:
            coords = (torch.floor((contract(triangles_center) + self.bound) /
                                  (2 * self.bound) * target_resolution)
                      .long().clamp(0, target_resolution - 1))  # [N*T, 3]
        else:
            coords = (torch.floor((triangles_center + self.bound) /
                                  (2 * self.bound) * target_resolution)
                      .long().clamp(0, target_resolution - 1))  # [N*T, 3]
        error_mask = error_grid[tuple(coords.T)].float().cpu().numpy()

        # mesh = trimesh.Trimesh(verts, faces, process=False)
        # path = os.path.join(self.opt.workspace, 'mesh', f'mesh_origin.ply')
        # mesh.export(path)

        print(f'faces selected ratio: {error_mask.sum() / error_mask.shape[0] * 100:.2f}%')
        # verts, faces = remove_masked_trigs(verts=verts, faces=faces, mask=mask, dilation=0)

        verts, faces = remove_masked_trigs(verts=verts, faces=faces, mask=error_mask, dilation=0)

        mesh = trimesh.Trimesh(verts, faces, process=False)
        path = os.path.join(self.opt.workspace, 'mesh', f'mesh_all_selected_1.ply')
        mesh.export(path)

        # remove the faces which have the length over self.opt.max_edge_len
        verts, faces = remove_selected_vt_by_edge_length(verts, faces, self.opt.max_edge_len)

        # remove the isolated component composed by a limited number of triangles
        # verts, faces = remove_selected_isolated_faces(verts, faces, self.opt.min_iso_size)

        mesh = trimesh.Trimesh(verts, faces, process=False)
        path = os.path.join(self.opt.workspace, 'mesh', f'mesh_all_selected_2.ply')
        mesh.export(path)

        # reduce floaters by post-processing...
        verts, faces = clean_mesh(verts, faces, min_f=self.opt.clean_min_f,
                                  min_d=self.opt.clean_min_d, v_pct=0,
                                  repair=False, remesh=False)

        mesh = trimesh.Trimesh(verts, faces, process=False)
        path = os.path.join(self.opt.workspace, 'mesh', f'mesh_all_selected.ply')
        mesh.export(path)

        # for cas in range(len(self.cascade_list)):
        #     print(f'cas: {self.cascade_list[cas]}')
        #
        #     if error_mask[self.f_cumsum[cas]:self.f_cumsum[cas + 1]].sum() > 0:
        #         cur_v, cur_f = remove_masked_trigs(verts=verts[self.v_cumsum[cas]:self.v_cumsum[cas + 1]],
        #                                            faces=faces[self.f_cumsum[cas]:self.f_cumsum[cas + 1]]
        #                                                  - self.v_cumsum[cas],
        #                                            mask=error_mask[self.f_cumsum[cas]:self.f_cumsum[cas + 1]],
        #                                            dilation=0)
        #         ## reduce floaters by post-processing...
        #         cur_v, cur_f = clean_mesh(cur_v, cur_f, min_f=self.opt.clean_min_f_base * cas,
        #                                   min_d=self.opt.clean_min_d,
        #                                   repair=False, remesh=False)
        #         mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
        #         path = os.path.join(self.opt.workspace, 'mesh', f'mesh_{self.cascade_list[cas]}_selected.ply')
        #         mesh.export(path)

        self.vertices = torch.from_numpy(verts).float().contiguous().to(self.device)
        self.triangles = torch.from_numpy(faces).int().contiguous().to(self.device)

        if self.opt.vert_offset:
            self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

        return verts, faces

    @torch.no_grad()
    def update_triangles_errors(self, loss, mask_mesh=None):
        # loss: [H, W], detached!

        # always call after render_stage1, so self.triangles_errors_id is not None.
        if mask_mesh is not None:
            self.triangles_errors_id = self.triangles_errors_id[mask_mesh]
        indices = self.triangles_errors_id.view(-1).long()

        mask = (indices >= 0)

        indices = indices[mask].contiguous()
        values = loss.view(-1)[mask].contiguous()

        global TORCH_SCATTER

        if TORCH_SCATTER is None:
            import torch_scatter
            TORCH_SCATTER = torch_scatter

        # print(f'\nvalues: {values.shape}, {values.min()}, {values.max()}')
        # print(f'indices: {indices.shape}, {indices.min()}, {indices.max()}')
        # print(f'self.triangles_errors: {self.triangles_errors.shape}, {self.triangles_errors.min()}, '
        #       f'{self.triangles_errors.max()}')

        TORCH_SCATTER.scatter_add(values, indices, out=self.triangles_errors)
        TORCH_SCATTER.scatter_add(torch.ones_like(values), indices, out=self.triangles_errors_cnt)

        self.triangles_errors_id = None

    def run_merf(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, shading='full',
                 update_proposal=True, baking=False, **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device

        # pre-calculate near far
        nears, fars = near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer,
                                         self.min_near)

        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, [0]])
            fars = torch.minimum(fars, cam_near_far[:, [1]])

        # mix background color
        if bg_color is None:
            bg_color = 1

        results = {}

        # hierarchical sampling
        if self.training:
            all_bins = []
            all_weights = []

        # sample xyzs using a mixed linear + lindisp function
        if self.opt.data_format == 'nerf':
            spacing_fn = lambda x: x
            spacing_fn_inv = lambda x: x
        else:
            spacing_fn = lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
            spacing_fn_inv = lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x))

        s_nears = spacing_fn(nears)  # [N, 1]
        s_fars = spacing_fn(fars)  # [N, 1]

        bins = None
        weights = None

        for prop_iter in range(len(self.opt.num_steps)):

            if prop_iter == 0:
                # uniform sampling
                # [1, T+1]
                bins = torch.linspace(0, 1, self.opt.num_steps[prop_iter] + 1, device=device).unsqueeze(0)
                bins = bins.expand(N, -1)  # [N, T+1]
                if perturb:
                    bins = bins + (torch.rand_like(bins) - 0.5) / (self.opt.num_steps[prop_iter])
                    bins = bins.clamp(0, 1)
            else:
                # pdf sampling
                bins = sample_pdf(bins, weights, self.opt.num_steps[prop_iter] + 1, perturb).detach()  # [N, T+1]

            real_bins = spacing_fn_inv(s_nears * (1 - bins) + s_fars * bins)  # [N, T+1] in [near, far]

            rays_t = (real_bins[..., 1:] + real_bins[..., :-1]) / 2  # [N, T]
            xyzs = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * rays_t.unsqueeze(2)  # [N, T, 3]

            if self.opt.contract:
                results['real_xyzs'] = xyzs  # in [-real_bound, real_bound]
                xyzs = contract(xyzs)

            if prop_iter != len(self.opt.num_steps) - 1:
                # query proposal density
                with torch.set_grad_enabled(update_proposal):
                    #  sigmas = self.density(xyzs, proposal=True)['sigma']  # [N, T]
                    sigmas = self.prop_mlp[0](self.prop_encoders[0]((xyzs + self.bound) / (2 * self.bound)))
                    sigmas = math.density_activation(sigmas).squeeze(-1)
            else:
                # last iter: query nerf
                # dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)  # [N, T, 3]
                dirs = rays_d.view(-1, 3)  # [N, 3]
                dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)

                # outputs = self(xyzs, dirs, shading=shading)
                # sigmas = outputs['sigma']
                # features = outputs['features']

                # features, sigmas = self.query_representation(xyzs)  # UxKx7 and UxKx1.
                features, sigmas = self.forward(xyzs, bound=self.bound)
                sigmas = sigmas.squeeze(-1)

                # alpha_threshold = self.alpha_threshold(kwargs['step'])
                if 'alpha_threshold' in kwargs.keys() and kwargs['alpha_threshold'] and kwargs['alpha_threshold'] > 0:
                    with torch.enable_grad():
                        sigmas = simulate_alpha_culling(sigmas, xyzs, dirs, kwargs['alpha_threshold'],
                                                        self.grid_config['voxel_size_to_use']).squeeze(-1)

            # sigmas to weights
            deltas = (real_bins[..., 1:] - real_bins[..., :-1])  # [N, T]
            # if self.opt.render == 'mixed' and self.opt.no_check is False:
            #     deltas_sigmas = deltas * sigmas * self.voxel_alive_check(xyzs, self.opt.use_occ_grid,
            #                                                              self.opt.use_mesh_occ_grid)  # [N, T]
            # else:
            #     deltas_sigmas = deltas * sigmas  # [N, T]

            deltas_sigmas = deltas * sigmas  # [N, T]

            # opaque background
            if not baking and self.opt.background == 'last_sample':
                deltas_sigmas = torch.cat(
                    [deltas_sigmas[..., :-1], torch.full_like(deltas_sigmas[..., -1:], torch.inf)], dim=-1)

            alphas = 1 - torch.exp(-deltas_sigmas)  # [N, T]

            transmittance = torch.cumsum(deltas_sigmas[..., :-1], dim=-1)  # [N, T-1]
            transmittance = torch.cat([torch.zeros_like(transmittance[..., :1]), transmittance], dim=-1)  # [N, T]
            transmittance = torch.exp(-transmittance)  # [N, T]

            weights = alphas * transmittance  # [N, T]
            weights.nan_to_num_(0)

            if self.training:
                all_bins.append(bins)
                all_weights.append(weights)

        results['xyzs'] = xyzs  # [N, T, 3] in [-2, 2]
        results['weights'] = weights  # [N, T]
        results['alphas'] = alphas  # [N, T]

        if baking:
            # results['rgbs'] = rgbs
            return results

        # composite
        weights_sum = torch.sum(weights, dim=-1).clamp(0, 1)  # [N]
        bg_weights = (1 - weights_sum).clamp(0, 1)  # [N]
        depth = torch.sum(weights * rays_t, dim=-1)  # [N]

        features_blended = torch.sum(weights.unsqueeze(-1) * features, dim=-2)  # [N, 7]
        features_blended[:, :3] = features_blended[:, :3] + bg_weights[::, None] * bg_color  # [N, 3]
        # view_dirs = torch.sum(weights.unsqueeze(-1) * outputs['dirs'], dim=-2)  # [N, 27]
        # view_dirs = outputs['dirs']
        view_dirs = coord.pos_enc(dirs, min_deg=0, max_deg=4, append_identity=True)
        # view_dirs = self.view_encoder(dirs)

        diffuse = features_blended[:, :3]
        results['diffuse'] = diffuse

        if shading == 'diffuse':
            image = diffuse
        else:
            specular = self.DeferredMLP(torch.cat([features_blended, view_dirs], dim=-1))
            results['specular'] = specular

            # print(f"\nresults['diffuse']: {results['diffuse'].min().item(), results['diffuse'].max().item()}")
            # print(f"results['specular']: {results['specular'].min().item(), results['specular'].max().item()}")

            image = (diffuse + specular).clamp(0, 1)

        # extra results
        if self.training:
            results['num_points'] = xyzs.shape[0] * xyzs.shape[1]

            if self.opt.lambda_proposal > 0 and update_proposal:
                results['proposal_loss'] = proposal_loss(all_bins, all_weights)

            if self.opt.lambda_distort > 0:
                results['distort_loss'] = distort_loss(bins, weights)

            if self.opt.lambda_sparsity > 0:
                # Sample a fixed number of points within [-2,2]^3.
                num_random_samples = 2 ** 17 // 8

                # random_positions = torch.tensor(np.random.uniform(
                #     low=grid_utils.WORLD_MIN,
                #     high=grid_utils.WORLD_MAX,
                #     size=(num_random_samples, 3),
                # ), device=image.device, dtype=torch.float32, requires_grad=True)
                random_positions = torch.tensor(np.random.uniform(
                    low=-self.bound,
                    high=self.bound,
                    size=(num_random_samples, 3),
                ), device=image.device, dtype=torch.float32, requires_grad=True)
                random_viewdirs = torch.tensor(np.random.normal(size=(num_random_samples, 3)),
                                               device=image.device, dtype=torch.float32)
                random_viewdirs /= torch.linalg.norm(random_viewdirs, dim=-1, keepdim=True)
                _, density = self.forward(random_positions, bound=self.bound)
                results['sparsity_loss'] = yu_sparsity_loss(random_positions, random_viewdirs,
                                                            density, self.grid_config['voxel_size_to_use'],
                                                            self.opt.contract)

        results['weights_sum'] = weights_sum
        results['depth'] = depth
        results['image'] = image

        return results

    def run(self, rays_o, rays_d, cam_near_far=None, shading='full', update_proposal=False, **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]
        if self.opt.render == 'grid':
            return self.run_merf(rays_o, rays_d, cam_near_far=cam_near_far, shading=shading,
                                 update_proposal=update_proposal, **kwargs)
        elif self.opt.render == 'mesh':
            return self.run_surface_render(rays_o, rays_d, shading=shading,
                                           update_proposal=update_proposal, **kwargs)
        elif self.opt.render == 'mixed':
            assert 'rays_d_all' in kwargs.keys()
            if self.vertices is not None:
                return self.run_mixed_render(rays_o, rays_d, cam_near_far=cam_near_far, shading=shading,
                                             update_proposal=update_proposal, **kwargs)
            else:
                return self.run_merf(rays_o, rays_d, cam_near_far=cam_near_far, shading=shading,
                                     update_proposal=update_proposal, **kwargs)

    # def voxel_alive_check(self, xyzs, use_occ_grid=True, use_mesh_occ_grid=True):
    #     # voxel_alive_mask: alive = 1 and dead = 0
    #     # xyzs: [N, T, 3], xyzs must be contracted (in [-2, 2])
    #     # put xyzs into [mcubes_reso^3] cubes
    #
    #     assert xyzs.max() <= self.bound and xyzs.min() >= -self.bound
    #
    #     xyzs_coords = (torch.floor((xyzs + self.bound) / (2 * self.bound) * self.opt.mcubes_reso).long()
    #                    .clamp(0, self.opt.mcubes_reso - 1))
    #     # [N, T]
    #     # xyzs_mask = self.voxel_alive_mask[xyzs_coords[..., 0], xyzs_coords[..., 1], xyzs_coords[..., 2]]
    #
    #     # if use_occ_grid and self.training:
    #     #     if use_mesh_occ_grid:
    #     #         xyzs_mask = (self.occ_grid * (~self.mesh_occ_mask))[
    #     #             xyzs_coords[..., 0], xyzs_coords[..., 1], xyzs_coords[..., 2]]
    #     #     else:
    #     #         xyzs_mask = self.occ_grid[xyzs_coords[..., 0], xyzs_coords[..., 1], xyzs_coords[..., 2]]
    #     # elif use_mesh_occ_grid:
    #     #     xyzs_mask = (~self.mesh_occ_mask)[xyzs_coords[..., 0], xyzs_coords[..., 1], xyzs_coords[..., 2]]
    #
    #     xyzs_mask = torch.ones_like(xyzs[..., 0], dtype=torch.bool)
    #
    #     if use_occ_grid and self.training:
    #         xyzs_mask *= self.occ_grid[xyzs_coords[..., 0], xyzs_coords[..., 1], xyzs_coords[..., 2]]
    #     if use_mesh_occ_grid:
    #         xyzs_mask[:, -1] *= (~self.mesh_occ_mask)[
    #             xyzs_coords[:, -1, 0], xyzs_coords[:, -1, 1], xyzs_coords[:, -1, 2]]
    #
    #     return xyzs_mask

    def run_mixed_render(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, shading='full',
                         update_proposal=True, baking=False, **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device
        results = {}
        eps = 1e-15

        # mix background color
        if bg_color is None:
            bg_color = 1

        # pre-calculate near far
        nears, fars = near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer,
                                         self.min_near)

        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, [0]])
            fars = torch.minimum(fars, cam_near_far[:, [1]])

        # sample xyzs using a mixed linear + lindisp function
        if self.opt.data_format == 'nerf':
            spacing_fn = lambda x: x
            spacing_fn_inv = lambda x: x
        else:
            spacing_fn = lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
            spacing_fn_inv = lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x))

        results_mesh = self.render_mesh(rays_o=rays_o, rays_d=kwargs['rays_d_all'],
                                        h0=kwargs['H'], w0=kwargs['W'], bg_color=bg_color,
                                        shading=shading, **kwargs)

        xyzs_mesh = results_mesh['xyzs']
        # mask_mesh = results_mesh['mask']
        mask_mesh = results_mesh['mask'] * (
                results_mesh['alphas'] == 1.0)  # results_mesh['mask']或者results_mesh['alphas']不对

        results['mask_mesh'] = mask_mesh  # [N]
        # erode to avoid black alias in mixed rendering
        # results['mask_mesh'] = erode(mask_mesh.squeeze(-1)).unsqueeze(-1)  # [N]
        # mask_mesh[results_mesh['alphas'] < 0.6] = 0
        # results['mask_mesh'][results['alphas_mesh'] == 1] = 1
        results['real_fars'] = fars.clone().detach()  # [N, 1]

        if mask_mesh.any():
            # rays_d_tmp = torch.abs(rays_d[mask_mesh] / torch.norm(rays_d[mask_mesh], dim=-1, keepdim=True))
            rays_d_tmp = torch.abs(rays_d[mask_mesh])
            avoid_zero_idx = rays_d_tmp.max(dim=-1, keepdim=True).indices
            fars_mesh = (torch.abs((xyzs_mesh[mask_mesh] - rays_o[mask_mesh]) / (rays_d_tmp + eps))
                         .take_along_dim(avoid_zero_idx, -1).clamp(max=results['real_fars'][mask_mesh]))
            fars[mask_mesh] = fars_mesh

        results['fars_depth'] = fars.squeeze()

        # hierarchical sampling
        if self.training:
            all_bins = []
            all_weights = []

        s_nears = spacing_fn(nears)  # [N, 1]
        s_fars = spacing_fn(fars)  # [N, 1]

        bins = None
        weights = None

        # step_size = (self.bound - (-self.bound)) / self.opt.triplane_resolution
        # num_steps = int(math.sqrt(3) * (self.bound - (-self.bound)) / step_size)
        # assert len(self.opt.num_steps) == 1
        for prop_iter in range(len(self.opt.num_steps)):

            if prop_iter == 0:
                # uniform sampling
                # [1, T+1]
                bins = torch.linspace(0, 1, self.opt.num_steps[prop_iter] + 1, device=device).unsqueeze(0)
                bins = bins.expand(N, -1)  # [N, T+1]
                if perturb:
                    bins = bins + (torch.rand_like(bins) - 0.5) / (self.opt.num_steps[prop_iter])
                    bins = bins.clamp(0, 1)
            else:
                # pdf sampling
                bins = sample_pdf(bins, weights, self.opt.num_steps[prop_iter] + 1, perturb).detach()  # [N, T+1]

            real_bins = spacing_fn_inv(s_nears * (1 - bins) + s_fars * bins)  # [N, T+1] in [near, far]

            rays_t = (real_bins[..., 1:] + real_bins[..., :-1]) / 2  # [N, T]
            xyzs = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * rays_t.unsqueeze(2)  # [N, T, 3]

            # EDITED
            # xyzs_offset = torch.zeros(xyzs.shape[0]).to(xyzs)
            # if mask_mesh.any():
            #     xyzs_mesh_ori = xyzs_mesh[mask_mesh]
            #     xyzs_recal = xyzs[mask_mesh, -1]
            #     xyzs_offset[mask_mesh] = torch.abs((xyzs_recal - xyzs_mesh_ori).mean(dim=-1))
            # results['xyzs_offset_depth'] = xyzs_offset

            if self.opt.contract:
                results['real_xyzs'] = xyzs  # in [-real_bound, real_bound]
                xyzs = contract(xyzs)

            if prop_iter != len(self.opt.num_steps) - 1:
                # query proposal density
                with torch.set_grad_enabled(update_proposal):
                    # sigmas = self.density(xyzs, proposal=prop_iter)['sigma']  # [N, T]
                    sigmas = self.prop_mlp[0](self.prop_encoders[0]((xyzs + self.bound) / (2 * self.bound)))
                    sigmas = math.density_activation(sigmas).squeeze(-1)

                    # if self.opt.density_scale > 0:
                    #     pos_coords = (
                    #         torch.floor((xyzs + self.bound) / (2 * self.bound) * self.density_scale.shape[0]).long()
                    #         .clamp(0, self.density_scale.shape[0] - 1))
                    #     scale = self.density_scale[pos_coords[..., 0], pos_coords[..., 1], pos_coords[..., 2]]
                    #     sigmas = sigmas * scale
            else:
                # last iter: query nerf
                dirs = rays_d.view(-1, 3)  # [N, 3]
                dirs /= torch.linalg.norm(dirs, dim=-1, keepdim=True)

                # outputs = self(xyzs, dirs, shading=shading)
                # sigmas = outputs['sigma']
                # features = outputs['features']

                features, sigmas = self.forward(xyzs, bound=self.bound)
                sigmas = sigmas.squeeze(-1)

                # print('\nalpha_threshold: ', kwargs['alpha_threshold'])
                # alpha_threshold = 0.005
                with torch.enable_grad():
                    sigmas = simulate_alpha_culling(sigmas, xyzs, dirs, kwargs['alpha_threshold'],
                                                    self.grid_config['voxel_size_to_use']).squeeze(-1)
                # print('\nalive voxels: ', (sigmas > 0.0).float().sum(-1).min().item(),
                #       (sigmas > 0.0).float().sum(-1).max().item(),
                #       (sigmas > 0.0).float().sum(-1).mean().item())

            # sigmas to weights
            deltas = (real_bins[..., 1:] - real_bins[..., :-1])  # [N, T]
            # if self.opt.no_check:
            #     deltas_sigmas = deltas * sigmas  # [N, T]
            # else:
            #     deltas_sigmas = deltas * sigmas * self.voxel_alive_check(xyzs, self.opt.use_occ_grid,
            #                                                              self.opt.use_mesh_occ_grid)  # [N, T]
            deltas_sigmas = deltas * sigmas  # [N, T]

            # opaque background
            # if not baking and self.opt.background == 'last_sample':
            #     deltas_sigmas = torch.cat(
            #         [deltas_sigmas[..., :-1], torch.full_like(deltas_sigmas[..., -1:], torch.inf)], dim=-1)
            assert self.opt.background != 'last_sample'
            # deltas_sigmas = torch.cat([deltas_sigmas, torch.full_like(deltas_sigmas[..., -1:], torch.inf)], dim=-1)

            alphas = 1 - torch.exp(-deltas_sigmas)  # [N, T]

            transmittance = torch.cumsum(deltas_sigmas[..., :-1], dim=-1)  # [N, T-1]
            transmittance = torch.cat([torch.zeros_like(transmittance[..., :1]), transmittance], dim=-1)  # [N, T]
            transmittance = torch.exp(-transmittance)  # [N, T]

            weights = alphas * transmittance  # [N, T+1]
            weights.nan_to_num_(0)

            if self.training:
                all_bins.append(bins)
                all_weights.append(weights)

        results['xyzs'] = xyzs  # [N, T, 3] in [-2, 2]
        results['weights'] = weights  # [N, T]
        results['alphas'] = alphas  # [N, T]

        if baking:
            return results

        # composite
        # if self.opt.ras_mask:
        #     weights[~results['mask_mesh']] = 0
        weights_sum = torch.sum(weights, dim=-1).clamp(0, 1)  # [N]
        depth = torch.sum(weights * rays_t, dim=-1)  # [N]

        mesh_weights = (1 - weights_sum).clamp(0, 1) * results_mesh['alphas']
        bg_weights = (1 - weights_sum).clamp(0, 1) * (1.0 - results_mesh['alphas'])
        # mesh_weights = (1 - weights_sum).clamp(0, 1) * results_mesh['alphas'] # [N]
        # mesh_weights[~results['mask_mesh']] = 0

        # bg_weights = (1 - weights_sum - mesh_weights).clamp(0, 1)  # [N]
        # bg_weights[results['mask_mesh']] = 0

        # mesh_weights = (1 - weights_sum).clamp(0, 1)  # [N]
        # mesh_weights[~results['mask_mesh']] = results_mesh['alphas'][~results['mask_mesh']]

        # bg_weights = (1 - weights_sum - mesh_weights).clamp(0, 1)  # [N]
        # bg_weights[results['mask_mesh']] = 1 - results_mesh['alphas'][results['mask_mesh']]

        results['weights_mesh'] = mesh_weights
        results['weights_bg'] = bg_weights

        # features_mesh = results_mesh['features'].masked_fill(~results['mask_mesh'][:, None], bg_color)
        # features_blended = torch.sum(torch.cat([weights, bg_weights], dim=1).unsqueeze(-1) *
        #                              torch.cat([features, features_mesh.unsqueeze(1)], dim=1), dim=-2)  # [N, 7]
        # diffuse = features_blended[:, :3] + (~results['mask_mesh'][:, None]) * bg_weights * bg_color
        # features_blended += bg_weights[::, None] * features_blended_mesh
        # results['diffuse_mesh_bgWeight'] = (bg_weights[::, None] * features_blended_mesh)[:, :3]

        features_blended = torch.sum(weights.unsqueeze(-1) * features, dim=-2)  # [N, 7]
        diffuse_voxel = features_blended[:, :3]  # [N, 3]
        features_mesh = results_mesh['features'] * mesh_weights[::, None]

        # results['diffuse_mesh_raw'] = results_mesh['features'][:, :3]
        # results['diffuse_mesh'] = features_mesh[:, :3]
        # results['diffuse_voxel'] = diffuse_voxel
        # results['diffuse'] = (
        #         results['diffuse_mesh'] + results['diffuse_voxel'] + bg_weights[::, None] * bg_color).clamp(0, 1)

        results['diffuse_mesh_raw'] = (results_mesh['features'][:, :3] + bg_weights[::, None] * bg_color).clamp(0, 1)
        results['diffuse_mesh'] = features_mesh[:, :3]
        results['diffuse_voxel'] = diffuse_voxel
        results['diffuse'] = (
                results['diffuse_mesh'] + results['diffuse_voxel'] + bg_weights[::, None] * bg_color).clamp(0, 1)

        results['diffuse_voxel'] = (
                diffuse_voxel + bg_weights[::, None] * bg_color + mesh_weights[::, None] * bg_color).clamp(0, 1)
        results['diffuse_mesh'] = (features_mesh[:, :3] + bg_weights[::, None] * bg_color).clamp(0, 1)

        if shading == 'diffuse':
            image = results['diffuse']
        else:
            view_dirs = coord.pos_enc(dirs, min_deg=0, max_deg=4, append_identity=True)
            rgb_specular = self.DeferredMLP((torch.cat([results['diffuse'],
                                                        features_blended[:, 3:] + features_mesh[:, 3:],
                                                        view_dirs, ], dim=1)))
            results['specular'] = rgb_specular
            image = (results['diffuse'] + results['specular']).clamp(0, 1)

        # extra results
        if self.training:
            results['num_points'] = xyzs.shape[0] * xyzs.shape[1]
            results['weights'] = weights
            results['alphas'] = alphas

            if self.opt.lambda_proposal > 0 and update_proposal:
                results['proposal_loss'] = proposal_loss(all_bins, all_weights)

            if self.opt.lambda_distort > 0:
                results['distort_loss'] = distort_loss(bins, weights)

            if self.opt.lambda_sparsity > 0:
                # Sample a fixed number of points within [-2,2]^3.
                num_random_samples = 2 ** 17 // 8

                random_positions = torch.tensor(np.random.uniform(
                    low=grid_utils.WORLD_MIN,
                    high=grid_utils.WORLD_MAX,
                    size=(num_random_samples, 3),
                ), device=image.device, dtype=torch.float32, requires_grad=True)
                random_viewdirs = torch.tensor(np.random.normal(size=(num_random_samples, 3)),
                                               device=image.device, dtype=torch.float32)
                random_viewdirs /= torch.linalg.norm(random_viewdirs, dim=-1, keepdim=True)
                _, density = self.forward(random_positions, bound=self.bound)
                results['sparsity_loss'] = yu_sparsity_loss(random_positions, random_viewdirs,
                                                            density, self.grid_config['voxel_size_to_use'],
                                                            self.opt.contract)

            if self.opt.render == 'mixed':
                if self.opt.lambda_mesh_weight > 0:
                    # mesh weight loss
                    if self.opt.lambda_ec_weight > 0:
                        error_convert = self.find_error_in_grid(results['xyzs'][results['mask_mesh']])
                        print(f'[INFO] error_convert: {error_convert.mean().item()}, {error_convert.min().item()}, '
                              f'{error_convert.max().item()}')
                        results['mesh_weight_loss'] = 1.0 - (
                                torch.exp(mesh_weights[results['mask_mesh']] - 1.0) * error_convert).mean()
                    else:
                        results['mesh_weight_loss'] = 1.0 - torch.exp(mesh_weights[results['mask_mesh']] - 1.0).mean()
                if self.opt.lambda_bg_weight > 0:
                    results['bg_weight_loss'] = (bg_weights ** 2).mean()
                    # results['bg_weight_loss'] = 1.0 - torch.exp(-1.0 * bg_weights).mean()
        else:
            results['full_mesh'] = (results['diffuse_mesh'] +
                                    self.DeferredMLP((torch.cat([results['diffuse_mesh'],
                                                                 features_mesh[:, 3:],
                                                                 view_dirs, ],
                                                                dim=1)))).clamp(0, 1)

            results['full_mesh_raw'] = (results['diffuse_mesh_raw'] +
                                        self.DeferredMLP((torch.cat([results['diffuse_mesh_raw'],
                                                                     results_mesh['features'][:, 3:],
                                                                     view_dirs, ], dim=1)))).clamp(0, 1)
            if results['full_mesh_raw'][~results['mask_mesh']].shape[0] > 0:
                results['full_mesh_raw'][~results['mask_mesh']] = bg_color

            results['full_voxel'] = (results['diffuse_voxel'] +
                                     self.DeferredMLP((torch.cat([results['diffuse_voxel'],
                                                                  features_blended[:, 3:],
                                                                  view_dirs, ],
                                                                 dim=1)))).clamp(0, 1)

        results['weights_voxels'] = weights_sum
        results['weights_vosh'] = weights_sum + mesh_weights
        results['depth'] = depth
        results['image'] = image
        results['fars'] = fars  # [N, 3]

        return results

    def find_error_in_grid(self, xyzs):
        coords = (torch.floor((xyzs + self.bound) / (2 * self.bound) * self.error_grid.shape[0])
                  .long().clamp(0, self.error_grid.shape[0] - 1))  # [N*T, 3]
        return self.error_grid[tuple(coords.T)].float()

    def run_surface_render(self, rays_o, rays_d, bg_color=None, shading='full', **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        results = {}

        # mix background color
        if bg_color is None:
            bg_color = 1

        results_mesh = self.render_mesh(rays_o=rays_o, rays_d=kwargs['rays_d_all'],
                                        h0=kwargs['H'], w0=kwargs['W'], bg_color=bg_color,
                                        shading=shading, **kwargs)

        results['mask_mesh'] = results_mesh['mask']
        # results['mask_mesh'] = results_mesh['mask'] * (results_mesh['alphas'] == 1.0)  # [N]

        # print("features shape: {}".format(results_mesh['features'].shape))
        # print("alphas shape: {}".format(results_mesh['alphas'].shape))
        results_mesh['features'] = results_mesh['features'] * results_mesh['alphas'].view(-1, 1)

        depth = results_mesh['depth']  # [N]

        results['diffuse'] = results_mesh['features'][:, :3] + (1.0 - results_mesh['alphas'][:, None]) * bg_color

        if shading == 'diffuse':
            image = results['diffuse']
        else:
            dirs = rays_d.view(-1, 3)  # [N, 3]
            dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
            view_dirs = coord.pos_enc(dirs, min_deg=0, max_deg=4, append_identity=True)
            results['specular'] = self.DeferredMLP((torch.cat([results_mesh['features'], view_dirs], dim=1)))
            image = (results['diffuse'] + results['specular']).clamp(0, 1)

        results['depth'] = depth
        results['image'] = image
        results['weights_sum'] = results_mesh['weights_sum']
        results['xyzs'] = results_mesh['xyzs']

        if 'vis_error' in self.opt and self.opt.vis_error:
            mask = results['mask_mesh']
            error_val = torch.zeros_like(depth)

            if self.opt.contract:
                xyzs = contract(results_mesh['xyzs'][mask])  # [the number of 1 in mask, 3] in (-2, 2)
            else:
                xyzs = results_mesh['xyzs'][mask]  # [the number of 1 in mask, 3]

            target_resolution = [32, 64, 128, 256, 512]
            for resolution in target_resolution:
                # error_grid = F.interpolate(self.error_grid.unsqueeze(0).unsqueeze(0).float(),
                #                            size=[resolution] * 3, mode='trilinear').squeeze(0).squeeze(0)
                error_grid = F.max_pool3d(self.error_grid.unsqueeze(0).unsqueeze(0).float(),
                                          self.error_grid.shape[0] // resolution,
                                          stride=self.error_grid.shape[0] // resolution).squeeze(0).squeeze(0)

                coords = (torch.floor((xyzs + self.bound) / (2 * self.bound) * resolution)
                          .long().clamp(0, resolution - 1))  # [N*T, 3]
                error_val_mask = error_grid[tuple(coords.T)]
                error_val[mask] = error_val_mask.float()
                # print(f'error_eval: {error_val.mean().item()}, {error_val.min().item()}, {error_val.max().item()}')
                # for i in range(coords.shape[0]):
                #     if error_val[i] > 0.5:
                #         print(f'[{i}], {coords[i]}: {error_val[i]}')

                # visualize error in red and blue
                error_rgb = torch.zeros_like(image)

                mesh_error_bigger = torch.zeros_like(error_val)
                mesh_error_bigger[error_val > 0] = error_val[error_val > 0]
                mesh_error_bigger = (mesh_error_bigger - mesh_error_bigger.min()) / (
                        mesh_error_bigger.max() - mesh_error_bigger.min())
                # print(f'\n[INFO] mesh_error_bigger: {mesh_error_bigger.mean().item()}, '
                #       f'{mesh_error_bigger.min().item()}, '
                #       f'{mesh_error_bigger.max().item()}')
                threshold = torch.quantile(mesh_error_bigger[mesh_error_bigger > 0].flatten(), 0.9)
                mesh_error_bigger[mesh_error_bigger < threshold] = 0.0
                error_rgb = error_rgb + mesh_error_bigger.view(-1, 1) * torch.tensor([1.0, 0.0, 0.0],
                                                                                     device=image.device)

                if (error_val < 0).any():
                    voxel_error_bigger = torch.zeros_like(error_val)
                    voxel_error_bigger[error_val < 0] = -error_val[error_val < 0]
                    voxel_error_bigger = (voxel_error_bigger - voxel_error_bigger.min()) / (
                            voxel_error_bigger.max() - voxel_error_bigger.min())
                    # print(f'[INFO] voxel_error_bigger: {voxel_error_bigger.mean().item()}, '
                    #       f'{voxel_error_bigger.min().item()}, '
                    #       f'{voxel_error_bigger.max().item()}')
                    error_rgb = error_rgb + voxel_error_bigger.view(-1, 1) * torch.tensor([0.0, 0.0, 1.0],
                                                                                          device=image.device)
                results[f'error_rgb_{resolution}'] = error_rgb

                # visualize error in greyscale
                error_grey = torch.zeros_like(depth)
                error_grey[error_val > 0] = error_val[error_val > 0]
                error_grey = (error_grey - error_grey.min()) / (error_grey.max() - error_grey.min())
                threshold = torch.quantile(error_grey[error_grey > 0].flatten(), 0.9)
                error_grey[error_grey < threshold] = 0.0

                results[f'error_grey_{resolution}'] = error_grey

        return results

    @torch.no_grad()
    def export_mesh_assert_cas(self, path, h0=2048, w0=2048, png_compression_level=3):
        # png_compression_level: 0 is no compression, 9 is max (default will be 3)
        self.mesh_occ_mask = None
        device = self.vertices.device
        os.makedirs(path, exist_ok=True)

        if self.vertices_offsets is not None:
            v = (self.vertices + self.vertices_offsets).detach()
        else:
            v = self.vertices.detach()
        f = self.triangles.detach()

        m = pml.Mesh(v.cpu().numpy(), f.cpu().numpy())
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh')

        def _export_obj(v, f, h0, w0, ssaa=1, cas=0):
            # v, f: torch Tensor

            v_np = v.cpu().numpy()  # [N, 3]
            f_np = f.cpu().numpy()  # [M, 3]

            # print(f'[INFO] exporting cas {cas}')
            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uv in contracted space
            atlas = xatlas.Atlas()
            atlas.add_mesh(contract(v).cpu().numpy() if self.opt.contract else v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
            pack_options = xatlas.PackOptions()
            # pack_options.blockAlign = True
            # pack_options.bruteForce = False
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0

            print('vt.shape: ', vt.shape, ' ft.shape: ', ft.shape, ' uv.shape: ', uv.shape)

            # tmp_glctx = dr.RasterizeGLContext()
            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
            # rast, _ = dr.rasterize(tmp_glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

            # masked query
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)

            if self.opt.contract:
                xyzs = contract(xyzs)

            # feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)
            feats = torch.zeros(h * w, 7, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask]  # [M, 3]

                all_feats = []
                # batched inference to avoid OOM
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        # features, _ = self(xyzs[head:tail], only_feat=True)

                        # features, _ = self.DensityAndFeaturesMLP(xyzs[head:tail], bound=self.bound)
                        if 'mesh_encoder' in self.opt and self.opt.mesh_encoder:
                            features, _ = self.DensityAndFeaturesMLP_mesh(xyzs[head:tail], bound=self.bound)
                        else:
                            features, _ = self.DensityAndFeaturesMLP(xyzs[head:tail], bound=self.bound)
                        features = quantize.simulate_quantization(features, -7.0, 7.0)
                        features = torch.sigmoid(features)

                        all_feats.append(features)

                        # mask_features, _ = self.DensityAndFeaturesMLP(contract(xyzs[mask_flatten].detach()))
                        # mask_features = quantize.simulate_quantization(
                        #     mask_features, self.range_features[0], self.range_features[1]
                        # )
                        # mask_features = torch.sigmoid(mask_features)

                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0).float()

            # feats = torch.sigmoid(feats)  # sigmoid for vosh
            feats = feats.view(h, w, -1)  # 7 channels in nerf2mesh

            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)
            feats = feats.astype(np.uint8)
            # print('feats: ', feats[..., :3].min(), feats[..., :3].max())

            ### NN search as a queer antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=32)  # pad width
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            # do ssaa after the NN search, in numpy
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)  # albedo
            feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGBA2BGRA)  # visibility features
            # feats0, feats1 = feats[..., :3], feats[..., 3:]

            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(path, f'feat0_{cas}.jpg'), feats0)
            cv2.imwrite(os.path.join(path, f'feat1_{cas}.png'), feats1)

            # save obj (v, vt, f /)
            backup_path = os.path.join(path, '../backup')
            os.makedirs(backup_path, exist_ok=True)
            obj_file = os.path.join(backup_path, f'mesh_{cas}.obj')
            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:

                fp.write(f'mtllib mesh_{cas}.mtl \n')

                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n')

                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl defaultMat \n')
                for i in range(len(f_np)):
                    fp.write(
                        f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            scene = a3d.Scene.from_file(obj_file)
            scene.save(obj_file.replace('.obj', '.drc'))
            scene.save(obj_file.replace(backup_path, path).replace('.obj', '.glb'))

            # shutil.rmtree(obj_file)

            mtl_file = os.path.join(backup_path, f'mesh_{cas}.mtl')
            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl defaultMat \n')
                fp.write(f'Ka 1 1 1 \n')
                fp.write(f'Kd 1 1 1 \n')
                fp.write(f'Ks 0 0 0 \n')
                fp.write(f'Tr 1 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0 \n')
                fp.write(f'map_Kd feat0_{cas}.jpg \n')

        idx = 0
        for cas in self.cascade_list:
            ms.compute_selection_by_condition_per_vertex(
                condselect=f'(x < {cas}) && (x > -{cas}) && (y < {cas}) && (y > -{cas}) '
                           f'&& (z < {cas}) && (z > -{cas})')
            ms.compute_selection_transfer_vertex_to_face()
            ms.generate_from_selected_faces()

            ms.set_current_mesh(len(ms) - 1)
            m = ms.current_mesh()
            if m.vertex_matrix().shape[0] > 0:
                _export_obj(torch.tensor(m.vertex_matrix(), device=v.device).float().contiguous(),
                            torch.tensor(m.face_matrix(), device=v.device).contiguous(),
                            h0, w0, self.opt.ssaa, idx)

            ms.set_current_mesh(0)
            m = ms.current_mesh()
            idx += 1
            # print('mesh excluded : ', m.vertex_matrix().shape, m.face_matrix().shape)

        # assert m.vertex_matrix().shape[0] == 0

        if m.vertex_matrix().shape[0] > 0:
            _export_obj(torch.tensor(m.vertex_matrix(), device=v.device).float().contiguous(),
                        torch.tensor(m.face_matrix(), device=v.device).contiguous(),
                        h0, w0, self.opt.ssaa, idx)

    @torch.no_grad()
    def export_mesh_assert_by_number(self, path, h0=2048, w0=2048, png_compression_level=3):
        # png_compression_level: 0 is no compression, 9 is max (default will be 3)
        self.mesh_occ_mask = None
        device = self.vertices.device
        os.makedirs(path, exist_ok=True)

        if self.vertices_offsets is not None:
            v = (self.vertices + self.vertices_offsets).detach()
        else:
            v = self.vertices.detach()
        f = self.triangles.detach()

        def _export_obj(v, f, h0, w0, ssaa=1, cas=0):
            # v, f: torch Tensor

            v_np = v.cpu().numpy()  # [N, 3]
            f_np = f.cpu().numpy()  # [M, 3]

            # print(f'[INFO] exporting cas {cas}')
            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uv in contracted space
            atlas = xatlas.Atlas()
            atlas.add_mesh(contract(v).cpu().numpy() if self.opt.contract else v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
            pack_options = xatlas.PackOptions()
            # pack_options.blockAlign = True
            # pack_options.bruteForce = False
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0

            print('vt.shape: ', vt.shape, ' ft.shape: ', ft.shape, ' uv.shape: ', uv.shape)

            # tmp_glctx = dr.RasterizeGLContext()
            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
            # rast, _ = dr.rasterize(tmp_glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

            # masked query
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)

            if self.opt.contract:
                xyzs = contract(xyzs)

            # feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)
            feats = torch.zeros(h * w, 7, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask]  # [M, 3]

                all_feats = []
                # batched inference to avoid OOM
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        # features, _ = self(xyzs[head:tail], only_feat=True)

                        # features, _ = self.DensityAndFeaturesMLP(xyzs[head:tail], bound=self.bound)
                        if 'mesh_encoder' in self.opt and self.opt.mesh_encoder:
                            features, _ = self.DensityAndFeaturesMLP_mesh(xyzs[head:tail], bound=self.bound)
                        else:
                            features, _ = self.DensityAndFeaturesMLP(xyzs[head:tail], bound=self.bound)
                        features = quantize.simulate_quantization(features, -7.0, 7.0)
                        features = torch.sigmoid(features)

                        all_feats.append(features)

                        # mask_features, _ = self.DensityAndFeaturesMLP(contract(xyzs[mask_flatten].detach()))
                        # mask_features = quantize.simulate_quantization(
                        #     mask_features, self.range_features[0], self.range_features[1]
                        # )
                        # mask_features = torch.sigmoid(mask_features)

                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0).float()

            # feats = torch.sigmoid(feats)  # sigmoid for vosh
            feats = feats.view(h, w, -1)  # 7 channels in nerf2mesh

            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)
            feats = feats.astype(np.uint8)
            # print('feats: ', feats[..., :3].min(), feats[..., :3].max())

            ### NN search as a queer antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=32)  # pad width
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            # do ssaa after the NN search, in numpy
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)  # albedo
            feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGBA2BGRA)  # visibility features
            # feats0, feats1 = feats[..., :3], feats[..., 3:]

            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(path, f'feat0_{cas}.jpg'), feats0)
            cv2.imwrite(os.path.join(path, f'feat1_{cas}.png'), feats1)

            # save obj (v, vt, f /)
            backup_path = os.path.join(path, '../backup')
            os.makedirs(backup_path, exist_ok=True)
            obj_file = os.path.join(backup_path, f'mesh_{cas}.obj')
            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:

                fp.write(f'mtllib mesh_{cas}.mtl \n')

                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n')

                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl defaultMat \n')
                for i in range(len(f_np)):
                    fp.write(
                        f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            scene = a3d.Scene.from_file(obj_file)
            scene.save(obj_file.replace('.obj', '.drc'))
            scene.save(obj_file.replace(backup_path, path).replace('.obj', '.glb'))

            # shutil.rmtree(obj_file)

            mtl_file = os.path.join(backup_path, f'mesh_{cas}.mtl')
            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl defaultMat \n')
                fp.write(f'Ka 1 1 1 \n')
                fp.write(f'Kd 1 1 1 \n')
                fp.write(f'Ks 0 0 0 \n')
                fp.write(f'Tr 1 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0 \n')
                fp.write(f'map_Kd feat0_{cas}.jpg \n')

        idx, head = 0, 0
        local_face_num = self.opt.local_face_num

        while head < f.shape[0]:
            tail = min(head + local_face_num, f.shape[0])
            _export_obj(v, f[head:tail], h0, w0, self.opt.ssaa, idx)
            idx += 1
            head = tail

        scene_params = {
            'cas_num': idx,
        }
        with open(os.path.join(path, 'mesh_params.json'), 'w') as f:
            json.dump(scene_params, f, indent=2)
