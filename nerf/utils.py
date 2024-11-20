import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX
import mcubes
import shutil

import numpy as np
import json

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchmetrics.functional import structural_similarity_index_measure

import trimesh
import nvdiffrast.torch as dr
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
from meshutils import *
from .renderer import contract, uncontract
from nerf import quantize

from skimage import measure


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2 = epsilon * epsilon

    def forward(self, pred_rgb, gt_rgb):
        value = torch.sqrt(torch.pow(pred_rgb - gt_rgb, 2) + self.epsilon2)
        return value


def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian
    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L


def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()


def laplacian_smooth_loss(verts, faces, cotan=False):
    with torch.no_grad():
        if cotan:
            L = laplacian_cot(verts, faces.long())
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            mask = norm_w > 0
            norm_w[mask] = 1.0 / norm_w[mask]
        else:
            L = laplacian_uniform(verts, faces.long())
    if cotan:
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]

    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # if color is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd])

    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))

    trimesh.Scene([pc, axes, box]).show()


def create_dodecahedron_cameras(radius=1, center=np.array([0, 0, 0])):
    vertices = np.array([
        -0.57735, -0.57735, 0.57735,
        0.934172, 0.356822, 0,
        0.934172, -0.356822, 0,
        -0.934172, 0.356822, 0,
        -0.934172, -0.356822, 0,
        0, 0.934172, 0.356822,
        0, 0.934172, -0.356822,
        0.356822, 0, -0.934172,
        -0.356822, 0, -0.934172,
        0, -0.934172, -0.356822,
        0, -0.934172, 0.356822,
        0.356822, 0, 0.934172,
        -0.356822, 0, 0.934172,
        0.57735, 0.57735, -0.57735,
        0.57735, 0.57735, 0.57735,
        -0.57735, 0.57735, -0.57735,
        -0.57735, 0.57735, 0.57735,
        0.57735, -0.57735, -0.57735,
        0.57735, -0.57735, 0.57735,
        -0.57735, -0.57735, -0.57735,
    ]).reshape((-1, 3), order="C")

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    # construct camera poses by lookat
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # forward is simple, notice that it is in fact the inversion of camera direction!
    forward_vector = normalize(vertices - center)
    # pick a temp up_vector, usually [0, 1, 0]
    up_vector = np.array([0, 1, 0], dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    # cross(up, forward) --> right
    right_vector = normalize(np.cross(up_vector, forward_vector, axis=-1))
    # rectify up_vector, by cross(forward, right) --> up
    up_vector = normalize(np.cross(forward_vector, right_vector, axis=-1))

    ### construct c2w
    poses = np.eye(4, dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)
    poses[:, :3, 3] = vertices

    return poses


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''

    device = poses.device

    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))  # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    if N > 0:

        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

        else:  # random sampling
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()

    else:
        inds = torch.arange(H * W, device=device)

    zs = -torch.ones_like(i)  # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy  # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1)  # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True) 
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(
        1)  # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]

    rays_o = poses[:, :3, 3].expand_as(rays_d)  # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results


def visualize_rays(rays_o, rays_d):
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(0, rays_o.shape[0], 10):
        ro = rays_o[i]
        rd = rays_d[i]

        segs = np.array([[ro, ro + rd * 3]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


# def torch_vis_2d(x, renormalize=False):
#     # x: [3, H, W], [H, W, 3] or [1, H, W] or [H, W]
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch
#
#     if isinstance(x, torch.Tensor):
#         if len(x.shape) == 3 and x.shape[0] == 3:
#             x = x.permute(1, 2, 0).squeeze()
#         x = x.detach().cpu().numpy()
#
#     print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
#
#     x = x.astype(np.float32)
#
#     # renormalize
#     if renormalize:
#         x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)
#
#     plt.imshow(x)
#     plt.show()


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

        return psnr

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='vgg', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

        return v

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


def create_point_pyramid(points):
    # points: Float[Tensor, "3 height width depth"]) -> List[Float[Tensor, "3 height width depth"]]
    """
    Create a point pyramid for multi-resolution evaluation.

    Args:
        points: A torch tensor containing 3D points.

    Returns:
        A list of torch tensors representing points at different resolutions.
    """
    points_pyramid = [points]
    for _ in range(3):
        points = torch.nn.AvgPool3d(2, stride=2)(points[None])[0]
        points_pyramid.append(points)
    points_pyramid = points_pyramid[::-1]
    return points_pyramid


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 save_interval=1,  # save once every $ epoch (independently from eval)
                 max_keep_ckpt=1,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 vosh_interval=None
                 ):

        self.opt = opt
        self.writer = None
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.vosh_interval = vosh_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.glctx = None

        # try out torch 2.0
        if torch.__version__[0] == '2':
            model = torch.compile(model)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler

        self.optimizer_fn = optimizer
        if optimizer is None:
            # naive adam
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        else:
            self.optimizer = self.optimizer_fn(self.model)

        self.lr_scheduler_fn = lr_scheduler
        if lr_scheduler is None:
            # fake scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)
        else:
            self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }
        self.measures = []

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        os.makedirs(self.workspace, exist_ok=True)
        self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
        self.log_ptr = open(self.log_path, "a+")

        self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
        self.best_path = f"{self.ckpt_path}/{self.name}.pth"
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.voxel_error_grid_path = None
        error_grid = None

        if self.opt.render != 'grid':
            self.continue_vosh = False
            self.occ_grid_path = None

            self.mesh_path = os.path.join(self.workspace, 'mesh')
            os.makedirs(self.mesh_path, exist_ok=True)

            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}*.pth'))

            if not checkpoint_list:
                checkpoint_list = sorted(glob.glob(f'{opt.vol_path}/checkpoints/mesh*.pth'))
                if not checkpoint_list or ('use_vol_pth' in self.opt and self.opt.use_vol_pth):
                    checkpoint_list = sorted(glob.glob(f'{opt.vol_path}/checkpoints/vol*.pth'))
                assert len(checkpoint_list), f'No checkpoints found in {opt.vol_path}/checkpoints !'
                checkpoint = checkpoint_list[-1]
                shutil.copy(checkpoint, self.ckpt_path)
                self.log(f"[INFO] Copy checkpoint from {checkpoint}")
                if 'mesh_encoder' in self.opt and self.opt.mesh_encoder:
                    mesh_ckpt = sorted(glob.glob(f'{opt.vol_path}/checkpoints/mesh*.pth'))[-1]
                    shutil.copy(mesh_ckpt, self.ckpt_path)
                    self.log(f"[INFO] Copy mesh checkpoint from {mesh_ckpt}")

                # occ_grid_list = sorted(glob.glob(f'{opt.vol_path}/merf_occ_grid_vol.pt'))
                # if len(occ_grid_list):
                #     self.occ_grid_path = occ_grid_list[-1]

            if self.opt.render == 'mesh':
                voxel_error_grid_list = sorted(glob.glob(f'{opt.vol_path}/voxel_error_grid*.pt'))
                if len(voxel_error_grid_list):
                    self.voxel_error_grid_path = voxel_error_grid_list[-1]
            # elif self.opt.render == 'mixed' and self.opt.mesh_select > 0:
            elif self.opt.render == 'mixed' and 0 < self.opt.mesh_select < 1.0:
                error_grid_list = sorted(glob.glob(f'{opt.vol_path}/error_grid*.pt'))
                if len(error_grid_list):
                    error_grid = torch.load(error_grid_list[-1]).to(self.device)
                    error_grid.requires_grad = False
                    self.log(f'[INFO] Load error grid {error_grid_list[-1]}!')
                    self.log(f'[INFO] error>0: {torch.sum(error_grid > 0)}, error<0: {torch.sum(error_grid < 0)}')
                    verts, faces = self.model.remove_faces_in_selected_voxels(error_grid,
                                                                              mesh_select=self.opt.mesh_select,
                                                                              keep_center=opt.keep_center)
                    self.log(f'[INFO] update mesh with {len(verts)} verts, {len(faces)} faces')

                #         error_grid = 1.0 - (error_grid - error_grid.min()) / (
                #                 error_grid.max() - error_grid.min())
                #
                #         if self.opt.ec_center > 0:
                #             center_mask = torch.ones_like(error_grid).bool()
                #             half_resolution = center_mask.shape[0] // 2
                #             half_itv = int(self.opt.ec_center * half_resolution)
                #             center_mask[half_resolution - half_itv:half_resolution + half_itv,
                #             half_resolution - half_itv:half_resolution + half_itv,
                #             half_resolution - half_itv:half_resolution + half_itv, ] = False
                #             error_grid[center_mask] = 1.0
                #             self.log(f'[INFO] Load error grid with center {self.opt.ec_center}!')
                # elif self.opt.render == 'mixed' and self.opt.lambda_ec_weight > 0 and self.opt.lambda_mesh_weight > 0:
                #     error_grid_list = sorted(glob.glob(f'{opt.vol_path}/error_grid*.pt'))
                #     if len(error_grid_list):
                #         error_grid = torch.load(error_grid_list[-1]).to(self.device)
                #         error_grid.requires_grad = False
                #         # error_grid[error_grid < 0] = 0
                #         # 1 - error_grid for mesh weight loss
                #         self.log(f'[INFO] Load error grid {error_grid_list[-1]}!')
                #         self.log(f'[INFO] error>0: {torch.sum(error_grid > 0)}, error<0: {torch.sum(error_grid < 0)}')
                #
                #         error_grid = 1.0 - (error_grid - error_grid.min()) / (
                #                 error_grid.max() - error_grid.min())
                #
                #         if self.opt.ec_center > 0:
                #             center_mask = torch.ones_like(error_grid).bool()
                #             half_resolution = center_mask.shape[0] // 2
                #             half_itv = int(self.opt.ec_center * half_resolution)
                #             center_mask[half_resolution - half_itv:half_resolution + half_itv,
                #             half_resolution - half_itv:half_resolution + half_itv,
                #             half_resolution - half_itv:half_resolution + half_itv, ] = False
                #             error_grid[center_mask] = 1.0
                #             self.log(f'[INFO] Load error grid with center {self.opt.ec_center}!')
                else:
                    raise Exception('No error grid found for E_convert!')

        if error_grid is not None:
            self.model.set_error_grid(error_grid)

        backup_path = os.path.join(self.workspace, 'backup')
        os.makedirs(backup_path, exist_ok=True)
        os.makedirs(os.path.join(backup_path, 'nerf'), exist_ok=True)

        file_list = os.listdir('.')
        for filename in os.listdir('nerf/'):
            file_list.append(os.path.join('nerf', filename))
        for filename in file_list:
            if filename.endswith('py'):
                shutil.copy(filename, os.path.join(backup_path, filename))

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | '
                 f'{"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.log(opt)
        self.log(self.model)

        if self.use_checkpoint == "scratch":
            self.log("[INFO] Training from scratch ...")
        elif self.use_checkpoint == "latest":
            self.log("[INFO] Loading latest checkpoint ...")
            self.load_checkpoint()
        elif self.use_checkpoint == "latest_model":
            self.log("[INFO] Loading latest checkpoint (model only)...")
            self.load_checkpoint(model_only=True)
        elif self.use_checkpoint == "best":
            if os.path.exists(self.best_path):
                self.log("[INFO] Loading best checkpoint ...")
                self.load_checkpoint(self.best_path)
            else:
                self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                self.load_checkpoint()
        else:  # path to ckpt
            self.log(f"[INFO] Loading {self.use_checkpoint} ...")
            self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):

        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None  # [1/N, 2] or None
        images = data['images']  # [N, 3/4]

        N, C = images.shape

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device)  # [N, 3], pixel-wise random.
        else:  # white / last_sample
            bg_color = 1

        if C == 4:
            gt_mask = images[..., 3:]
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_mask = None
            gt_rgb = images

        shading = 'diffuse' if self.global_step < self.opt.diffuse_step else 'full'
        update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0

        if self.opt.render == 'mesh' or self.opt.alpha_thres == 0:
            alpha_threshold = None
        else:
            alpha_threshold = self.model.alpha_threshold(self.global_step)

        if self.opt.render != 'grid':
            outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True,
                                        cam_near_far=cam_near_far, shading=shading, update_proposal=update_proposal,
                                        mvp=data['mvp'], rays_d_all=data['rays_d_all'],
                                        rays_i=data['rays_i'], rays_j=data['rays_j'], H=data['H'], W=data['W'],
                                        alpha_threshold=alpha_threshold)

        else:
            outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True,
                                        cam_near_far=cam_near_far, shading=shading, update_proposal=update_proposal,
                                        alpha_threshold=alpha_threshold)

        # loss
        pred_rgb = outputs['image']
        mask_mesh = None
        # if self.opt.render == 'mesh':
        #     mask_mesh = outputs['mask_mesh']
        #     pred_rgb, gt_rgb = pred_rgb[mask_mesh], gt_rgb[mask_mesh]
        #     outputs['specular'] = outputs['specular'][mask_mesh]

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)  # [N, 3] --> [N]
        loss_dict = {'loss_smoothL1': loss.mean().item()}

        if gt_mask is not None and self.opt.lambda_mask > 0:
            if self.opt.render == 'mixed':
                pred_mask = outputs['weights_vosh']
            else:
                pred_mask = outputs['weights_sum']
            # if self.opt.render == 'mesh':
            #     pred_mask, gt_mask = pred_mask[mask_mesh], gt_mask[mask_mesh]
            assert self.opt.criterion == 'MSE'
            loss = loss + self.opt.lambda_mask * self.criterion(pred_mask, gt_mask.squeeze(1))

        if self.opt.render == 'mesh' and self.opt.refine:
            self.model.update_triangles_errors(loss.detach(), mask_mesh)

        # depth loss
        if 'depth' in data:
            gt_depth = data['depth'].view(-1, 1)
            pred_depth = outputs['depth'].view(-1, 1)

            lambda_depth = self.opt.lambda_depth * min(1.0, self.global_step / 1000)
            mask = gt_depth > 0

            loss_depth = lambda_depth * self.criterion(pred_depth * mask, gt_depth * mask)  # [N]
            loss = loss + loss_depth
            loss_dict['loss_depth'] = loss_depth.mean().item()

        loss = loss.mean()

        # extra loss
        if 'specular' in outputs and self.opt.lambda_specular > 0:
            loss_spec = self.opt.lambda_specular * (outputs['specular'] ** 2).mean()
            loss = loss + loss_spec
            loss_dict['loss_spec'] = loss_spec.item()

        if update_proposal and 'proposal_loss' in outputs and self.opt.lambda_proposal > 0:
            loss_prop = self.opt.lambda_proposal * outputs['proposal_loss']
            loss = loss + loss_prop
            loss_dict['loss_prop'] = loss_prop.item()

        if 'distort_loss' in outputs and self.opt.lambda_distort > 0:
            # progressive to avoid bad init
            lambda_distort = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_distort
            loss_dist = lambda_distort * outputs['distort_loss']
            loss = loss + loss_dist
            loss_dict['loss_dist'] = loss_dist.item()

        if 'lambda_entropy' in self.opt and self.opt.lambda_entropy > 0:
            w = outputs['alphas'].clamp(1e-5, 1 - 1e-5)
            entropy = - w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
            loss_entro = self.opt.lambda_entropy * (entropy.mean())
            loss = loss + loss_entro
            loss_dict['loss_entro'] = loss_entro.item()

        if 'sparsity_loss' in outputs and self.opt.lambda_sparsity > 0:
            loss_spar = outputs['sparsity_loss'] * self.opt.lambda_sparsity
            loss = loss + loss_spar
            loss_dict['loss_spar'] = loss_spar.item()

        if self.opt.render == 'mixed':
            if self.opt.lambda_mesh_weight > 0:
                loss_mesh_weight = outputs['mesh_weight_loss'] * self.opt.lambda_mesh_weight
                loss = loss + loss_mesh_weight
                loss_dict['loss_mesh_weight'] = loss_mesh_weight.item()

            if self.opt.lambda_bg_weight > 0:
                loss_bg_weight = outputs['bg_weight_loss'] * self.opt.lambda_bg_weight
                loss = loss + loss_bg_weight
                loss_dict['loss_bg_weight'] = loss_bg_weight.item()

        # if 'lambda_offsets' in self.opt and self.opt.lambda_entropy > 0:
        #     abs_offsets_inner = self.model.vertices_offsets[:self.model.v_cumsum[1]].abs()
        #     abs_offsets_outer = self.model.vertices_offsets[self.model.v_cumsum[1]:].abs()
        #
        #     # inner mesh
        #     loss_offsets = (abs_offsets_inner ** 2).sum(-1).mean()
        #
        #     # outer mesh
        #     if self.opt.bound > 1:
        #         loss_offsets = loss_offsets + 0.1 * (abs_offsets_outer ** 2).sum(-1).mean()
        #
        #     loss = loss + self.opt.lambda_offsets * loss_offsets

        if 'lambda_lap' in self.opt and self.opt.lambda_lap > 0 and self.model.vertices_offsets is not None:
            loss_lap = laplacian_smooth_loss(self.model.vertices + self.model.vertices_offsets, self.model.triangles)
            loss = loss + self.opt.lambda_lap * loss_lap

        # adaptive num_rays
        if 'adaptive_num_rays' in self.opt and self.opt.adaptive_num_rays:
            self.opt.num_rays = int(round((self.opt.num_points / outputs['num_points']) * self.opt.num_rays))

        # if 'lambda_mesh_weight' in self.opt and self.opt.lambda_mesh_weight > 0:
        #     print(f"\nrgb : {loss:.5f}, "
        #           f"distort: {loss_dict['loss_dist']:.5f}, "
        #           f"mesh_weight: {loss_dict['loss_mesh_weight']:.5f}")

        return pred_rgb, gt_rgb, loss, loss_dict

    def post_train_step(self):

        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            # # progressive...
            # lambda_tv = min(1.0, self.global_step / 10000) * self.opt.lambda_tv
            lambda_tv = self.opt.lambda_tv

            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)

            # different tv weights for inner and outer points
            self.model.apply_total_variation(lambda_tv)

    def eval_step(self, data):

        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        images = data['images']  # [H, W, 3/4]
        H, W, C = images.shape

        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None  # [1/N, 2] or None

        # eval with fixed white background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        if self.opt.render == 'mesh' or self.opt.alpha_thres == 0:
            alpha_threshold = None
        else:
            alpha_threshold = self.model.alpha_threshold(self.global_step)

        if self.opt.render != 'grid':
            outputs = self.model.render(rays_o, rays_d, bg_color=bg_color, perturb=False,
                                        cam_near_far=cam_near_far, mvp=data['mvp'], rays_d_all=data['rays_d_all'],
                                        rays_i=data['rays_i'], rays_j=data['rays_j'], H=data['H'], W=data['W'],
                                        alpha_threshold=alpha_threshold)
        else:
            outputs = self.model.render(rays_o, rays_d, bg_color=bg_color, perturb=False,
                                        cam_near_far=cam_near_far, alpha_threshold=alpha_threshold)

        # pred = {'rgb': outputs['image'].reshape(H, W, 3),
        #         'depth': outputs['depth'].reshape(H, W), }
        #
        # if 'diffuse' in outputs.keys():
        #     pred['diffuse'] = outputs['diffuse'].reshape(H, W, 3)
        # if 'specular' in outputs.keys():
        #     pred['specular'] = outputs['specular'].reshape(H, W, 3)
        # if 'dirs' in outputs.keys():
        #     pred['dirs'] = outputs['dirs'].reshape(H, W, 3)
        #
        # if 'depth_mesh' in outputs.keys():
        #     pred['depth_mesh'] = outputs['depth_mesh'].reshape(H, W)
        #     pred['mask_mesh'] = outputs['mask_mesh'].reshape(H, W)
        #     pred['alphas_mesh'] = outputs['alphas_mesh'].reshape(H, W)

        pred = {'rgb': outputs['image'].reshape(H, W, 3)}

        for k in outputs.keys():
            if k == 'image': continue
            if outputs[k].numel() == H * W * 3:
                pred[k] = outputs[k].reshape(H, W, 3)
            elif outputs[k].numel() == H * W:
                pred[k] = outputs[k].reshape(H, W)
            else:
                pred[k] = outputs[k]

        loss = self.criterion(pred['rgb'], gt_rgb).mean()

        return pred, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, shading='full'):

        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        H, W = data['H'], data['W']

        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None  # [1/N, 2] or None

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        if self.opt.render == 'mesh' or self.opt.alpha_thres == 0:
            alpha_threshold = None
        else:
            alpha_threshold = self.model.alpha_threshold(self.global_step)

        if self.opt.render != 'grid':
            outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=perturb,
                                        cam_near_far=cam_near_far, shading=shading,
                                        mvp=data['mvp'], rays_d_all=data['rays_d_all'],
                                        H=data['H'], W=data['W'], alpha_threshold=alpha_threshold)
        else:
            outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=perturb,
                                        cam_near_far=cam_near_far, shading=shading, alpha_threshold=alpha_threshold)

        pred = {'rgb': outputs['image'].reshape(H, W, 3)}

        for k in outputs.keys():
            if k == 'image': continue
            if outputs[k].numel() == H * W * 3:
                pred[k] = outputs[k].reshape(H, W, 3)
            elif outputs[k].numel() == H * W:
                pred[k] = outputs[k].reshape(H, W)
            else:
                pred[k] = outputs[k]

        return pred

    @torch.no_grad()
    def save_baking(self, loader=None, occupancy_grid=None, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'assets')
        os.makedirs(save_path, exist_ok=True)

        assert self.opt.triplane_resolution <= 0

        self.model.eval()

        self.log(f"==> Start Baking, save results to {save_path}")

        device = self.device
        bound = self.model.bound  # always 2 in contracted mode
        BS = 8  # block size
        LS = self.opt.grid_resolution
        AS = LS // BS  # 64, atlas size
        # thresh = 0.005  # 0.005 in paper
        scene_params = {
            'sparse_grid_resolution': LS,
            "sparse_grid_voxel_size": 2 * bound / LS,  # 4/1024
            'data_block_size': BS,
            'has_vol': True,
            'slice_depth': 1,
            "format": "png",
            'range_features': [-7.0, 7.0],
            'range_density': [-14.0, 14.0],
        }

        # covers the [-2, 2]^3 space
        occupancy_rate = occupancy_grid.cpu().sum().item() / occupancy_grid.numel() * 100.
        if occupancy_grid is None:
            if loader is not None:
                occupancy_grid = self.cal_occ_grid(loader)
            else:
                raise ValueError('No loader or occupancy_grid is provided.')
        else:
            self.log(f'[INFO] Occupancy rate: '
                     f'{occupancy_grid.cpu().sum().item() / occupancy_grid.numel() * 100.:.4f}% '
                     f'with resolution {occupancy_grid.shape[0]}')

        # if self.opt.render == 'mixed':
        #     self.model.update_mesh_occ_mask(loader)
        #     # occupancy_grid *= ~self.model.mesh_occ_mask
        #     mesh_occ_mask = self.model.mesh_occ_mask if self.opt.mesh_check_ratio == 1 else (
        #         F.interpolate(self.model.mesh_occ_mask.float().unsqueeze(0).unsqueeze(0), size=occupancy_grid.shape,
        #                       mode='nearest').squeeze(0).squeeze(0))
        #     self.log(f'[INFO] mesh_check_ratio: {self.opt.mesh_check_ratio}')
        #     self.log(f'[INFO] mesh_occ_grid: {mesh_occ_mask.sum().item()} and '
        #              f'occupancy_grid: {occupancy_grid.sum().item()}')
        #     self.log(
        #         f'[INFO] mesh_occ_grid in occupancy_grid: '
        #         f'{torch.where(self.model.mesh_occ_mask.cpu(), occupancy_grid,
        #                        torch.zeros_like(occupancy_grid)).sum().item()}')
        #     occupancy_grid = torch.where(mesh_occ_mask, torch.zeros_like(occupancy_grid), occupancy_grid)
        #     self.log(f'[INFO] Occupancy rate after mesh_occ_grid: '
        #              f'{occupancy_grid.sum().item() / occupancy_grid.numel() * 100.:.4f}% '
        #              f'with resolution {occupancy_grid.shape[0]}')
        #     torch.save(self.model.mesh_occ_mask.cpu(), os.path.join(self.opt.workspace, 'mesh_occ_mask_cpu_vosh.pt'))

        # maxpooling
        occupancy_grid_L1 = (F.max_pool3d(occupancy_grid.float().unsqueeze(0).unsqueeze(0), 8, stride=8)
                             .squeeze(0).squeeze(0)).bool()  # LS // 8
        self.log(f'[INFO] Occ blocks: {occupancy_grid_L1.sum().item()} with resolution {occupancy_grid_L1.shape[0]}')

        if self.opt.render == 'mixed' and self.opt.use_mesh_occ_grid:
            self.model.update_mesh_occ_mask(loader)
            self.log(f'[INFO] mesh_check_ratio: {self.opt.mesh_check_ratio}')
            mesh_occ_mask = F.interpolate(self.model.mesh_occ_mask.float().unsqueeze(0).unsqueeze(0),
                                          size=occupancy_grid_L1.shape, mode='nearest').squeeze(0).squeeze(0).bool()
            self.log(f'[INFO] mesh_occ_mask: {mesh_occ_mask.sum().item()} with resolution {mesh_occ_mask.shape[0]}')
            self.log(f'[INFO] mesh_occ_grid in occupancy_grid_L1: '
                     f'{torch.where(mesh_occ_mask, occupancy_grid_L1, torch.zeros_like(occupancy_grid_L1)).sum().item()}')
            occupancy_grid_L1 = torch.where(mesh_occ_mask, torch.zeros_like(occupancy_grid_L1), occupancy_grid_L1)
            self.log(f'[INFO] Occ blocks after mesh_occ_grid: {occupancy_grid_L1.sum().item()} '
                     f'with resolution {occupancy_grid_L1.shape[0]}')

        occupancy_grid_L1 = occupancy_grid_L1.cpu()
        occupancy_grid_L2 = (F.max_pool3d(occupancy_grid_L1.float().unsqueeze(0).unsqueeze(0), 2, stride=2)
                             .squeeze(0).squeeze(0)).bool()  # LS // 16
        occupancy_grid_L3 = (F.max_pool3d(occupancy_grid_L2.float().unsqueeze(0).unsqueeze(0), 2, stride=2)
                             .squeeze(0).squeeze(0)).bool()  # LS // 32
        occupancy_grid_L4 = (F.max_pool3d(occupancy_grid_L3.float().unsqueeze(0).unsqueeze(0), 2, stride=2)
                             .squeeze(0).squeeze(0)).bool()  # LS // 64
        occupancy_grid_L5 = (F.max_pool3d(occupancy_grid_L4.float().unsqueeze(0).unsqueeze(0), 2, stride=2)
                             .squeeze(0).squeeze(0)).bool()  # LS // 128

        # grid features
        coords = torch.nonzero(occupancy_grid_L1)  # [N, 3]

        if coords.shape[0] == 0:
            coords = torch.tensor([[0, 0, 0]])

        # total occ blocks to save
        N_occ = coords.shape[0]
        # per depth slice we can save at most 255x255 occ blocks.
        # due to limit of max texture size (2048), we actually only have 2048 / 9 = 227 max size
        atlas_blocks_z = N_occ // (227 * 227) + 1
        N_per_slice = math.ceil(N_occ / atlas_blocks_z)
        atlas_blocks_x = math.ceil(math.sqrt(N_per_slice))
        atlas_blocks_y = math.ceil(N_per_slice / atlas_blocks_x)
        atlas_width = 9 * atlas_blocks_x
        atlas_height = 9 * atlas_blocks_y
        atlas_depth = 9 * atlas_blocks_z

        self.log(f'atlas block size: ({atlas_blocks_x}, {atlas_blocks_y}, {atlas_blocks_z})')

        has_vol = False if occupancy_rate * 10000.0 < 1.0 else True

        scene_params['atlas_width'] = atlas_width
        scene_params['atlas_height'] = atlas_height
        scene_params['atlas_depth'] = atlas_depth
        scene_params['num_slices'] = atlas_depth
        scene_params['atlas_blocks_x'] = atlas_blocks_x
        scene_params['atlas_blocks_y'] = atlas_blocks_y
        scene_params['atlas_blocks_z'] = atlas_blocks_z

        scene_params['has_vol'] = has_vol

        self.log('has vol: {}'.format(has_vol))

        atlas_indices = 255 * torch.ones([AS, AS, AS, 3], dtype=torch.long)  # [64, 64, 64, 3]
        atlas_data = torch.zeros([atlas_width, atlas_height, atlas_depth, 8], dtype=torch.float32)  # [W, H, D, 8]

        # grid indice
        indices = torch.arange(N_occ)
        indices_w = indices // (atlas_blocks_y * atlas_blocks_z)
        indices_hd = indices % (atlas_blocks_y * atlas_blocks_z)
        indices_h = indices_hd // atlas_blocks_z
        indices_d = indices_hd % atlas_blocks_z

        # xyz --> whd
        atlas_indices[tuple(coords.T)] = torch.stack([indices_w, indices_h, indices_d], dim=1)  # [N, 3]

        # convert coords to 9x9x9 xyzs
        xyzs_000 = (coords / AS) * 2 * bound - bound  # [N, 3], left-top point of the 9x9x9 block
        grid_size = 2 * bound / AS
        step_size = grid_size / BS

        # loops
        if self.opt.grid_resolution > 0:
            self.log(f'[INFO] baking grid features...')
            for i in range(BS + 1):
                for j in range(BS + 1):
                    for k in range(BS + 1):
                        # print(f'[INFO] baking features for grid ({i}, {j}, {k})...')
                        # [N, 3]
                        # xyzs = (xyzs_000 + torch.tensor([i, j, k], dtype=torch.float32) * step_size).to(device)
                        xyzs = (xyzs_000 + step_size * 0.5 +
                                torch.tensor([i, j, k], dtype=torch.float32) * step_size).to(device)
                        # query features
                        # f_grid = self.model.quantize_feature(self.model.grid(xyzs, self.model.bound),
                        #                                      baking=True).cpu()
                        # features, density = self.model.query_representation(xyzs, baking=True)
                        features, density = self.model.DensityAndFeaturesMLP(xyzs, bound=bound)
                        features = quantize.quantize_float_to_byte(torch.sigmoid(features))
                        density = quantize.quantize_float_to_byte(torch.sigmoid(density))

                        f_grid = (torch.cat([features[..., :3], density,
                                             features[..., 3:]], dim=-1)).cpu()

                        # plot_pointcloud(xyzs.cpu().numpy(), torch.sigmoid(features[..., :3]).cpu().numpy())

                        # place in grid (note the coordinate conventions!)
                        atlas_data[indices_w * (BS + 1) + j, indices_h * (BS + 1) + k,
                                   indices_d * (BS + 1) + i] = f_grid

        # save all assets

        def imwrite(f, x):
            if x.shape[-1] == 1:
                cv2.imwrite(f, x)
            elif x.shape[-1] == 3:
                cv2.imwrite(f, cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(f, cv2.cvtColor(x, cv2.COLOR_RGBA2BGRA))

        # occupancy_grid_8-128.png
        imwrite(os.path.join(save_path, 'occupancy_grid_8.png'),
                occupancy_grid_L1.cpu().numpy().transpose(0, 2, 1).reshape(
                    occupancy_grid_L1.shape[0] * occupancy_grid_L1.shape[0], occupancy_grid_L1.shape[0], 1)
                .astype(np.uint8).repeat(4, axis=-1))
        imwrite(os.path.join(save_path, 'occupancy_grid_16.png'),
                occupancy_grid_L2.cpu().numpy().transpose(0, 2, 1).reshape(
                    occupancy_grid_L2.shape[0] * occupancy_grid_L2.shape[0], occupancy_grid_L2.shape[0], 1)
                .astype(np.uint8).repeat(4, axis=-1))
        imwrite(os.path.join(save_path, 'occupancy_grid_32.png'),
                occupancy_grid_L3.cpu().numpy().transpose(0, 2, 1).reshape(
                    occupancy_grid_L3.shape[0] * occupancy_grid_L3.shape[0], occupancy_grid_L3.shape[0], 1)
                .astype(np.uint8).repeat(4, axis=-1))
        imwrite(os.path.join(save_path, 'occupancy_grid_64.png'),
                occupancy_grid_L4.cpu().numpy().transpose(0, 2, 1).reshape(
                    occupancy_grid_L4.shape[0] * occupancy_grid_L4.shape[0], occupancy_grid_L4.shape[0], 1)
                .astype(np.uint8).repeat(4, axis=-1))
        imwrite(os.path.join(save_path, 'occupancy_grid_128.png'),
                occupancy_grid_L5.cpu().numpy().transpose(0, 2, 1).reshape(
                    occupancy_grid_L5.shape[0] * occupancy_grid_L5.shape[0], occupancy_grid_L5.shape[0], 1)
                .astype(np.uint8).repeat(4, axis=-1))

        # atlas_indices.png
        imwrite(os.path.join(save_path, 'sparse_grid_block_indices.png'),
                atlas_indices.cpu().clamp(0, 255).numpy().transpose(0, 2, 1, 3).reshape(AS * AS, AS, 3).astype(
                    np.uint8))

        # feature_xxx.png & rgba_xxx.png
        for i in range(atlas_depth):
            # density = atlas_data[..., i, :1]
            # rgb = atlas_data[..., i, 1:4]
            # rgb_and_density = torch.cat([rgb, density], dim=-1)  # they place density after rgb...
            rgb_and_density = atlas_data[..., i, :4]
            feature = atlas_data[..., i, 4:]

            print(f'[INFO] grid pre {i}: '
                  f'rgb: {atlas_data[..., i, :3].min().item()} ~ {atlas_data[..., i, :3].max().item()} '
                  f'density: {atlas_data[..., i, 3].min().item()} ~ {atlas_data[..., i, 3].max().item()} '
                  f'f: {atlas_data[..., i, 4:].min().item()} ~ {atlas_data[..., i, 4:].max().item()}')
            # print(f'[INFO] grid slice {i}: '
            #       f'density: {density.min().item()} ~ {density.max().item()} '
            #       f'rgb: {rgb.min().item()} ~ {rgb.max().item()} '
            #       f'f: {feature.min().item()} ~ {feature.max().item()}')

            imwrite(os.path.join(save_path, f'sparse_grid_rgb_and_density_{i:03d}.png'),
                    rgb_and_density.cpu().numpy().transpose(1, 0, 2).astype(np.uint8))
            imwrite(os.path.join(save_path, f'sparse_grid_features_{i:03d}.png'),
                    feature.cpu().numpy().transpose(1, 0, 2).astype(np.uint8))

        # mlp
        params = dict(self.model.DeferredMLP.view_mlp.named_parameters())

        for k, p in params.items():
            p_np = p.detach().cpu().numpy().T
            # 'net.0.weight' --> '0_weights'
            # 'net.0.bias' --> '0_bias'
            scene_params[k[4:].replace('.', '_').replace('weight', 'weights')] = p_np.tolist()

        # save scene_params.json
        with open(os.path.join(save_path, 'scene_params.json'), 'w') as f:
            json.dump(scene_params, f, indent=2)

        if self.opt.render == 'mixed':
            self.log(f"==> Start Exporting mesh, save results to {save_path}")
            os.makedirs(os.path.join(save_path, 'mesh'), exist_ok=True)
        else:
            self.log(f"==> Finished baking.")

    @torch.no_grad()
    def cal_occ_grid(self, loader, resolution=None):
        self.model.eval()
        self.model.set_training(False)
        bound = self.model.bound  # always 2 in contracted mode
        if resolution is None:
            resolution = self.opt.grid_resolution

        # covers the [-2, 2]^3 space
        occupancy_grid = torch.zeros([resolution, resolution, resolution], dtype=torch.bool,
                                     device=self.device, requires_grad=False)

        if self.opt.render == 'mesh' or self.opt.alpha_thres == 0:
            alpha_threshold = None
        else:
            alpha_threshold = self.model.alpha_threshold(self.global_step)

        self.log(f"==> Calculating occupancy_grid...")
        for i, data in enumerate(tqdm.tqdm(loader)):
            rays_o = data['rays_o']  # [N, 3]
            rays_d = data['rays_d']  # [N, 3]

            cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None  # [1/N, 2] or None

            if self.opt.render == 'mixed':
                outputs = self.model.render(rays_o, rays_d, cam_near_far=cam_near_far, shading='diffuse', baking=True,
                                            mvp=data['mvp'], rays_d_all=data['rays_d_all'],
                                            H=data['H'], W=data['W'], alpha_threshold=alpha_threshold)
            else:
                outputs = self.model.render(rays_o, rays_d, cam_near_far=cam_near_far, shading='diffuse', baking=True,
                                            H=data['H'], W=data['W'], alpha_threshold=alpha_threshold)

            xyzs = outputs['xyzs'].reshape(-1, 3)
            weights = outputs['weights'].reshape(-1)
            alphas = outputs['alphas'].reshape(-1)

            # thresholding
            mask = (weights > self.opt.alpha_thres) & (alphas > self.opt.alpha_thres)
            xyzs = xyzs[mask]  # [N*T, 3] in [-2, 2]

            # mark occupancy, write data
            # [-bound, bound] -> [0, resolution-1]
            coords = torch.floor((xyzs + bound) / (2 * bound) * resolution).long().clamp(0, resolution - 1)  # [N*T, 3]
            occupancy_grid[tuple(coords.T)] = 1

        # occupancy_grid = occupancy_grid.cpu()
        self.log(f'[INFO] Occupancy rate: '
                 f'{occupancy_grid.cpu().sum().item() / (occupancy_grid.shape[0] ** 3) * 100.:.2f}% '
                 f'with resolution {resolution} and N = {rays_o.shape[0]}')
        self.model.set_training(True)
        return occupancy_grid

    # @torch.no_grad()
    # def run_mesh_error_epoch(self, loader, resolution=None):
    #     bound = self.model.bound  # always 2 in contracted mode
    #     if resolution is None:
    #         resolution = self.opt.mcubes_reso
    #     mesh_occ_mask = torch.zeros([resolution] * 3, dtype=torch.bool, device=self.device)
    #
    #     self.log(f"==> Calculating mesh_error_grid...")
    #
    #     # get mesh error
    #     for i, data in enumerate(tqdm.tqdm(loader)):
    #         results_mesh = self.model.render_mesh(rays_o=None, rays_d=data['rays_d_all'],
    #                                               h0=data['H'], w0=data['W'], mvp=data['mvp'])
    #         dirs = data['rays_d_all'] / torch.norm(data['rays_d_all'], dim=-1, keepdim=True)
    #         dirs = self.model.view_encoder(dirs)
    #         image = (results_mesh['features'][:, :3] +
    #                  self.model.view_mlp(torch.cat([results_mesh['features'][:, :3],
    #                                                 results_mesh['features'][:, 3:],
    #                                                 dirs], dim=1)))
    #         image = image.clamp(0, 1)
    #         loss = self.criterion(image, data['images_all']).mean(-1)  # [N, 3] --> [N]
    #
    #         self.model.update_triangles_errors(loss.detach())
    #
    #         xyzs = contract(results_mesh['xyzs'][results_mesh['mask']])
    #         coords = torch.floor((xyzs + bound) / (2 * bound) * resolution).long().clamp(0, resolution - 1)  # [N, 3]
    #         mesh_occ_mask[tuple(coords.T)] = 1
    #
    #     return self.model.get_error_grid(), mesh_occ_mask

    @torch.no_grad()
    def cal_diff(self, loader):
        mesh_error_grid_list = sorted(glob.glob(f'{self.opt.workspace}/mesh_error_grid*.pt'))
        if mesh_error_grid_list:
            print(f'[INFO] Load mesh_error_grid from {mesh_error_grid_list[-1]}')
            mesh_error_grid = torch.load(mesh_error_grid_list[-1]).to(self.device).half()
        else:
            mesh_error_grid = self.run_mesh_error_epoch(loader).half()
            torch.save(mesh_error_grid, os.path.join(self.opt.workspace, 'mesh_error_grid.pt'))

        # voxel_error_grid_list = sorted(glob.glob(f'{self.opt.workspace}/voxel_error_grid*.pt'))
        # if voxel_error_grid_list:
        #     print(f'[INFO] Load voxel_error_grid from {voxel_error_grid_list[-1]}')
        #     voxel_error_grid = torch.load(voxel_error_grid_list[-1]).to(self.device)
        # else:
        #     voxel_error_grid = self.run_voxel_error_epoch(loader)
        #     torch.save(voxel_error_grid, os.path.join(self.opt.workspace, 'voxel_error_grid.pt'))

        assert self.voxel_error_grid_path is not None, 'Please specify the path of voxel_error_grid'
        print(f'[INFO] Load voxel_error_grid from {self.voxel_error_grid_path}')
        voxel_error_grid = torch.load(self.voxel_error_grid_path).to(self.device).half()

        if mesh_error_grid.shape[0] < voxel_error_grid.shape[0]:
            voxel_error_grid = F.interpolate(voxel_error_grid.unsqueeze(0).unsqueeze(0).float(),
                                             size=[mesh_error_grid.shape[0]] * 3,
                                             mode='trilinear').squeeze(0).squeeze(0)
        elif mesh_error_grid.shape[0] > voxel_error_grid.shape[0]:
            mesh_error_grid = F.interpolate(mesh_error_grid.unsqueeze(0).unsqueeze(0).float(),
                                            size=[voxel_error_grid.shape[0]] * 3,
                                            mode='trilinear').squeeze(0).squeeze(0)

        error_grid = mesh_error_grid - voxel_error_grid
        error_grid[voxel_error_grid == 0] = 0
        error_grid[mesh_error_grid == 0] = 0
        # error_grid = (error_grid - error_grid.min()) / (error_grid.max() - error_grid.min())
        torch.save(error_grid, os.path.join(self.opt.workspace, 'error_grid.pt'))
        return error_grid

    @torch.no_grad()
    def run_mesh_error_epoch(self, loader, resolution=None):
        # bound = self.model.bound  # always 2 in contracted mode
        if resolution is None:
            resolution = self.opt.grid_resolution
        mesh_error_grid = torch.zeros([resolution] * 3, dtype=torch.float32, device=self.device)
        mesh_cnt_grid = torch.zeros([resolution] * 3, dtype=torch.long, device=self.device)

        self.log(f"==> Calculating mesh_error_grid...")

        # get mesh error
        for i, data in enumerate(tqdm.tqdm(loader)):
            results = self.model.run_surface_render(rays_o=data['rays_o_all'], rays_d=data['rays_d_all'],
                                                    H=data['H'], W=data['W'], mvp=data['mvp'],
                                                    rays_d_all=data['rays_d_all'])
            image = results['image']
            mask = results['mask_mesh']

            C = data['images'].shape[-1]
            bg_color = 1
            if C == 4:
                gt_mask = data['images'][..., 3:]
                gt_rgb = data['images'][..., :3] * data['images'][..., 3:] + bg_color * (1 - data['images'][..., 3:])
            else:
                gt_mask = None
                gt_rgb = data['images']

            loss = self.criterion(image, gt_rgb).mean(-1)[mask]  # [the number of 1 in mask]
            if self.opt.contract:
                xyzs_old = results['xyzs'][mask]
                xyzs = contract(xyzs_old)  # [the number of 1 in mask, 3] in (-2, 2)
            else:
                xyzs = results['xyzs'][mask]  # [the number of 1 in mask, 3]

            if xyzs.shape[0] > 0:
                coords = (torch.floor((xyzs + self.model.bound) / (2 * self.model.bound) * resolution)
                          .long().clamp(0, resolution - 1))  # [N*T, 3]
                # mesh_cnt_grid[tuple(coords.T)] += 1
                # mesh_error_grid[tuple(coords.T)] += loss
                mesh_cnt_grid.index_put_((coords[:, 0], coords[:, 1], coords[:, 2]),
                                         torch.ones_like(coords[:, 0]), accumulate=True)
                mesh_error_grid.index_put_((coords[:, 0], coords[:, 1], coords[:, 2]), loss, accumulate=True)

            # self.model.update_triangles_errors(loss.detach())

        mesh_cnt_grid[mesh_cnt_grid < 1] = 1
        mesh_error_grid = mesh_error_grid.half()
        mesh_error_grid = mesh_error_grid.div(mesh_cnt_grid)
        return mesh_error_grid

    @torch.no_grad()
    def run_voxel_error_epoch(self, loader, resolution=None, thresh=0.005):
        self.model.set_training(False)
        if resolution is None:
            resolution = self.opt.grid_resolution
        voxel_error_grid = torch.zeros([resolution, resolution, resolution], dtype=torch.float32, device=self.device)
        voxel_cnt_grid = torch.zeros([resolution, resolution, resolution], dtype=torch.long, device=self.device)

        self.log(f"==> Calculating voxel_error_grid...")
        if self.opt.render == 'mesh' or self.opt.alpha_thres == 0:
            alpha_threshold = None
        else:
            alpha_threshold = self.model.alpha_threshold(self.global_step)

        # get grid error
        for i, data in enumerate(tqdm.tqdm(loader)):
            cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None  # [1/N, 2] or None
            results = self.model.render(data['rays_o'], data['rays_d'], cam_near_far=cam_near_far,
                                        H=data['H'], W=data['W'], alpha_threshold=alpha_threshold)
            # results = self.model.render(data['rays_o'], data['rays_d'], cam_near_far=cam_near_far,
            # run_voxel_error=True, mvp=data['mvp'], rays_d_all=data['rays_d_all'],
            # rays_i=data['rays_i'], rays_j=data['rays_j'], H=data['H'], W=data['W'])

            image = results['image']  # [N, 3]
            weights = results['weights']  # [N, T]
            alphas = results['alphas']

            # thresholding
            mask = (weights > thresh) & (alphas > thresh)
            weights[~mask] = 0

            C = data['images'].shape[-1]
            bg_color = 1
            if C == 4:
                gt_mask = data['images'][..., 3:]
                gt_rgb = data['images'][..., :3] * data['images'][..., 3:] + bg_color * (1 - data['images'][..., 3:])
            else:
                gt_mask = None
                gt_rgb = data['images']

            # [N*T]
            loss = (self.criterion(image, gt_rgb.view(-1, 3)).mean(-1).detach()[:, None] * weights).view(-1)
            mask = loss > 0

            xyzs = results['xyzs'].view(-1, 3)[mask]  # [the number of 1 in mask, 3] in (-2, 2)

            if xyzs.shape[0] > 0:
                coords = (torch.floor((xyzs + self.model.bound) / (2 * self.model.bound) * resolution)
                          .long().clamp(0, resolution - 1))  # [N*T, 3]
                # voxel_cnt_grid[tuple(coords.T)] += 1
                # voxel_error_grid[tuple(coords.T)] += loss[mask]
                voxel_cnt_grid.index_put_((coords[:, 0], coords[:, 1], coords[:, 2]),
                                          torch.ones_like(coords[:, 0]), accumulate=True)
                voxel_error_grid.index_put_((coords[:, 0], coords[:, 1], coords[:, 2]), loss[mask], accumulate=True)

            # self.model.update_triangles_errors(loss.detach())

        voxel_cnt_grid[voxel_cnt_grid < 1] = 1
        voxel_error_grid = voxel_error_grid.half()
        voxel_error_grid = voxel_error_grid.div(voxel_cnt_grid)
        # print('[INFO] Voxel_error_grid: ', voxel_error_grid.shape, voxel_error_grid.min(), voxel_error_grid.max())
        return voxel_error_grid

    # @torch.no_grad()
    # def cal_mesh_occ_grid(self, loader, occ_grid=None, resolution=None):
    #
    #     self.model.eval()
    #     bound = self.model.bound  # always 2 in contracted mode
    #     # thresh = 0.005  # 0.005 in paper
    #     if resolution is None:
    #         resolution = self.opt.grid_resolution
    #
    #     # covers the [-2, 2]^3 space
    #     mesh_occ_mask = torch.zeros([resolution, resolution, resolution], dtype=torch.bool, device=self.device)
    #
    #     self.log(f"==> Calculating mesh_occ_grid...")
    #     # mark mesh position
    #     for i, data in enumerate(tqdm.tqdm(loader)):
    #         results_mesh = self.model.render_mesh(rays_o=None, rays_d=data['rays_d_all'],
    #                                               h0=data['H'], w0=data['W'], mvp=data['mvp'], early_return=True)
    #
    #         # mark occupancy, write data
    #         # [-bound, bound] -> [0, resolution-1]
    #         xyzs = contract(results_mesh['xyzs'][results_mesh['mask']])
    #         coords = torch.floor((xyzs + bound) / (2 * bound) * resolution).long().clamp(0, resolution - 1)  # [N, 3]
    #         mesh_occ_mask[tuple(coords.T)] = 1
    #
    #     # self.log(f'[INFO] Mesh_occ_mask rate: '
    #     #          f'{mesh_occ_mask.float().sum().item() / mesh_occ_mask.numel() * 100.:.2f}% '
    #     #          f'with resolution {resolution}')
    #     # torch.save(mesh_occ_mask, os.path.join(self.opt.workspace, 'mesh_occ_grid.pt'))
    #
    #     if occ_grid is not None:
    #         self.log(f'[INFO] Mesh in occ_grid rate: '
    #                  f'{(mesh_occ_mask * occ_grid).float().sum().item() / occ_grid.numel() * 100.:.2f}% '
    #                  f'with resolution {resolution}')
    #     return mesh_occ_mask

    @torch.no_grad()
    def mark_unseen_triangles(self, vertices, triangles, mvps, H, W):
        # vertices: coords in world system
        # mvps: [B, 4, 4]
        device = self.device

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).contiguous().float().to(device)

        if isinstance(triangles, np.ndarray):
            triangles = torch.from_numpy(triangles).contiguous().int().to(device)

        mask = torch.zeros_like(triangles[:, 0])  # [M,], for face.

        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(output_db=False)

        for mvp in tqdm.tqdm(mvps):
            vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0),
                                         torch.transpose(mvp.to(device), 0, 1)).float().unsqueeze(0)  # [1, N, 4]

            # ENHANCE: lower resolution since we don't need that high?
            rast, _ = dr.rasterize(self.glctx, vertices_clip, triangles, (H, W))  # [1, H, W, 4]

            # collect the triangle_id (it is offseted by 1)
            trig_id = rast[..., -1].long().view(-1) - 1

            # no need to accumulate, just a 0/1 mask.
            mask[trig_id] += 1  # wrong for duplicated indices, but faster.
            # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)

        mask = (mask == 0)  # unseen faces by all cameras

        self.log(f'[mark unseen trigs] {mask.sum()} from {mask.shape[0]}')

        return mask  # [N]

    @torch.no_grad()
    def get_occ_grid(self, loader):
        if self.occ_grid_path is not None:
            occ_grid = torch.load(self.occ_grid_path).to(self.model.device)
            self.log(f'[INFO] Load occ_grid from {self.occ_grid_path}')
            self.log(f'[INFO] Occupancy rate: '
                     f'{occ_grid.float().sum().item() / occ_grid.numel() * 100.:.4f}% '
                     f'with resolution {occ_grid.shape[0]}')
        else:
            occ_grid = self.cal_occ_grid(loader)
            # torch.save(occ_grid, os.path.join(self.opt.workspace, 'merf_occ_grid_cpu.pt'))
        return occ_grid

    # @torch.no_grad()
    # def voxel_to_mesh(self, resolution=1024, S=128, loader=None):
    #
    #     def export_mesh(s_path, vv, tt, q_bound):
    #         m = trimesh.Trimesh(vv, tt, process=False)
    #         s_path = os.path.join(s_path, f'mesh_{q_bound}.ply')
    #         m.export(s_path)
    #         self.log(f'[INFO] Saved mesh into {s_path}')
    #         self.log('[INFO] <============================>')
    #
    #     self.log('[INFO] <============================>')
    #     self.log('[INFO] Extracting mesh from grid...')
    #     save_path = os.path.join(self.workspace, 'mesh')
    #     os.makedirs(save_path, exist_ok=True)
    #     device = self.device
    #     bound = self.model.bound  # always 2 in contracted mode
    #     real_bound = self.model.real_bound  # 128
    #     # sequentially load cascaded meshes
    #     """
    #         Computes the isosurface of a signed distance function (SDF) defined by the
    #         callable `sdf` in a given bounding box with a specified resolution. The SDF
    #         is sampled at a set of points within a regular grid, and the marching cubes
    #         algorithm is used to generate a mesh that approximates the isosurface at a
    #         specified isovalue `level`.
    #
    #         Args:
    #             sdf: A callable function that takes as input a tensor of size
    #                 (N, 3) containing 3D points, and returns a tensor of size (N,) containing
    #                 the signed distance function evaluated at those points.
    #             output_path: The output directory where the resulting mesh will be saved.
    #             resolution: The resolution of the grid used to sample the SDF.
    #             bounding_box_min: The minimum coordinates of the bounding box in which the SDF
    #                 will be evaluated.
    #             bounding_box_max: The maximum coordinates of the bounding box in which the SDF
    #                 will be evaluated.
    #             isosurface_threshold: The isovalue at which to approximate the isosurface.
    #             coarse_mask: A binary mask tensor of size ("height", "width", "depth") that indicates the regions
    #                 of the bounding box where the SDF is expected to have a zero-crossing. If
    #                 provided, the algorithm first evaluates the SDF at the coarse voxels where
    #                 the mask is True, and then refines the evaluation within these voxels using
    #                 a multi-scale approach. If None, evaluates the SDF at all points in the
    #                 bounding box.
    #         Returns:
    #             A torch tensor with the SDF values evaluated at the given points.
    #         """
    #
    #     resolution: int = 1024
    #     bounding_box_min, bounding_box_max = (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)
    #     isosurface_threshold = 10.0
    #
    #     # Check if resolution is divisible by 512
    #     assert (
    #             resolution % 512 == 0
    #     ), f"""resolution must be divisible by 512, got {resolution}.
    #            This is important because the algorithm uses a multi-resolution approach
    #            to evaluate the SDF where the mimimum resolution is 512."""
    #
    #     # Initialize variables
    #     crop_n = 128
    #     N = resolution // crop_n
    #     grid_min = bounding_box_min
    #     grid_max = bounding_box_max
    #     xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    #     ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    #     zs = np.linspace(grid_min[2], grid_max[2], N + 1)
    #
    #     # Initialize meshes list
    #     meshes = []
    #
    #     # Iterate over the grid
    #     for i in range(N):
    #         for j in range(N):
    #             for k in range(N):
    #                 # Calculate grid cell boundaries
    #                 x_min, x_max = xs[i], xs[i + 1]
    #                 y_min, y_max = ys[j], ys[j + 1]
    #                 z_min, z_max = zs[k], zs[k + 1]
    #
    #                 # Create point grid
    #                 x = np.linspace(x_min, x_max, crop_n)
    #                 y = np.linspace(y_min, y_max, crop_n)
    #                 z = np.linspace(z_min, z_max, crop_n)
    #                 xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    #                 points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).to(
    #                     device)
    #
    #                 # Construct point pyramids
    #                 points = points.reshape(crop_n, crop_n, crop_n, 3)
    #                 _, val = self.model.forward(points.reshape(-1, 3))
    #                 z = val.detach().cpu().numpy()
    #
    #                 if not (np.min(z) > isosurface_threshold or np.max(z) < isosurface_threshold):
    #                     z = z.astype(np.float32)
    #                     verts, faces, normals, _ = measure.marching_cubes(  # type: ignore
    #                         volume=z.reshape(crop_n, crop_n, crop_n),
    #                         level=isosurface_threshold,
    #                         spacing=(
    #                             (x_max - x_min) / (crop_n - 1),
    #                             (y_max - y_min) / (crop_n - 1),
    #                             (z_max - z_min) / (crop_n - 1),
    #                         ),
    #                     )
    #                     verts = verts + np.array([x_min, y_min, z_min])
    #
    #                     meshcrop = trimesh.Trimesh(verts, faces, normals)
    #                     meshes.append(meshcrop)
    #
    #     combined_mesh = trimesh.util.concatenate(meshes)
    #     # combined_mesh.export(os.path.join(save_path, f'mesh_ori_{resolution}.ply'))
    #     v, t = combined_mesh.vertices, combined_mesh.faces
    #     v = uncontract(torch.from_numpy(v)).numpy().astype(np.float32)
    #     combined_mesh_uncontracted = trimesh.Trimesh(v, t)
    #     combined_mesh_uncontracted.export(os.path.join(save_path, f'mesh_uncontracted_{resolution}.ply'))
    #
    #     return

    @torch.no_grad()
    def voxel_to_mesh(self, occ_grid, resolution=1024, S=128, loader=None):

        def export_mesh(s_path, vv, tt, q_bound):
            m = trimesh.Trimesh(vv, tt, process=False)
            s_path = os.path.join(s_path, f'mesh_{q_bound}.ply')
            m.export(s_path)
            self.log(f'[INFO] Saved mesh into {s_path}')
            self.log('[INFO] <============================>')

        self.log('[INFO] <============================>')
        self.log('[INFO] Extracting mesh from grid...')
        save_path = os.path.join(self.workspace, 'mesh')
        os.makedirs(save_path, exist_ok=True)
        device = self.device
        bound = self.model.bound  # always 2 in contracted mode
        real_bound = self.model.real_bound  # 128
        # sequentially load cascaded meshes
        vertices = []
        triangles = []
        v_cumsum = [0]
        f_cumsum = [0]

        # query
        X = torch.linspace(-1, 1, resolution).split(S)
        Y = torch.linspace(-1, 1, resolution).split(S)
        Z = torch.linspace(-1, 1, resolution).split(S)

        last_query_bound = 0
        for query_bound in self.model.cascade_list:
            sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = query_bound * torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1),
                                                       zz.reshape(-1, 1)], dim=-1).to(device)  # [S, 3]
                        with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                            _, val = self.model.forward(pts, bound=bound)
                            # val = self.model.forward(x=pts, shading='diffuse')['sigma']

                        sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys),
                        zi * S: zi * S + len(zs)] = val.reshape(len(xs), len(ys), len(zs))

            query_occ_grid = 1
            if query_bound < bound:
                half_resolution = int(resolution / 2)
                half_itv = int(query_bound / 2.0 * resolution / 2)
                query_occ_grid = occ_grid[half_resolution - half_itv:half_resolution + half_itv,
                                 half_resolution - half_itv:half_resolution + half_itv,
                                 half_resolution - half_itv:half_resolution + half_itv, ]
                query_occ_grid = F.interpolate(query_occ_grid.float().unsqueeze(0).unsqueeze(0), size=[resolution] * 3,
                                               mode='nearest').squeeze(0).squeeze(0) > 0

            sigmas = torch.nan_to_num(sigmas, 0)
            mean_density = torch.mean(sigmas.clamp(min=0)).item()
            sigmas = sigmas * query_occ_grid

            sigmas = sigmas.cpu().numpy()

            density_thresh = min(mean_density, self.model.density_thresh)
            self.log(f'[INFO] using density_thresh={density_thresh} in query_bound {query_bound}')
            v, t = mcubes.marching_cubes(sigmas, density_thresh)

            v = v / (resolution - 1.0) * 2 - 1  # range in [-1, 1]
            v = v * query_bound
            v = v.astype(np.float32)
            t = t.astype(np.int32)

            # half_grid_size = bound / resolution
            # if query_bound > 1.0:
            #     assert query_bound == bound
            #     v = v * (bound - half_grid_size)

            # decimation
            decimate_target = self.opt.decimate_target[len(vertices)]
            if 0 < decimate_target < t.shape[0]:
                v, t = decimate_mesh(v, t, decimate_target, remesh=False)

            # ## visibility test.
            # if loader is not None:
            #     dataset = loader._data
            #     visibility_mask = self.mark_unseen_triangles(v, t, dataset.mvps, dataset.H,
            #                                                  dataset.W).cpu().numpy()
            #     v, t = remove_masked_trigs(v, t, visibility_mask,
            #                                dilation=self.opt.visibility_mask_dilation)

            # remove the center (already covered by previous cascades)
            if last_query_bound:
                _r = last_query_bound - 0.05
                v, t = remove_selected_verts(v, t, f'(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && '
                                                   f'(y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})')

            if query_bound > 1.0:
                # warp back (uncontract)
                if self.opt.contract:
                    v = uncontract(torch.from_numpy(v)).numpy().astype(np.float32)

                v, t = remove_selected_verts(v, t, f'(x <= {-real_bound}) || '
                                                   f'(x >= {real_bound}) || '
                                                   f'(y <= {-real_bound}) || '
                                                   f'(y >= {real_bound}) || '
                                                   f'(z <= {-real_bound} ) || '
                                                   f'(z >= {real_bound})')
                # export_mesh(save_path, v, t, str(query_bound)+'_uncleaned')
                # remove the faces which have the length over self.opt.max_edge_len
                # v, t = remove_selected_vt_by_edge_length(v, t, self.opt.max_edge_len)

                # remove the isolated component composed by a limited number of triangles
                # v, t = remove_selected_isolated_faces(v, t, self.opt.min_iso_size)

            ## reduce floaters by post-processing...
            v, t = clean_mesh(v, t, min_f=self.opt.clean_min_f,
                              min_d=self.opt.clean_min_d,
                              repair=True, remesh=False)

            # decimate_target = self.opt.decimate_target[len(vertices)]
            # if 0 < decimate_target < t.shape[0]:
            #     v, t = decimate_mesh(v, t, decimate_target, remesh=False)

            # v, t = close_holes_meshfix(v, t)
            export_mesh(save_path, v, t, query_bound)

            last_query_bound = query_bound

            vertices.append(v)
            triangles.append(t + v_cumsum[-1])

            v_cumsum.append(v_cumsum[-1] + v.shape[0])
            f_cumsum.append(f_cumsum[-1] + t.shape[0])

        vertices = np.concatenate(vertices, axis=0)
        triangles = np.concatenate(triangles, axis=0)

        # visibility test.
        if loader is not None:
            dataset = loader._data
            visibility_mask = self.mark_unseen_triangles(vertices, triangles, dataset.mvps, dataset.H,
                                                         dataset.W).cpu().numpy()
            vertices, triangles = remove_masked_trigs(vertices, triangles, visibility_mask,
                                                      dilation=self.opt.visibility_mask_dilation)

        export_mesh(save_path, vertices, triangles, 'all')

        # # remove the faces which have the length over self.opt.max_edge_len
        # vertices, triangles = remove_selected_vt_by_edge_length(vertices, triangles, self.opt.max_edge_len)
        #
        # # remove the isolated component composed by a limited number of triangles
        # vertices, triangles = remove_selected_isolated_faces(vertices, triangles, self.opt.min_iso_size)

        # export_mesh(save_path, vertices, triangles, 'all_clean')

        # inner_idx = (np.array(self.opt.cascade_list) <= 1.0).sum()
        # if inner_idx > 1:
        #     self.log('[INFO] extract mesh which bound < self.opt.cascade_list[inner_idx-1]...')
        #     vertices, triangles = (remove_selected_verts(vertices, triangles,
        #                                                  f'(x <= {-self.opt.cascade_list[inner_idx - 1]}) || '
        #                                                  f'(x >= {self.opt.cascade_list[inner_idx - 1]}) || '
        #                                                  f'(y <= {-self.opt.cascade_list[inner_idx - 1]}) || '
        #                                                  f'(y >= {self.opt.cascade_list[inner_idx - 1]}) || '
        #                                                  f'(z <= {-self.opt.cascade_list[inner_idx - 1]} ) || '
        #                                                  f'(z >= {self.opt.cascade_list[inner_idx - 1]})'))
        #     export_mesh(save_path, vertices, triangles, f'{self.opt.cascade_list[inner_idx - 1]}_updated')

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        if self.opt.render == 'mesh' and self.opt.vis_error:
            assert self.epoch == max_epochs, 'vis_error only works for evaluation.'

        if (self.opt.render == 'mixed' and self.opt.use_mesh_occ_grid and
                (self.opt.vert_offset or self.model.mesh_occ_mask is None)):
            self.model.update_mesh_occ_mask(train_loader)

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            if (self.opt.render == 'mixed' and self.opt.use_mesh_occ_grid and
                    (self.opt.vert_offset or self.model.mesh_occ_mask is None)):
                self.model.update_mesh_occ_mask(train_loader)

            self.train_one_epoch(train_loader)

            if self.epoch % self.save_interval == 0 or self.epoch == max_epochs:
                self.save_checkpoint(full=True, best=False, is_last=self.epoch == max_epochs)

            if self.epoch % self.eval_interval == 0:
                if self.epoch != max_epochs:
                    self.evaluate_one_epoch(valid_loader)

                # self.save_checkpoint(full=True, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.6f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

            all_preds_feat = []
            all_preds_diffuse = []
            all_preds_spec = []
            all_preds_mesh = []
            all_preds_voxel = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                pred_dict = self.test_step(data)

                vis_list_in_pred_dict = ['rgb', 'depth',
                                         'specular', 'diffuse',
                                         'full_mesh', 'full_voxel', ]

                for vis_name in vis_list_in_pred_dict:
                    if vis_name not in pred_dict:
                        continue
                    # if pred_dict[vis_name].dtype != torch.bool:
                    #     print(f'{vis_name}: {pred_dict[vis_name].shape}, '
                    #           f'{pred_dict[vis_name].min()}, {pred_dict[vis_name].max()}')
                    if 'depth' in vis_name:
                        pred_item = pred_dict[vis_name].detach().cpu().numpy()
                        pred_item = (pred_item - pred_item.min()) / (
                                pred_item.max() - pred_item.min() + 1e-6)  # [N]
                    else:
                        if pred_dict[vis_name].dtype != torch.bool and (
                                pred_dict[vis_name].min() < 0 or pred_dict[vis_name].max() > 1.0):
                            print(f'\n{vis_name}: {pred_dict[vis_name].shape}, '
                                  f'{pred_dict[vis_name].min()}, {pred_dict[vis_name].max()} '
                                  f'in evaluate_one_epoch')
                        pred_item = pred_dict[vis_name].clamp(0, 1).detach().cpu().numpy()
                    pred_item = (pred_item * 255).astype(np.uint8)
                    pred_dict[vis_name] = pred_item

                if write_video:
                    all_preds.append(pred_dict['rgb'])
                    all_preds_depth.append(pred_dict['depth'])
                    # all_preds_feat.append(pred_dict['feat'])
                    all_preds_diffuse.append(pred_dict['diffuse'])
                    all_preds_spec.append(pred_dict['specular'])
                    all_preds_mesh.append(pred_dict['full_mesh'])
                    all_preds_voxel.append(pred_dict['full_voxel'])

                # if write_video:
                #     all_preds.append(pred)
                #     all_preds_depth.append(pred_depth)
                # else:
                #     cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                #                 cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                #     cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)  # [N, H, W, 3]
            all_preds_depth = np.stack(all_preds_depth, axis=0)  # [N, H, W, 3]
            all_preds_diffuse = np.stack(all_preds_diffuse, axis=0)  # [N, H, W, 3]
            all_preds_spec = np.stack(all_preds_spec, axis=0)  # [N, H, W, 3]
            all_preds_mesh = np.stack(all_preds_mesh, axis=0)  # [N, H, W, 3]
            all_preds_voxel = np.stack(all_preds_voxel, axis=0)  # [N, H, W, 3]

            # all_preds_depth = np.stack(all_preds_depth, axis=0)  # [N, H, W]

            # print('all_preds0: ', all_preds.shape)
            # fix ffmpeg not divisible by 2
            all_preds = np.pad(all_preds, (
                (0, 0), (0, 1 if all_preds.shape[1] % 2 != 0 else 0), (0, 1 if all_preds.shape[2] % 2 != 0 else 0),
                (0, 0)))
            all_preds_depth = np.pad(all_preds_depth, (
                (0, 0), (0, 1 if all_preds_depth.shape[1] % 2 != 0 else 0),
                (0, 1 if all_preds_depth.shape[2] % 2 != 0 else 0)))
            # all_preds_feat = np.pad(all_preds_feat, (
            #     (0, 0), (0, 1 if all_preds_feat.shape[1] % 2 != 0 else 0),
            #     (0, 1 if all_preds_feat.shape[2] % 2 != 0 else 0),
            #     (0, 0)))
            all_preds_diffuse = np.pad(all_preds_diffuse, (
                (0, 0), (0, 1 if all_preds_diffuse.shape[1] % 2 != 0 else 0),
                (0, 1 if all_preds_diffuse.shape[2] % 2 != 0 else 0),
                (0, 0)))
            all_preds_spec = np.pad(all_preds_spec, (
                (0, 0), (0, 1 if all_preds_spec.shape[1] % 2 != 0 else 0),
                (0, 1 if all_preds_spec.shape[2] % 2 != 0 else 0),
                (0, 0)))
            all_preds_mesh = np.pad(all_preds_mesh, (
                (0, 0), (0, 1 if all_preds_mesh.shape[1] % 2 != 0 else 0),
                (0, 1 if all_preds_mesh.shape[2] % 2 != 0 else 0),
                (0, 0)))
            all_preds_voxel = np.pad(all_preds_voxel, (
                (0, 0), (0, 1 if all_preds_voxel.shape[1] % 2 != 0 else 0),
                (0, 1 if all_preds_voxel.shape[2] % 2 != 0 else 0),
                (0, 0)))

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=30, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=30, quality=8,
                             macro_block_size=1)
            # imageio.mimwrite(os.path.join(save_path, f'{name}_feat.mp4'), all_preds_feat, fps=30, quality=8,
            #                  macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_diffuse.mp4'), all_preds_diffuse, fps=30, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_spec.mp4'), all_preds_spec, fps=30, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_mesh.mp4'), all_preds_mesh, fps=30, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_voxel.mp4'), all_preds_voxel, fps=30, quality=8,
                             macro_block_size=1)

        self.log(f"==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0 and self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            preds, truths, loss_net, loss_dict = self.train_step(data)

            loss = loss_net

            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss_net.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, shading='full'):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'mvp': mvp,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'index': [0],
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            # here spp is used as perturb random seed! (but not perturb the first sample)
            preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp,
                                                shading=shading)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = (F.interpolate(preds.unsqueeze(0).permute(0, 3, 1, 2), size=(H, W), mode='nearest')
                     .permute(0, 2, 3, 1).squeeze(0).contiguous())
            preds_depth = (F.interpolate(preds_depth.unsqueeze(0).unsqueeze(1), size=(H, W), mode='nearest')
                           .squeeze(0).squeeze(1))

        pred = preds.detach().cpu().numpy()
        pred_depth = preds_depth.detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            # update grid every 16 steps
            # if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
            #     self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            preds, truths, loss_net, loss_dict = self.train_step(data)

            loss = loss_net

            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss_net.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    for k in loss_dict.keys():
                        # print(f"{k}: {loss_dict[k]}")
                        self.writer.add_scalar("train/" + k, loss_dict[k], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss / self.local_step:.6f}), "
                        f"lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss / self.local_step:.6f})")
                pbar.update(loader.batch_size)

            if self.opt.render == 'mesh' and self.opt.refine and self.global_step in self.opt.refine_steps:
                self.log(f'\n[INFO] refine and decimate mesh at {self.global_step} step')
                self.model.refine_and_decimate()

                # reinit optim since params changed.
                self.optimizer = self.optimizer_fn(self.model)
                self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")

    def evaluate_one_epoch(self, loader, name=None, only_metric=False):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.opt.render}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                pred_dict, truths, loss = self.eval_step(data)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    metric_vals = []
                    for metric in self.metrics:
                        metric_val = metric.update(pred_dict['rgb'], truths)
                        metric_vals.append(metric_val)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    # save_path_depth = os.path.join(self.workspace, 'validation',
                    #                                f'{name}_{self.local_step:04d}_depth.png')
                    save_path_error = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_error_{metric_vals[0]:.2f}.png')
                    # metric_vals[0] should be the PSNR

                    # self.log(f"==> Saving validation image to {save_path}")
                    if only_metric is False:

                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        vis_list_in_pred_dict = ['rgb', 'depth',
                                                 'depth_mesh', 'mask_mesh',
                                                 'weights_sum', 'weights_mesh', 'weights_bg',
                                                 'specular', 'diffuse', 'diffuse_mesh', 'diffuse_voxel',
                                                 'diffuse_mesh_raw', 'full_mesh', 'full_voxel', 'full_mesh_raw',
                                                 'error_rgb_32', 'error_rgb_64', 'error_rgb_128', 'error_rgb_256',
                                                 'error_rgb_512',
                                                 'error_grey_32', 'error_grey_64', 'error_grey_128', 'error_grey_256',
                                                 'error_grey_512',
                                                 # f"depth_weights_{self.opt.alpha_thres:.3f}"
                                                 ]

                        # pred = pred_dict['rgb'].detach().cpu().numpy()
                        # pred = (pred * 255).astype(np.uint8)
                        # cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

                        for vis_name in vis_list_in_pred_dict:
                            if vis_name not in pred_dict:
                                continue
                            # if pred_dict[vis_name].dtype != torch.bool:
                            #     print(f'{vis_name}: {pred_dict[vis_name].shape}, '
                            #           f'{pred_dict[vis_name].min()}, {pred_dict[vis_name].max()}')
                            if 'depth' in vis_name:
                                pred_item = pred_dict[vis_name].detach().cpu().numpy()
                                pred_item = (pred_item - pred_item.min()) / (
                                        pred_item.max() - pred_item.min() + 1e-6)  # [N]
                            else:
                                if pred_dict[vis_name].dtype != torch.bool and (
                                        pred_dict[vis_name].min() < 0 or pred_dict[vis_name].max() > 1.0):
                                    print(f'\n{vis_name}: {pred_dict[vis_name].shape}, '
                                          f'{pred_dict[vis_name].min()}, {pred_dict[vis_name].max()} '
                                          f'in evaluate_one_epoch')
                                pred_item = pred_dict[vis_name].clamp(0, 1).detach().cpu().numpy()
                            pred_item = (pred_item * 255).astype(np.uint8)
                            if pred_item.shape[-1] == 3:
                                cv2.imwrite(save_path.replace('rgb', vis_name),
                                            cv2.cvtColor(pred_item, cv2.COLOR_RGB2BGR))
                            else:
                                cv2.imwrite(save_path.replace('rgb', vis_name), pred_item)

                        # pred_depth = pred_dict['depth'].detach().cpu().numpy()
                        # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                        # pred_depth = (pred_depth * 255).astype(np.uint8)
                        # cv2.imwrite(save_path_depth, pred_depth)

                        truth = truths.detach().cpu().numpy()
                        truth = (truth * 255).astype(np.uint8)
                        cv2.imwrite(save_path.replace('rgb', 'gt'), cv2.cvtColor(truth, cv2.COLOR_RGB2BGR))

                        pred = pred_dict['rgb'].detach().cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)
                        error = np.abs(truth.astype(np.float32) - pred.astype(np.float32)).mean(-1).astype(np.uint8)
                        # print(save_path_error, error.shape)
                        cv2.imwrite(save_path_error, error)

                    # if 'depth_mesh' in pred_dict:
                    #     pred_depth_mesh = pred_dict['depth_mesh'].detach().cpu().numpy()
                    #     pred_depth_mesh = (pred_depth_mesh - pred_depth_mesh.min()) / \
                    #                       (pred_depth_mesh.max() - pred_depth_mesh.min() + 1e-6)
                    #     pred_depth_mesh = (pred_depth_mesh * 255).astype(np.uint8)
                    #     cv2.imwrite(save_path_depth.replace('depth', 'depth_mesh'), pred_depth_mesh)
                    #
                    #     pred_mask_mesh = pred_dict['mask_mesh'].detach().cpu().numpy()
                    #     pred_mask_mesh = (pred_mask_mesh * 255).astype(np.uint8)
                    #     cv2.imwrite(save_path_depth.replace('depth', 'mask'), pred_mask_mesh)
                    #
                    #     pred_alpha_mesh = pred_dict['alphas_mesh'].detach().cpu().numpy()
                    #     pred_alpha_mesh = (pred_alpha_mesh * 255).astype(np.uint8)
                    #     cv2.imwrite(save_path_depth.replace('depth', 'alpha_mesh'), pred_alpha_mesh)

                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss / self.local_step:.6f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            measures = []
            for metric in self.metrics:
                measures.append(metric.measure())
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()
            self.measures.append(measures)

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True, is_last=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            self.log(f'[INFO] saving checkpoint to {os.path.join(self.ckpt_path, file_path)}...')
            if self.opt.render == 'mesh' and is_last and 'vertices_offsets' in state['model']:
                del state['model']['vertices_offsets']
            self.log(f'[INFO] delete vertex offsets from {file_path} in last step in mesh refine...')

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt,
                    # so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    # if 'density_grid' in state['model']:
                    #     del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
                    self.log(f'[INFO] saving checkpoint to {self.best_path}...')

            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):

        if checkpoint is None:  # load latest
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}*.pth'))
            if not checkpoint_list:
                if 'use_vol_pth' in self.opt and self.opt.use_vol_pth:
                    checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/vol*.pth'))
                else:
                    checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))

            if checkpoint_list:
                if len(checkpoint_list) > 1 and self.name not in checkpoint_list[-1].split('/')[-1]:
                    checkpoint = checkpoint_list[-2]
                else:
                    checkpoint = checkpoint_list[-1]
                if self.name in checkpoint.split('/')[-1]:
                    self.continue_vosh = True
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, abort loading latest model.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        # deal with unwanted prefix '_orig_mod.'
        state_dict = checkpoint_dict['model']
        # unwanted_prefix = '_orig_mod.'
        # for k, v in list(state_dict.items()):
        #     if not k.startswith(unwanted_prefix):
        #         state_dict[unwanted_prefix + k] = state_dict.pop(k)

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.log(f"[INFO] load at epoch {checkpoint_dict['epoch']}, global step {checkpoint_dict['global_step']}")
        if self.opt.render == 'grid' or self.continue_vosh:
            self.stats = checkpoint_dict['stats']
            self.epoch = checkpoint_dict['epoch']
            self.global_step = checkpoint_dict['global_step']
            self.log(f"[INFO] continue using epoch and global_step")

        # if self.opt.reset_lr:
        #     current_step = self.global_step
        #     self.log(f'[INFO] reset lr to 0.1**((iter-{current_step})/({self.opt.iters - current_step}))')
        #     self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self.optimizer, lambda iter: 0.1 ** ((iter - current_step) / (self.opt.iters - current_step)))

        if self.opt.render == 'grid' or self.continue_vosh:
            if self.optimizer and 'optimizer' in checkpoint_dict:
                try:
                    self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                    self.log("[INFO] loaded optimizer.")
                except:
                    self.log("[WARN] Failed to load optimizer.")

            if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
                try:
                    self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                    self.log("[INFO] loaded scheduler.")
                except:
                    self.log("[WARN] Failed to load scheduler.")
        else:
            self.log("[INFO] skipped optimizer and scheduler in vosh training.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

        if self.opt.render == 'mixed' and self.opt.mesh_encoder:
            if 'DensityAndFeaturesMLP_mesh.mlp.net.0.weight' in state_dict.keys():
                self.log(f"[INFO] loaded mesh encoder from {checkpoint}")
            else:
                checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/mesh*.pth'))
                assert checkpoint_list is not None, "No mesh checkpoint found."
                mesh_checkpoint = checkpoint_list[-1]
                mesh_checkpoint_dict = torch.load(mesh_checkpoint, map_location=self.device)['model']
                for key in list(mesh_checkpoint_dict.keys()):
                    v = mesh_checkpoint_dict.pop(key)
                    if key.startswith('DensityAndFeaturesMLP'):
                        mesh_checkpoint_dict[key.replace('DensityAndFeaturesMLP.', '')] = v

                self.model.DensityAndFeaturesMLP_mesh.load_state_dict(mesh_checkpoint_dict)
                self.log(f"[INFO] loaded mesh encoder from {mesh_checkpoint}")
