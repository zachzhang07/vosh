import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from nerf import grid_utils
from nerf import quantize
from nerf import math as merf_math


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
        for l in range(self.num_layers):
            nn.init.kaiming_uniform_(self.net[l].weight)
        #     if self.net[l].bias is not None:
        #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.net[l].weight)
        #         bound = 1 / math.sqrt(fan_in)
        #         nn.init.uniform_(self.net[l].bias, -bound, bound)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class DeferredMLP(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.view_mlp = MLP(3 + 4 + 27, 3, 16, 3, bias=bias)

    def forward(self, x):
        x = torch.sigmoid(self.view_mlp(x))
        return x


class DensityAndFeaturesMLP(nn.Module):
    def __init__(self, hidden_dim=64, interpolation='linear', bias=False, data_format='colmap'):
        super().__init__()
        if(data_format != 'nerf'):
            self.encoder, self.in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=20,
                                                log2_hashmap_size=21, desired_resolution=8192 + 1,
                                                interpolation=interpolation)
        else:
            self.encoder, self.in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=16,
                                            log2_hashmap_size=19, desired_resolution=2048,
                                            interpolation=interpolation)


        self.mlp = MLP(self.in_dim, hidden_dim, hidden_dim, 1, bias=bias)
        self.density_mlp = MLP(hidden_dim, 1, hidden_dim, 1, bias=bias)
        self.feature_mlp = MLP(hidden_dim, 7, hidden_dim, 1, bias=bias)

    def forward(self, positions, bound):
        positions = (positions + bound) / (2 * bound)  # [0, 1]
        x = self.encoder(positions)
        x = torch.relu(self.mlp(x))
        density = self.density_mlp(x)
        features = self.feature_mlp(x)
        return features, density


class NeRFNetwork(NeRFRenderer):
    def __init__(self, opt):

        super().__init__(opt)
        self.grid_config = grid_utils.calculate_grid_config(self.opt)
        self.range_features = [-7.0, 7.0]  # Value range for appearance features.
        self.range_density = [-14.0, 14.0]  # Value range for density features.

        self.DensityAndFeaturesMLP = DensityAndFeaturesMLP(bias=False, data_format = opt.data_format)
        self.DeferredMLP = DeferredMLP(bias=True)

        # proposal network
        if not self.opt.cuda_ray:
            self.prop_encoders = nn.ModuleList()
            self.prop_mlp = nn.ModuleList()

            prop0_encoder, prop0_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=10,
                                                      log2_hashmap_size=16, desired_resolution=512)
            prop0_mlp = MLP(prop0_in_dim, 1, 64, 2, bias=True)
            self.prop_encoders.append(prop0_encoder)
            self.prop_mlp.append(prop0_mlp)

        if self.opt.render == 'mixed' and self.opt.mesh_encoder:
            self.DensityAndFeaturesMLP_mesh = DensityAndFeaturesMLP(bias=False, data_format = opt.data_format)

    def forward(self, x, **kwargs):
        # The structure of the representation is enforced by grid simulation. All grid
        # values are predicted by a single MLP (DensityAndFeaturesMLP).

        # Flatten positions (makes our life easier).
        batch_shape = x.shape[:-1]
        positions = x.reshape(-1, 3)  # in: UxKx3, out: U*Kx3.

        # Prepare grid simulation, afterward `positions` have the shape S*U*Kx3.
        # Triplane_positions_local, sparse_grid_positions_local: U*Kx3.
        _positions, triplane_positions_local, sparse_grid_positions_local = (
            grid_utils.get_eval_positions_and_local_coordinates(
                positions, self.opt, self.grid_config
            )
        )

        # Query MLP at grid corners (S*U*Kx7 and S*U*Kx1).
        features, density = self.DensityAndFeaturesMLP(_positions, kwargs['bound'])

        # Simulate quantization on MLP outputs.
        features = quantize.simulate_quantization(
            features, self.range_features[0], self.range_features[1]
        )

        # Grid simulation: bi-lineary and/or tri-linearly interpolate outputs.
        features = grid_utils.interpolate_based_on_local_coordinates(
            features, triplane_positions_local, sparse_grid_positions_local, self.opt
        )  # U*Kx7.

        # Apply activation functions after interpolation. Doing this after
        # interpolation increases the model's representational power.
        features = torch.sigmoid(features)

        density = quantize.simulate_quantization(
            density, self.range_density[0], self.range_density[1]
        )

        density = grid_utils.interpolate_based_on_local_coordinates(
            density, triplane_positions_local, sparse_grid_positions_local, self.opt
        )  # U*Kx1.

        density = merf_math.density_activation(density)

        if self.opt.render == 'mixed':
            # assert positions.max() <= self.bound and positions.min() >= -self.bound
            # positions: [N*K, 3], positions must be contracted (in [-2, 2])
            # put positions into [mcubes_reso^3] grid

            if self.opt.use_mesh_occ_grid:
                pos_coords = (torch.floor((positions + self.bound) / (2 * self.bound) * self.mesh_occ_mask.shape[0])
                              .long().clamp(0, self.mesh_occ_mask.shape[0] - 1))
                # pos_mask = (~self.mesh_occ_mask)[pos_coords[..., 0], pos_coords[..., 1], pos_coords[..., 2]]
                density = torch.where(self.mesh_occ_mask[tuple(pos_coords.T)][..., None],
                                      torch.zeros_like(density), density)

        # Unflatten results.
        def unflatten(x):
            return x.reshape(*batch_shape, -1)

        features = unflatten(features)  # UxKx7.
        density = unflatten(density)  # UxKx1.
        return features, density

    # @torch.no_grad()
    # def forward_mesh(self, x, **kwargs):
    #     # The structure of the representation is enforced by grid simulation. All grid
    #     # values are predicted by a single MLP (DensityAndFeaturesMLP).
    #
    #     # Flatten positions (makes our life easier).
    #     batch_shape = x.shape[:-1]
    #     positions = x.reshape(-1, 3)  # in: UxKx3, out: U*Kx3.
    #
    #     # Prepare grid simulation, afterward `positions` have the shape S*U*Kx3.
    #     # Triplane_positions_local, sparse_grid_positions_local: U*Kx3.
    #     _positions, triplane_positions_local, sparse_grid_positions_local = (
    #         grid_utils.get_eval_positions_and_local_coordinates(
    #             positions, self.opt, self.grid_config
    #         )
    #     )
    #
    #     # Query MLP at grid corners (S*U*Kx7 and S*U*Kx1).
    #     features, _ = self.DensityAndFeaturesMLP_mesh(_positions, kwargs['bound'])
    #
    #     # Simulate quantization on MLP outputs.
    #     features = quantize.simulate_quantization(
    #         features, self.range_features[0], self.range_features[1]
    #     )
    #
    #     # Grid simulation: bi-lineary and/or tri-linearly interpolate outputs.
    #     features = grid_utils.interpolate_based_on_local_coordinates(
    #         features, triplane_positions_local, sparse_grid_positions_local, self.opt
    #     )  # U*Kx7.
    #
    #     # Apply activation functions after interpolation. Doing this after
    #     # interpolation increases the model's representational power.
    #     features = torch.sigmoid(features)
    #
    #     # Unflatten results.
    #     def unflatten(x):
    #         return x.reshape(*batch_shape, -1)
    #
    #     features = unflatten(features)  # UxKx7.
    #     return features, _

    # def forward(self, x, d=None, **kwargs):
    #     return self.query_representation(positions=x, **kwargs)
    #

    # def query_representation(self, positions, **kwargs):
    #     # The structure of the representation is enforced by grid simulation. All grid
    #     # values are predicted by a single MLP (DensityAndFeaturesMLP).
    #
    #     # Flatten positions (makes our life easier).
    #     batch_shape = positions.shape[:-1]
    #     positions = positions.reshape(-1, 3)  # in: UxKx3, out: U*Kx3.
    #
    #     # Prepare grid simulation, afterward `positions` have the shape S*U*Kx3.
    #     # Triplane_positions_local, sparse_grid_positions_local: U*Kx3.
    #     _positions, triplane_positions_local, sparse_grid_positions_local = (
    #         grid_utils.get_eval_positions_and_local_coordinates(
    #             positions, self.opt, self.grid_config
    #         )
    #     )
    #
    #     # Query MLP at grid corners (S*U*Kx7 and S*U*Kx1).
    #     features, density = self.DensityAndFeaturesMLP(_positions, kwargs['bound'])
    #
    #     # Simulate quantization on MLP outputs.
    #     features = quantize.simulate_quantization(
    #         features, self.range_features[0], self.range_features[1]
    #     )
    #
    #     # Grid simulation: bi-lineary and/or tri-linearly interpolate outputs.
    #     features = grid_utils.interpolate_based_on_local_coordinates(
    #         features, triplane_positions_local, sparse_grid_positions_local, self.opt
    #     )  # U*Kx7.
    #
    #     # Apply activation functions after interpolation. Doing this after
    #     # interpolation increases the model's representational power.
    #     features = torch.sigmoid(features)
    #
    #     # if 'only_feat' in kwargs.keys() and kwargs['only_feat']:
    #     #     return features, None
    #
    #     density = quantize.simulate_quantization(
    #         density, self.range_density[0], self.range_density[1]
    #     )
    #
    #     density = grid_utils.interpolate_based_on_local_coordinates(
    #         density, triplane_positions_local, sparse_grid_positions_local, self.opt
    #     )  # U*Kx1.
    #
    #     density = merf_math.density_activation(density)
    #
    #     if self.opt.render == 'mixed':
    #         # assert positions.max() <= self.bound and positions.min() >= -self.bound
    #         # positions: [N*K, 3], positions must be contracted (in [-2, 2])
    #         # put positions into [mcubes_reso^3] grid
    #
    #         # if self.opt.density_scale > 0:
    #         #     pos_coords = (
    #         #         torch.floor((positions + self.bound) / (2 * self.bound) * self.density_scale.shape[0]).long()
    #         #         .clamp(0, self.density_scale.shape[0] - 1))
    #         #     scale = self.density_scale[pos_coords[..., 0], pos_coords[..., 1], pos_coords[..., 2]]
    #         #     density = density * scale[..., None]
    #
    #         if self.opt.use_mesh_occ_grid:
    #             pos_coords = (torch.floor((positions + self.bound) / (2 * self.bound) * self.mesh_occ_mask.shape[0])
    #                           .long().clamp(0, self.mesh_occ_mask.shape[0] - 1))
    #             # pos_mask = (~self.mesh_occ_mask)[pos_coords[..., 0], pos_coords[..., 1], pos_coords[..., 2]]
    #             density = torch.where(self.mesh_occ_mask[tuple(pos_coords.T)][..., None],
    #                                   torch.zeros_like(density), density)
    #
    #     # Unflatten results.
    #     def unflatten(x):
    #         return x.reshape(*batch_shape, -1)
    #
    #     features = unflatten(features)  # UxKx7.
    #     density = unflatten(density)  # UxKx1.
    #     return features, density

    # optimizer utils
    def get_params(self, lr=1.0):
        params = super().get_params()

        params.extend([{'params': self.DensityAndFeaturesMLP.parameters(), 'lr': lr}, ])
        params.extend([{'params': self.DeferredMLP.parameters(), 'lr': lr}, ])

        if not self.opt.cuda_ray:
            params.extend([
                {'params': self.prop_encoders.parameters(), 'lr': lr},
                {'params': self.prop_mlp.parameters(), 'lr': lr},
            ])

        # if self.opt.render == 'mixed' and self.opt.density_scale > 0:
        #     params.extend([{'params': self.density_scale, 'lr': lr, 'weight_decay': 0}, ])

        if self.opt.render == 'mixed' and self.opt.mesh_encoder:
            params.extend([{'params': self.DensityAndFeaturesMLP_mesh.parameters(), 'lr': lr}, ])

        return params

    def apply_total_variation(self, lambda_tv):
        if self.opt.grid_resolution:
            self.grid.grad_total_variation(lambda_tv)
        self.planeXY.grad_total_variation(lambda_tv * 0.1)
        self.planeXZ.grad_total_variation(lambda_tv * 0.1)
        self.planeYZ.grad_total_variation(lambda_tv * 0.1)
