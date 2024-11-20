# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simulate quantization during training and for quantizing after training."""

import itertools
import torch
from nerf import math

MAX_BYTE = 2.0 ** 8 - 1.0


def quantize_float_to_byte(x):
    """Converts float32 to uint8."""
    # return np.minimum(MAX_BYTE, np.maximum(0.0, np.round(MAX_BYTE * x))).astype(np.uint8)
    return torch.clamp(torch.round(MAX_BYTE * x), 0, MAX_BYTE)


def dequantize_byte_to_float(x):
    """Converts uint8 to float32."""
    return x.to(torch.float32) / MAX_BYTE


def differentiable_byte_quantize(x):
    """Implements rounding with a straight-through-estimator."""
    zero = x - x.detach()
    return zero + (
            torch.round(torch.clip(x, 0.0, 1.0) * MAX_BYTE) / MAX_BYTE
    ).detach()


def simulate_quantization(x, v_min, v_max):
    """Simulates quant. during training: [-inf, inf] -> [v_min, v_max]."""

    def denormalize(_x, _v_min, _v_max):
        """[0, 1] -> [v_min, v_max]."""
        return _v_min + _x * (_v_max - _v_min)

    x = torch.sigmoid(x)  # Bounded to [0, 1].
    x = differentiable_byte_quantize(x)  # quantize and dequantize.
    return denormalize(x, v_min, v_max)  # Bounded to [v_min, v_max].


def dequantize_and_interpolate(x_grid, data, v_min, v_max):
    """Dequantizes and denormalizes and then linearly interpolates grid values."""
    x_floor = torch.floor(x_grid).to(torch.int32)
    x_ceil = torch.ceil(x_grid).to(torch.int32)
    local_coordinates = x_grid - x_floor
    res = torch.zeros(x_grid.shape[:-1] + (data.shape[-1],)).to(x_grid.device)
    corner_coords = [[False, True] for _ in range(local_coordinates.shape[-1])]
    for z in itertools.product(*corner_coords):
        w = torch.ones(local_coordinates.shape[:-1]).to(x_grid.device)
        l = []
        for i, b in enumerate(z):
            w = w * (
                local_coordinates[..., i] if b else (1 - local_coordinates[..., i])
            )
            l.append(x_ceil[..., i] if b else x_floor[..., i])
        gathered_data = data[tuple(l)]
        gathered_data = dequantize_byte_to_float(gathered_data)
        gathered_data = math.denormalize(gathered_data, v_min, v_max)
        res = res + w[..., None] * gathered_data.reshape(res.shape)
    return res
