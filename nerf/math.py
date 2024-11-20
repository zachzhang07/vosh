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

"""Mathy utility functions."""
import torch
import numpy as np


def matmul(a, b):
    """torch.matmul defaults to bfloat16, but this helper function doesn't."""
    return torch.matmul(a, b)


# def safe_trig_helper(x, fn, t=100 * torch.pi):
#     """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
#     return fn(torch.nan_to_num(torch.where(torch.abs(x) < t, x, x % t)))
# 
# 
# def safe_cos(x):
#     """torch.cos() on a TPU may NaN out for large values."""
#     return safe_trig_helper(x, torch.cos)
# 
# 
# def safe_sin(x):
#     """torch.sin() on a TPU may NaN out for large values."""
#     return safe_trig_helper(x, torch.sin)


def safe_exp(x):
    """torch.exp() but with finite output and gradients for large inputs."""
    return torch.exp(torch.clamp(x, max=88.0))  # np.exp(89) is infinity.


def safe_exp_jvp(primals, tangents):
    """Override safe_exp()'s gradient so that it's large when inputs are large."""
    [x] = primals
    [x_dot] = tangents
    exp_x = safe_exp(x)
    exp_x_dot = exp_x * x_dot
    return exp_x, exp_x_dot


def safe_log(x):
    """torch.log() but with finite outputs/gradients for negative/huge inputs."""
    return torch.log(torch.clip(x, 1e-37, 1e37))  # torch.log(1e-38) is -infinity.


def safe_log_jvp(primals, tangents):
    """Override safe_log()'s gradient to always be finite."""
    [x] = primals
    [x_dot] = tangents
    log_x = safe_log(x)
    log_x_dot = x_dot / torch.maximum(1e-37, x)
    return log_x, log_x_dot


def log_lerp(t, v0, v1):
    """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
    if v0 <= 0 or v1 <= 0:
        raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
    lv0 = np.log(v0)
    lv1 = np.log(v1)
    return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(
        step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1
):
    """Continuous learning rate decay function.

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.

    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)


def density_to_alpha(x, step_size):
    return 1.0 - safe_exp(-x * step_size)


def density_activation(x):
    return safe_exp(x - 1.0)


def normalize(x, v_min, v_max):
    """[v_min, v_max] -> [0, 1]."""
    return (x - v_min) / (v_max - v_min)


def denormalize(x, v_min, v_max):
    """[0, 1] -> [v_min, v_max]."""
    return v_min + x * (v_max - v_min)


def sorted_lookup(x, xp, fps, device_is_tpu=False):
    """Lookup `x` into locations `xp` , return indices and each `[fp]` value."""
    if not isinstance(fps, tuple):
        raise ValueError(f'Input `fps` must be a tuple, but is {type(fps)}.')

    if device_is_tpu:
        # Identify the location in `xp` that corresponds to each `x`.
        # The final `True` index in `mask` is the start of the matching interval.
        mask = x[Ellipsis, None, :] >= xp[Ellipsis, :, None]

        def find_interval(x):
            # Grab the value where `mask` switches from True to False, and vice versa.
            # This approach takes advantage of the fact that `x` is sorted.
            x0 = np.max(np.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), -2)
            x1 = np.min(np.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), -2)
            return x0, x1

        idx0, idx1 = find_interval(np.arange(xp.shape[-1]))
        vals = [find_interval(fp) for fp in fps]
    else:
        # np.searchsorted() has slightly different conventions for boundary
        # handling than the rest of this codebase.

        idx = np.searchsorted(xp, x, side='right')
        idx1 = np.minimum(idx, xp.shape[-1] - 1)
        idx0 = np.maximum(idx - 1, 0)
        vals = []
        for fp in fps:
            fp0 = np.take_along_axis(fp, idx0, axis=-1)
            fp1 = np.take_along_axis(fp, idx1, axis=-1)
            vals.append((fp0, fp1))
    return (idx0, idx1), vals


def sorted_interp(
        x, xp, fp, device_is_tpu, eps=np.finfo(np.float32).eps ** 2
):
    """A version of interp() where xp and fp must be sorted."""
    (xp0, xp1), (fp0, fp1) = sorted_lookup(
        x, xp, (xp, fp), device_is_tpu=device_is_tpu
    )[1]
    offset = np.clip((x - xp0) / np.maximum(eps, xp1 - xp0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret
