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

"""Tools for manipulating coordinate spaces and distances along rays."""
import torch


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg).to(x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = torch.reshape((x[..., None, :] * scales[:, None]), shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1)
    )
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def piecewise_warp_fwd(x, eps=torch.finfo(torch.float32).eps):
    """A piecewise combo of linear and reciprocal to allow t_near=0."""
    return torch.where(x < 1, 0.5 * x, 1 - 0.5 / torch.maximum(eps, x))


def piecewise_warp_inv(x, eps=torch.finfo(torch.float32).eps):
    """The inverse of `piecewise_warp_fwd`."""
    return torch.where(x < 0.5, 2 * x, 0.5 / torch.maximum(eps, 1 - x))


def s_to_t(s, t_near, t_far):
    """Convert normalized distances ([0,1]) to world distances ([t_near, t_far])."""
    s_near, s_far = [piecewise_warp_fwd(x) for x in (t_near, t_far)]
    return piecewise_warp_inv(s * s_far + (1 - s) * s_near)


@torch.cuda.amp.autocast(enabled=False)
def contract(x):
    """The contraction function we proposed in MERF."""
    # For more info check out MERF: Memory-Efficient Radiance Fields for Real-time
    # View Synthesis in Unbounded Scenes: https://arxiv.org/abs/2302.12249,
    # Section 4.2
    # After contraction points lie within [-2,2]^3.
    x_abs = torch.abs(x)
    # Clamping to 1 produces correct scale inside |x| < 1.
    x_max = torch.max(torch.amax(x_abs, dim=-1, keepdim=True), torch.tensor(1.0))
    scale = 1 / x_max  # no divide by 0 because of previous maximum(1, ...)
    z = scale * x
    # The above produces coordinates like (x/z, y/z, 1)
    # but we still need to replace the "1" with \pm (2-1/z).
    idx = torch.argmax(x_abs, dim=-1, keepdim=True)
    negative = torch.take_along_dim(z, idx, axis=-1) < 0
    o = torch.where(negative, -2 + scale, 2 - scale)
    # Select the final values by coordinate.
    ival_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
    ival = torch.arange(x.shape[-1]).to(x.device).reshape(ival_shape)
    result = torch.where(x_max <= 1, x, torch.where(ival == idx, o, z))
    return result


def contract_jax(x):
    """The contraction function we proposed in MERF."""
    # For more info check out MERF: Memory-Efficient Radiance Fields for Real-time
    # View Synthesis in Unbounded Scenes: https://arxiv.org/abs/2302.12249,
    # Section 4.2
    # After contraction points lie within [-2,2]^3.
    x_abs = jnp.abs(x)
    # Clamping to 1 produces correct scale inside |x| < 1.
    x_max = jnp.maximum(1, jnp.amax(x_abs, axis=-1, keepdims=True))
    scale = 1 / x_max  # no divide by 0 because of previous maximum(1, ...)
    z = scale * x
    # The above produces coordinates like (x/z, y/z, 1)
    # but we still need to replace the "1" with \pm (2-1/z).
    idx = jnp.argmax(x_abs, axis=-1, keepdims=True)
    negative = jnp.take_along_axis(z, idx, axis=-1) < 0
    o = jnp.where(negative, -2 + scale, 2 - scale)
    # Select the final values by coordinate.
    ival_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
    ival = jnp.arange(x.shape[-1]).reshape(ival_shape)
    result = jnp.where(x_max <= 1, x, jnp.where(ival == idx, o, z))
    return result


def stepsize_in_squash(x, d, v, contractFlag):
    """Computes step size in contracted space."""
    # Approximately computes s such that ||c(x+d*s) - c(x)||_2 = v, where c is
    # the contraction function, i.e., we often need to know by how much (s) the ray
    # needs to be advanced to get an advancement of v in contracted space.
    #
    # The further we are from the scene's center, the larger steps in world space
    # we have to take to get the same advancement in contracted space.
    x.requires_grad_()
    if contractFlag:
        contract_0_grad = torch.autograd.grad(contract(x)[..., 0].sum(), x)[0]
        contract_1_grad = torch.autograd.grad(contract(x)[..., 1].sum(), x)[0]
        contract_2_grad = torch.autograd.grad(contract(x)[..., 2].sum(), x)[0]
    else:
        contract_0_grad = torch.autograd.grad(x[..., 0].sum(), x)[0]
        contract_1_grad = torch.autograd.grad(x[..., 1].sum(), x)[0]
        contract_2_grad = torch.autograd.grad(x[..., 2].sum(), x)[0]

    def helper(_x, _d):
        # _d: [N, 3] → [N, 1, 3]
        _d = _d[:, None, :]
        # contract_grad: [N, 3] → [N, 3, 1]

        return torch.sqrt(
            torch.bmm(_d, contract_0_grad[..., None]) ** 2
            + torch.bmm(_d, contract_1_grad[..., None]) ** 2
            + torch.bmm(_d, contract_2_grad[..., None]) ** 2
        ).reshape(-1)

    return v / helper(x, d)
