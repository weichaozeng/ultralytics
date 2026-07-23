import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from .ops.array_ops import torch_quantile
from .ops.image import nearest_neighbor_inpaint


class PerPixelBayesian(nn.Module):
    """
    Per-pixel adaptive-exposure smoothing using Bayesian runlength estimation.
    Pure PyTorch implementation, optimized with torch.compile.
    """

    def __init__(
        self,
        bocpd_gamma: float = 1e-3,
        memory_size: int = 10,
        subsampling: int = 1,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        normalize: bool = False,
        quantile: float = 1.0,
        min_filter_size: int = 5,
    ):
        """
        Initialize a PerPixelBayesian smoothing module.

        :param bocpd_gamma: BOCPD hazard rate
        :param memory_size: Number of runlength forecasters per pixel
        :param subsampling: Output subsampling factor
        :param s: Mask for hot pixels
        :param normalize: If True, normalize output
        :param quantile: Normalization quantile
        :param min_filter_size: Size for min filter of runlength
        """
        super().__init__()
        assert min_filter_size % 2 == 1, "Min. filter size must be odd"

        self.bocpd_gamma = bocpd_gamma
        self.memory_size = memory_size
        self.subsampling = max(int(subsampling), 1)
        self.hot_pixel_mask = hot_pixel_mask
        self.normalize = normalize
        self.quantile = quantile
        self.min_filter_size = min_filter_size

        self.t_absolute = 0
        self._h, self._w, self._t = (None, None, None)

        self.register_buffer("recons_tensor", None)
        self.register_buffer("alpha_ll", None)
        self.register_buffer("beta_ll", None)
        self.register_buffer("forecaster_distribution", None)
        self.register_buffer("forecaster_distribution_index", None)
        self.register_buffer("total_occupied", None)
        self.register_buffer("ema", None)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(bocpd_gamma={self.bocpd_gamma}, memory_size={self.memory_size}, "
            f"min_filter_size={self.min_filter_size})"
        )

    def set_cube(self, photon_cube: Bool[Tensor, "h w t"]):
        """Initializes or resizes state tensors based on input cube dimensions."""
        _h, _w, _t = photon_cube.shape
        device = photon_cube.device
        dtype = torch.float32

        if (_h, _w, _t) != (self._h, self._w, self._t):
            self._t = _t
            if self._t <= 0:
                out_t = 0
            elif self._t < self.subsampling:
                # Short tail cubes still emit one reconstruction at the final EMA state.
                out_t = 1
            else:
                out_t = self._t // self.subsampling
            self.recons_tensor = torch.zeros((_h, _w, out_t), device=device, dtype=dtype)

        if (_h, _w) != (self._h, self._w):
            self._h, self._w = _h, _w
            self.alpha_ll = torch.zeros(
                (_h, _w, self.memory_size), device=device, dtype=dtype
            )
            self.beta_ll = torch.zeros(
                (_h, _w, self.memory_size), device=device, dtype=dtype
            )
            self.forecaster_distribution = torch.zeros(
                (_h, _w, self.memory_size), device=device, dtype=dtype
            )
            self.forecaster_distribution_index = torch.zeros(
                (_h, _w, self.memory_size), device=device, dtype=torch.long
            )
            self.total_occupied = torch.ones((_h, _w), device=device, dtype=torch.long)
            self.ema = torch.zeros((_h, _w), device=device, dtype=dtype)
            self.sample_weight = torch.zeros((_h, _w), device=device, dtype=dtype)

        if not self.t_absolute:
            self.init_bocpd_arrays()

    def init_bocpd_arrays(self):
        """Resets all state tensors to their initial values."""
        if self.recons_tensor is not None:
            self.recons_tensor.zero_()
            self.alpha_ll.zero_()
            self.beta_ll.zero_()
            self.forecaster_distribution.zero_()
            self.forecaster_distribution_index.zero_()
            self.alpha_ll[..., 0] = 1.0
            self.beta_ll[..., 0] = 1.0
            self.forecaster_distribution[..., 0] = 1.0
            self.total_occupied.fill_(1)
            self.ema.zero_()

    def min_pool2d(
        self, x: Float[Tensor, "h w"], kernel_size: int
    ) -> Float[Tensor, "h w"]:
        """Performs 2D min-pooling on an image using max-pooling on the negative."""
        x_batched = x.unsqueeze(0).unsqueeze(0)
        padding = (kernel_size - 1) // 2
        pooled = -F.max_pool2d(
            -x_batched, kernel_size=kernel_size, stride=1, padding=padding
        )
        return pooled.squeeze(0).squeeze(0)

    @torch.compile
    def _get_reconstruction_torch(self, photon_cube: Tensor, t_absolute: int):
        """
        Iterates through time, performing vectorized updates for each time step.
        This method is JIT-compiled for performance.
        """
        for t_index in range(self._t):
            incoming_sample = photon_cube[..., t_index].float()

            run_lengths = (
                ((t_index + t_absolute + 1) - self.forecaster_distribution_index)
                .float()
                .clamp(min=0)
            )

            estimated_run_length = torch.expm1(
                torch.sum(
                    self.forecaster_distribution * torch.log1p(run_lengths), dim=-1
                )
            )

            if self.min_filter_size > 1:
                estimated_run_length = self.min_pool2d(
                    estimated_run_length, self.min_filter_size
                )

            self.sample_weight = 1 - torch.exp(-1 / estimated_run_length)

            self.ema = (
                self.ema * (1 - self.sample_weight)
                + self.sample_weight * incoming_sample
            )
            self._update_forecaster_and_laplace_torch(
                incoming_sample, t_index + t_absolute
            )

            if (t_index + 1) % self.subsampling == 0:
                self.recons_tensor[..., t_index // self.subsampling] = self.ema

        if self._t > 0 and self._t < self.subsampling:
            self.recons_tensor[..., 0] = self.ema

    def _update_forecaster_and_laplace_torch(self, reward: Tensor, t_index: int):
        """Performs a vectorized update of the BOCPD state for all pixels."""
        reward = reward.unsqueeze(-1)
        divisor = (self.alpha_ll + self.beta_ll).clamp(min=1e-9)
        likelihood = torch.where(
            reward == 1, self.alpha_ll / divisor, self.beta_ll / divisor
        )
        new_forecaster_prob = self.bocpd_gamma * torch.sum(
            likelihood * self.forecaster_distribution, dim=-1
        )
        self.forecaster_distribution *= (1 - self.bocpd_gamma) * likelihood
        indices_to_drop = torch.argmin(self.forecaster_distribution, dim=-1)
        mask_has_capacity = self.total_occupied < self.memory_size
        min_forecaster_prob = self.forecaster_distribution.gather(
            dim=-1, index=indices_to_drop.unsqueeze(-1)
        ).squeeze(-1)
        mask_should_replace = (~mask_has_capacity) & (
            new_forecaster_prob > min_forecaster_prob
        )
        mask_should_insert = mask_has_capacity | mask_should_replace
        insert_indices = torch.where(
            mask_has_capacity, self.total_occupied, indices_to_drop
        )

        if mask_should_insert.any():
            insert_indices_expanded = insert_indices.unsqueeze(-1)
            self.forecaster_distribution.scatter_(
                -1, insert_indices_expanded, new_forecaster_prob.unsqueeze(-1)
            )
            self.forecaster_distribution_index.scatter_(
                -1, insert_indices_expanded, t_index
            )
            self.alpha_ll.scatter_(-1, insert_indices_expanded, 1.0)
            self.beta_ll.scatter_(-1, insert_indices_expanded, 1.0)

        self.forecaster_distribution /= self.forecaster_distribution.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-9)
        self.alpha_ll += reward
        self.beta_ll += 1 - reward
        self.total_occupied += (mask_should_insert & mask_has_capacity).long()

    def estimated_run_length_hw(self) -> Tensor | None:
        """Return last-bin estimated run length ``(H, W)`` after ``process_photon_cube``.

        Rebuilds from BOCPD state so ``@torch.compile`` on the update loop need not
        write auxiliary attributes.
        """
        if not hasattr(self, "forecaster_distribution") or self.forecaster_distribution is None:
            return None
        if int(self.t_absolute) <= 0:
            return None
        # After process_photon_cube, t_absolute is past the last processed bin.
        run_lengths = (float(self.t_absolute) - self.forecaster_distribution_index.float()).clamp(min=0)
        estimated = torch.expm1(
            torch.sum(self.forecaster_distribution * torch.log1p(run_lengths), dim=-1)
        )
        if self.min_filter_size > 1:
            estimated = self.min_pool2d(estimated, self.min_filter_size)
        return estimated

    def clamp_recons(self, recons: Tensor) -> Tensor:
        """Clamps and optionally normalizes the reconstruction."""
        if recons.numel() == 0:
            return recons
        max_value = 1.0
        if self.normalize:
            max_value = torch_quantile(recons, self.quantile).clamp(min=1e-6)
        return (recons / max_value).clamp(0, 1)

    @torch.no_grad()
    def process_photon_cube(
        self,
        photon_cube: Bool[Tensor, "h w t"],
        bocpd_gamma: float = None,
        memory_size: int = None,
        subsampling: int = None,
        hot_pixel_mask: Bool[np.ndarray, "h w"] = None,
        quantile: float = None,
        normalize: bool = None,
        clear_states: bool = True,
        min_filter_size: int = None,
        **kwargs,
    ) -> Float[Tensor, "h w t"]:
        with torch.no_grad():
            if clear_states:
                self.t_absolute = 0

            self.update_hyperparams(
                bocpd_gamma=bocpd_gamma,
                memory_size=memory_size,
                subsampling=subsampling,
                hot_pixel_mask=hot_pixel_mask,
                normalize=normalize,
                quantile=quantile,
                min_filter_size=min_filter_size,
            )

            self.set_cube(photon_cube)
            self._get_reconstruction_torch(photon_cube, self.t_absolute)
            recons_ll = self.recons_tensor

            if self.hot_pixel_mask is not None:
                recons_ll = nearest_neighbor_inpaint(recons_ll, self.hot_pixel_mask)

            recons_ll = self.clamp_recons(recons_ll)
            self.t_absolute += self._t

            return recons_ll

    def update_hyperparams(self, **kwargs):
        """Dynamically update class attributes if new values are provided."""
        for name, value in kwargs.items():
            if hasattr(self, name) and value is not None:
                setattr(self, name, value)
