"""
Semi-Separable Dynamics (SSD) model and helpers.
Implements fast semi-separable state-space models for temporal sequence modeling.
"""
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from quanta_neural_networks.ops.array_ops import loguniform


def segsum(x):
    """
    Compute cumulative sum over the last dimension with stability.

    :param x: Input tensor
    :return: Cumulative segment sum tensor
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(
    u_ll: Float[Tensor, "batch length num_heads head_dim"],
    log_scalarA_bar: Float[Tensor, "batch length num_heads"],
    vecB_bar: Float[Tensor, "batch length num_heads state_dim"],
    vecC: Float[Tensor, "batch length num_heads state_dim"],
    chunk_size: int = 64,
) -> tuple[
    Float[Tensor, "batch length num_heads head_dim"],
    Float[Tensor, "batch num_heads state_dim"],
]:
    """
    Compute the minimal SSD blockwise discrete-time update.

    :param u_ll: Input sequence tensor (batch, length, num_heads, head_dim)
    :param log_scalarA_bar: Discretized log decay tensor
    :param vecB_bar: Discretized B tensor
    :param vecC: C tensor
    :param chunk_size: Block size for computation
    :return: (Y, final_state) --- tuple of output sequence and final state
    """
    assert u_ll.dtype == log_scalarA_bar.dtype == vecB_bar.dtype == vecC.dtype

    seq_len = u_ll.shape[1]
    padding_size = ceil(seq_len / chunk_size) * chunk_size - seq_len

    if padding_size > 0:
        padding_tuple = (0, 0, 0, 0, 0, padding_size)
        u_ll = F.pad(u_ll, padding_tuple)
        log_scalarA_bar = F.pad(log_scalarA_bar, padding_tuple[2:])
        vecB_bar = F.pad(vecB_bar, padding_tuple)
        vecC = F.pad(vecC, padding_tuple)

    # Rearrange into blocks/chunks
    u_ll, log_scalarA_bar, vecB_bar, vecC = [
        rearrange(x, "b (c l) ... -> b c l ...", l=chunk_size)
        for x in (u_ll, log_scalarA_bar, vecB_bar, vecC)
    ]

    log_scalarA_bar = rearrange(log_scalarA_bar, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(log_scalarA_bar, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(log_scalarA_bar))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", vecC, vecB_bar, L, u_ll)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", vecB_bar, decay_states, u_ll)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", vecC, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    # Remove sequence-length padding
    if padding_size > 0:
        Y = Y[:, :-padding_size]

    return Y, final_state


class SSD(nn.Module):
    """
    PyTorch implementation of Semi-separable Dynamics model (SSD).
    Supports online, sequential, and parallel scan operation for temporal data.

    :param in_dim: Input embedding dimension
    :param state_dim: Latent state dimension
    :param head_dim: Per-head dimensionality
    :param subsampling: Subsample temporal axis
    :param chunk_size: Processing chunk/block size
    :param identity_init: Use identity init for better convergence
    :param parallel_mode: Evaluate using parallel scan/recursion
    :param a_init_range: (min, max) for initial A
    :param delta_min: Minimum delta
    :param delta_max: Maximum delta
    """
    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        head_dim: int = 128,
        subsampling: int = 1,
        chunk_size: int = 32,
        identity_init: bool = True,
        parallel_mode: bool = False,
        a_init_range=(1, 16),
        delta_min: float = 5e-5,
        delta_max: float = 5e-3,
    ):
        super().__init__()

        assert in_dim % head_dim == 0
        num_heads = in_dim // head_dim
        self.in_dim = in_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.subsampling = subsampling

        # Refers to semi-separable matrix casting, but we'll call it parallel_scan for API uniformity
        self.parallel_mode = parallel_mode

        # Don't need gating and accumulation here
        # scalarA is negative
        self.abs_scalarA = torch.nn.Parameter(torch.zeros(num_heads))
        self.input_to_dt_proj = nn.Linear(in_dim, num_heads)
        self.input_to_vecB = nn.Linear(in_dim, num_heads * state_dim)
        self.input_to_vecC = nn.Linear(in_dim, num_heads * state_dim)

        if identity_init:
            nn.init.constant_(self.abs_scalarA, a_init_range[1])

            nn.init.constant_(self.input_to_dt_proj.weight, 0)
            nn.init.constant_(self.input_to_dt_proj.bias, 1)

            nn.init.constant_(self.input_to_vecB.weight, 0)
            nn.init.constant_(self.input_to_vecB.bias, a_init_range[1])

            nn.init.constant_(self.input_to_vecC.weight, 0)
            nn.init.constant_(self.input_to_vecC.bias, 1 / state_dim)

        else:
            nn.init.uniform_(self.abs_scalarA, *a_init_range)

            window = torch.from_numpy(
                loguniform(delta_min, delta_max, size=num_heads)
            ).float()

            self.input_to_dt_proj.bias.data = window

        self.hidden_state = 0.0
        self.t_prev = 0.0

    def clear_hidden_state(self):
        """Reset state for online/streaming operation."""
        self.hidden_state = 0.0
        self.t_prev = 0.0

    @staticmethod
    def zoh_discretize(
        scalarA: Float[Tensor, "..."],
        vecB: Float[Tensor, "... state_dim"],
        delta: Float[Tensor, "..."],
    ) -> tuple[Float[Tensor, "..."], Float[Tensor, "... state_dim"]]:
        """
        Discretize continuous model via zero-order hold (ZOH).

        :param scalarA: Continuous decay rates (negative values)
        :param vecB: Input-to-state vector
        :param delta: Time increment or step vector
        :return: (discrete_scalarA, discrete_vecB)
        """
        log_scalarA_bar = scalarA * delta
        vecB_bar = ((log_scalarA_bar.exp() - 1) / scalarA).unsqueeze(-1) * vecB
        return log_scalarA_bar, vecB_bar

    def project(
        self, in_vector: Float[Tensor, "... batch in_dim"]
    ) -> tuple[
        Float[Tensor, "... batch num_heads"],
        Float[Tensor, "... batch num_heads"],
        Float[Tensor, "... batch num_heads state_dim"],
        Float[Tensor, "... batch num_heads state_dim"],
    ]:
        """
        Projection from input to model parameters.

        :param in_vector: Input tensor with embedding dim last
        :return: Tuple of (scalarA, dt, vecB, vecC)
        """
        scalarA = -F.relu(self.abs_scalarA)

        dt = F.relu(self.input_to_dt_proj(in_vector))

        vecB = self.input_to_vecB(in_vector)
        vecB = rearrange(
            vecB,
            "... (num_heads state_dim) -> ... num_heads state_dim",
            num_heads=self.num_heads,
        )

        vecC = self.input_to_vecC(in_vector)
        vecC = rearrange(
            vecC,
            "... (num_heads state_dim) -> ... num_heads state_dim",
            num_heads=self.num_heads,
        )

        return scalarA, dt, vecB, vecC

    @staticmethod
    def spatial_to_embedding(
        tensor: Float[Tensor, "... in_channels height width"]
    ) -> tuple[Float[Tensor, "... batch in_channels"], tuple[int, int]]:
        """
        Flatten spatial tensor to sequence embedding for model.

        :param tensor: Spatial tensor [batch ... in_channels, height, width]
        :return: Tuple (flattened tensor, (height, width))
        """
        height, width = tensor.shape[-2:]
        tensor = rearrange(
            tensor, "... in_channels height width -> ... (height width) in_channels"
        )
        return tensor, (height, width)

    @staticmethod
    def embedding_to_spatial(
        tensor: Float[Tensor, "... batch in_channels"], height: int, width: int
    ) -> Float[Tensor, "... in_channels height width"]:
        """
        Recover spatial tensor from embedding representation.

        :param tensor: Flattened embedding tensor
        :param height: Height
        :param width: Width
        :return: Restored spatial tensor
        """
        tensor = rearrange(
            tensor,
            "... (height width) in_channels -> ... in_channels height width",
            height=height,
            width=width,
        )
        return tensor

    def forward_online(
        self,
        in_vector: Float[Tensor, "batch in_dim"],
        time_instant: float,
    ) -> Float[Tensor, "batch in_dim"]:
        """
        Online state-space model step for a single time instant.

        :param in_vector: Input vector/time step tensor
        :param time_instant: Corresponding time or index
        :return: Model output at this timestep
        """
        is_spatial_input = False
        if in_vector.ndim == 3:
            is_spatial_input = True
            in_vector, (height, width) = self.spatial_to_embedding(in_vector)

        assert in_vector.shape[-1] == self.in_dim

        # Project and obtain scalarA, vecB, vecC
        scalarA, dt, vecB, vecC = self.project(in_vector)

        # reshape in_vector_ll to extract heads
        in_vector = rearrange(
            in_vector,
            "batch (num_heads head_dim) -> batch num_heads 1 head_dim",
            head_dim=self.head_dim,
        )

        log_scalarA_bar, vecB_bar = self.zoh_discretize(
            scalarA,
            vecB,
            dt * (time_instant - self.t_prev),
        )
        scalarA_bar = log_scalarA_bar.exp().unsqueeze(-1).unsqueeze(-1)
        vecB_bar = vecB_bar.unsqueeze(-1)
        vecC = vecC.unsqueeze(-1)

        self.hidden_state: Float[Tensor, "batch num_heads state_dim head_dim"] = (
            scalarA_bar * self.hidden_state + vecB_bar * in_vector
        )

        output = (vecC * self.hidden_state).sum(dim=-2)

        output = rearrange(
            output, "batch num_heads head_dim -> batch (num_heads head_dim)"
        )

        if is_spatial_input:
            output = self.embedding_to_spatial(output, height, width)

        self.t_prev = time_instant

        return output

    def forward_sequential(
        self,
        in_vector_ll: Float[Tensor, "seq_len batch num_heads 1 head_dim"],
        log_scalarA_bar_ll: Float[Tensor, "seq_len batch num_heads 1 1"],
        vecB_bar_ll: Float[Tensor, "seq_len batch num_heads state_dim 1"],
        vecC_ll: Float[Tensor, "seq_len batch num_heads state_dim 1"],
        t_index_ll: list[int, ...],
    ) -> tuple[Float[Tensor, "seq_len batch in_dim"], list[int]]:
        """
        Step-by-step forward pass through a sequence (sequential scan).

        :param in_vector_ll: Sequence of input vectors
        :param log_scalarA_bar_ll: Sequence of scalarA decay terms
        :param vecB_bar_ll: Sequence of B vectors
        :param vecC_ll: Sequence of C (readout) vectors
        :param t_index_ll: List of time indices
        :return: (outputs, used_indices)
        """
        hidden_state = 0.0

        out_ll = []
        out_t_index_ll = []
        for e, t_index in enumerate(t_index_ll):
            hidden_state: Float[Tensor, "batch num_heads state_dim head_dim"] = (
                log_scalarA_bar_ll[e].exp() * hidden_state
                + vecB_bar_ll[e] * in_vector_ll[e]
            )

            if t_index % self.subsampling == 0:
                out = (vecC_ll[e] * hidden_state).sum(dim=-2)
                out = rearrange(
                    out, "batch num_heads head_dim -> batch (num_heads head_dim)"
                )
                out_ll.append(out)
                out_t_index_ll.append(t_index)

        # seq_len, batch, in_dim
        return torch.stack(out_ll, dim=0), out_t_index_ll

    def forward_parallel(
        self,
        in_vector_ll: Float[Tensor, "seq_len batch num_heads 1 head_dim"],
        log_scalarA_bar_ll: Float[Tensor, "seq_len batch num_heads 1 1"],
        vecB_bar_ll: Float[Tensor, "seq_len batch num_heads state_dim 1"],
        vecC_ll: Float[Tensor, "seq_len batch num_heads state_dim 1"],
        t_index_ll: list[int, ...],
    ) -> tuple[Float[Tensor, "seq_len batch in_dim"], list[int]]:
        """
        Parallel (blockwise scan) forward pass for sequence tensor.

        :param in_vector_ll: Sequence input
        :param log_scalarA_bar_ll: ScalarA decay sequence
        :param vecB_bar_ll: B sequence
        :param vecC_ll: C sequence
        :param t_index_ll: Indices
        :return: (outputs, used_indices)
        """
        in_vector_ll = rearrange(
            in_vector_ll,
            "seq_len batch num_heads 1 head_dim -> batch seq_len num_heads head_dim",
        )
        log_scalarA_bar_ll = rearrange(
            log_scalarA_bar_ll, "seq_len batch num_heads 1 1 -> batch seq_len num_heads"
        )

        vecB_bar_ll = rearrange(
            vecB_bar_ll,
            "seq_len batch num_heads state_dim 1 -> batch seq_len num_heads state_dim",
        )
        vecC_ll = rearrange(
            vecC_ll,
            "seq_len batch num_heads state_dim 1 -> batch seq_len num_heads state_dim",
        )

        out_ll, _ = ssd_minimal_discrete(
            in_vector_ll, log_scalarA_bar_ll, vecB_bar_ll, vecC_ll, self.chunk_size
        )

        out_ll = rearrange(
            out_ll,
            "batch seq_len num_heads head_dim -> seq_len batch (num_heads head_dim)",
        )

        subsampling_mask = np.where(np.array(t_index_ll) % self.subsampling == 0)
        out_ll = out_ll[subsampling_mask]
        t_index_ll = [t for t in t_index_ll if t % self.subsampling == 0]

        return out_ll, t_index_ll

    def forward(
        self,
        in_vector_ll: Float[Tensor, "seq_len batch in_dim"],
        t_index_ll: list[int, ...],
    ) -> tuple[Float[Tensor, "seq_len batch out_dim"], list[int, ...]]:
        """
        General forward interface: chooses sequential or parallel API based on parallel_mode.
        Supports spatial-sequence batch input.

        :param in_vector_ll: [seq_len, batch, in_dim] or 4D spatial-sequence
        :param t_index_ll: List of time/sequence indices
        :return: (outputs, used_indices)
        """
        is_spatial_input = False
        if in_vector_ll.ndim == 4:
            is_spatial_input = True
            in_vector_ll, (height, width) = self.spatial_to_embedding(in_vector_ll)

        assert in_vector_ll.shape[-1] == self.in_dim

        # Project and obtain scalarA, vecB, vecC
        scalarA_ll, dt_ll, vecB_ll, vecC_ll = self.project(in_vector_ll)

        # reshape in_vector_ll to extract heads
        in_vector_ll = rearrange(
            in_vector_ll,
            "seq_len batch (num_heads head_dim) -> seq_len batch num_heads 1 head_dim",
            head_dim=self.head_dim,
        )

        # Continuous-time parameterization
        device = in_vector_ll.device
        time_scale = torch.diff(torch.tensor(t_index_ll), prepend=torch.zeros(1)).to(
            device
        )
        time_scale = rearrange(time_scale, "seq_len -> seq_len 1 1")

        log_scalarA_bar_ll, vecB_bar_ll = self.zoh_discretize(
            scalarA_ll, vecB_ll, dt_ll * time_scale
        )

        log_scalarA_bar_ll = log_scalarA_bar_ll.unsqueeze(-1).unsqueeze(-1)
        vecB_bar_ll = vecB_bar_ll.unsqueeze(-1)
        vecC_ll = vecC_ll.unsqueeze(-1)

        if self.parallel_mode:
            out_ll, t_index_ll = self.forward_parallel(
                in_vector_ll, log_scalarA_bar_ll, vecB_bar_ll, vecC_ll, t_index_ll
            )
        else:
            out_ll, t_index_ll = self.forward_sequential(
                in_vector_ll, log_scalarA_bar_ll, vecB_bar_ll, vecC_ll, t_index_ll
            )

        if is_spatial_input:
            out_ll = self.embedding_to_spatial(out_ll, height, width)

        return out_ll, t_index_ll


if __name__ == "__main__":
    from loguru import logger

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch = 1
    seq_len = 128
    num_heads = 4
    head_dim = 64
    num_groups = 1
    state_dim = 128
    subsampling = 1
    in_dim = num_heads * head_dim
    ssd_model = SSD(
        in_dim=in_dim,
        head_dim=head_dim,
        state_dim=state_dim,
        identity_init=False,
        subsampling=subsampling,
    )

    in_vector_ll = torch.ones(seq_len, batch, in_dim)

    out_online_ll = []
    t_index_ll = list(range(1, seq_len + 1))
    for e, t_index in enumerate(t_index_ll):
        out_online_ll.append(
            ssd_model.forward_online(in_vector_ll[e], time_instant=t_index)
        )
    out_online_ll = torch.stack(out_online_ll, dim=0)

    t_index_ll = list(range(1, seq_len + 1))
    out_sequential_ll, t_index_ll = ssd_model.forward(in_vector_ll, t_index_ll)
    logger.info(f"Output shape {out_sequential_ll.shape}")

    subsampling_factor = 64
    t_index_ll = list(range(subsampling_factor, seq_len + 1, subsampling_factor))
    out_subsampled_ll, t_index_ll = ssd_model.forward(
        in_vector_ll[subsampling_factor - 1 :: subsampling_factor], t_index_ll
    )
    logger.info(f"Output shape {out_subsampled_ll.shape}")

    ssd_model.parallel_mode = True
    t_index_ll = list(range(1, seq_len + 1))
    out_parallel_ll, t_index_ll = ssd_model.forward(in_vector_ll, t_index_ll)
    logger.info(f"Output shape {out_parallel_ll.shape}")
