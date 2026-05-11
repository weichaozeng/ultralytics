"""
Array and tensor utility operations for computer vision and scientific computing.
"""
from math import ceil, floor
from math import log2
from random import random
from typing import Tuple

import numpy as np
import torch
from numba import njit, vectorize, int64, float64
from torch import Tensor


def shift_2d_replace(data, dx, dy, constant=False):
    """
    Shift a 2D numpy array, replacing the rolled values with a constant.
    
    :param data: The 2D numpy array to be shifted
    :param dx: The shift in x direction (columns)
    :param dy: The shift in y direction (rows)
    :param constant: The constant to replace rolled values
    :return: Shifted array with constant values in new/rolled cells
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def axis_varying_roll(array, roll_i, roll_j, shift_k):
    """
    Roll slices in a 3D array along 1st and 2nd axes, with axis-varying shift indices for each slice of 3rd axis.
    """
    assert roll_i.ndim == roll_j.ndim == shift_k.ndim == 1
    assert shift_k.size == roll_i.size == roll_j.size
    assert ((0 <= shift_k) & (shift_k < array.shape[2])).all()

    h, w, _ = array.shape
    out_array = np.zeros((h, w, shift_k.size), dtype=array.dtype)

    for e in range(shift_k.size):
        out_array[..., e] = np.roll(
            array[..., shift_k[e]], (roll_i[e], roll_j[e]), axis=(0, 1)
        )

    return out_array


def axis_varying_slice(array, roll_i, roll_j, shift_k):
    """
    Slice a 3D array with variable rolls on each axis; extracted slices can vary in i, j, k.
    """
    assert roll_i.ndim == roll_j.ndim == shift_k.ndim == 1
    assert shift_k.size == roll_i.size == roll_j.size
    assert ((0 <= shift_k) & (shift_k < array.shape[2])).all()

    h, w, _ = array.shape
    i_grid, j_grid = meshgrid(
        np.arange(h, dtype=int), np.arange(w, dtype=int), indexing="ij"
    )
    k_grid = np.zeros_like(i_grid)


def normalize_uint8(img):
    """
    Normalize an image to [0, 255] in uint8 dtype, input can be float or int.
    """
    return float_to_uint8(normalize(img))


def normalize(img):
    """
    Normalize input image to [0, 1]. Return 0.5 array if max==min.
    """
    if img.max() == img.min():
        return torch.ones_like(img) * 0.5
    return (img - img.min()) / (img.max() - img.min())


def float_to_uint8(img):
    """
    Convert a float or integer image to uint8.
    """
    if isinstance(img, Tensor):
        if img.is_floating_point():
            return (img * 255).to(torch.uint8)
        else:
            return img.to(torch.uint8)
    elif isinstance(img, np.ndarray):
        if np.issubdtype(img.dtype, np.floating):
            return (img * 255).astype(np.uint8)
        else:
            return img.astype(np.uint8)
    else:
        raise NotImplementedError


@njit(cache=True)
def meshgrid(x, y, indexing="ij"):
    """
    Create a meshgrid (Numba-accelerated).
    """
    assert indexing in ("ij", "xy")

    i_grid = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    j_grid = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for i in range(x.size):
        for j in range(y.size):
            i_grid[i, j] = i
            j_grid[i, j] = j

    if indexing == "ij":
        return i_grid, j_grid
    elif indexing == "xy":
        return i_grid.T, j_grid.T


def cartesian_product(*arrays):
    """
    Compute the Cartesian product of 1D arrays.
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


@vectorize([int64(float64)])
def dither_round(x: float):
    """
    Unbiased stochastic rounding: returns floor(x) with probability ceil(x)-x, else ceil(x).
    """
    x_int = floor(x)
    x_frac = x - x_int
    e = random() < x_frac
    return x_int + e


def nearest_power_of_two(number):
    """
    Returns the next power of two not less than 'number'.
    """
    # Returns next power of two following 'number'
    return pow(2, ceil(log2(number)))


def zero_pad_to_power_2(array, axis: Tuple = None):
    """
    Pad array so that shapes along axis/axes are power-of-two, pad at array end.
    """
    if isinstance(axis, int):
        axis = (axis,)

    pad_ll = []
    for e, dim in enumerate(array.shape):
        if e in axis:
            dim_deficit = nearest_power_of_two(dim) - dim
            pad_ll.append([0, dim_deficit])
        else:
            pad_ll.append([0, 0])

    return np.pad(array, pad_ll)


def pad_to_multiple(array: np.ndarray, factor: int, axes: tuple):
    """
    Pad np.ndarray so its shape on the listed axes is a multiple of 'factor'.
    """
    # Figure out how much we should pad along each axis.
    pad_amounts = []
    for axis in range(array.ndim):
        neg_axis = -(array.ndim - axis)
        if (axis in axes or neg_axis in axes) and (array.shape[axis] % factor != 0):
            axis_padding = factor - array.shape[axis] % factor
        else:
            axis_padding = 0
        pad_amounts.append((0, axis_padding))

    # Do the padding.
    return np.pad(array, tuple(pad_amounts))


def pad_to_size(x, size, pad_tensor=None):
    """
    Pad a torch tensor or numpy array to a target shape using constant tensor fill.
    """
    # padding = [0, size[1] - x.shape[-1], 0, size[0] - x.shape[-2]]
    # x = func.pad(x, padding, fill=0, padding_mode="constant")
    # The two lines above are not working as expected - maybe there's a
    # bug in func.pad? In the meantime we'll use the concat-based
    # padding code below.
    if pad_tensor is None:
        pad_tensor = torch.zeros((1,) * x.ndim, dtype=x.dtype, device=x.device)
    for dim in list(range(-1, -len(size) - 1, -1)):
        expand_shape = list(x.shape)
        expand_shape[dim] = size[dim] - x.shape[dim]
        if expand_shape[dim] == 0:
            continue

        # torch.concat allocates a new tensor. So, we're safe to use
        # torch.expand here (instead of torch.repeat) without worrying
        # about different elements of x referencing the same data.
        x = torch.concat([x, pad_tensor.expand(expand_shape)], dim)
    return x


def reduce_masked_mean(x, mask, dim=None, keepdim=False, eps: float = 1e-6):
    """
    Compute mean over elements where mask==True. Support multi-dim tensors and custom axes.
    """
    # returns shape-1
    # axis can be a list of axes
    assert x.shape == mask.shape
    kwargs = dict(dim=dim, keepdim=keepdim)

    return (x * mask).sum(**kwargs) / mask.sum(**kwargs).clamp(min=eps)


def reduce_masked_median(x, mask, dim=None):
    """
    Compute the median of x across specified mask and dims (NaN-masked).
    """
    # returns shape-1
    # axis can be a list of axes
    # No dim argument here
    x_nan = x.float().masked_fill(~mask.bool(), float("nan"))
    if dim is None:
        x_median = x_nan.nanmedian()
    else:
        x_median, _ = x_nan.nanmedian(dim=dim)
    return x_median


def loguniform(low=0, high=1, size=None):
    """
    Draw a sample from a log-uniform distribution in [low...high] (float sample).
    """
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


def floor_multiple_of(x: int, base: int = 5) -> int:
    """
    Return x floored to the largest multiple of base not greater than x.
    """
    return base * floor(x / base)


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Improved quantile implementation to overcome torch limitations; uses kthvalue.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)
