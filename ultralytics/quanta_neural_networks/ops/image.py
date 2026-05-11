"""
Image and scientific array utilities for intensity, gamma correction, tone mapping, and inpainting.
"""
import random
from random import random

import cv2
import numpy as np
import torch
from numba import vectorize, float64, uint8, int64
from scipy import spatial


def intensity_from_empirical_mean(
    empirical_mean,
    quantum_eff: float = 0.1,
    srgb_gamma_correct: bool = True,
    gamma_correction: float = 1.0,
    quantile: float = 0.975,
):
    """
    Calculate linear image intensity from empirical photon mean.

    :param empirical_mean: Array or tensor, empirical photon mean
    :param quantum_eff: SPAD quantum efficiency
    :param srgb_gamma_correct: If True, apply sRGB gamma correction
    :param gamma_correction: Gamma factor for final mapping
    :param quantile: Upper quantile for tone mapping
    :return: Rescaled intensity
    """
    # Linear intensity

    input_is_torch_tensor = False
    if isinstance(empirical_mean, torch.Tensor):
        input_is_torch_tensor = True
        device = empirical_mean.device
        empirical_mean = empirical_mean.numpy(force=True)

    intensity = -np.log(np.clip(1 - empirical_mean, 1e-6, 1)) / quantum_eff

    output = tone_map_intensity(
        intensity,
        quantile=quantile,
        srgb_gamma_correct=srgb_gamma_correct,
        gamma_correction=gamma_correction,
    )
    if input_is_torch_tensor:
        output = torch.from_numpy(output).to(device)
    return output


def tone_map_intensity(
    intensity,
    quantile: float = 0.975,
    srgb_gamma_correct: bool = True,
    gamma_correction: float = 1.0,
):
    """
    Tone-map intensity to compress highlights and match display/gamma needs.
    """
    # Map 97th percentile to 1
    intensity = intensity / np.quantile(intensity, quantile)
    intensity = np.clip(intensity, 0, 1)

    # sRGB gamma correct
    if srgb_gamma_correct:
        intensity = linear_to_srgb(intensity)

    # Gamma correct
    if gamma_correction:
        intensity = gamma_correct(intensity, gamma_correction)

    return intensity


def gamma_correct(img, gamma_correction: float = 1.0):
    """
    Apply gamma curve correction to image data.
    """
    dtype = img.dtype
    if np.issubdtype(dtype, np.integer):
        img = img / 255.0
        img **= gamma_correction
        img = img * 255.0

    elif np.issubdtype(dtype, np.floating):
        img **= gamma_correction

    return img.astype(dtype)


def srgb_to_linear(srgb):
    """
    Decode sRGB-encoded image to linear intensity range.
    """
    dtype = srgb.dtype
    is_integer_subdtype = np.issubdtype(dtype, np.integer)

    if is_integer_subdtype:
        srgb = srgb.astype(float) / 255.0

    linear = srgb.copy()
    less = linear <= 0.04045
    linear[less] = linear[less] / 12.92
    linear[~less] = np.power((linear[~less] + 0.055) / 1.055, 2.4)

    if is_integer_subdtype:
        linear *= 255

    return linear.astype(dtype)


def linear_to_srgb(linear):
    """
    Encode linear image intensities as sRGB.
    """
    dtype = linear.dtype
    is_integer_subdtype = np.issubdtype(dtype, np.integer)

    if is_integer_subdtype:
        linear = linear.astype(float) / 255.0

    srgb = linear.copy()
    less = linear <= 0.0031308
    srgb[less] = linear[less] * 12.92
    srgb[~less] = 1.055 * np.power(linear[~less], 1.0 / 2.4) - 0.055

    if is_integer_subdtype:
        srgb *= 255

    return srgb.astype(dtype)


@vectorize(
    [
        uint8(uint8, float64, float64, float64),
        uint8(int64, float64, float64, float64),
        uint8(float64, float64, float64, float64),
    ],
    target="parallel",
)
def draw_bit_plane(
    photon_count_tensor,
    quantum_eff: float = 0.5,
    scale_factor: float = 0.5,
    photon_count_offset: float = 0.0,
):
    """
    Stochastic bit-plane simulation given photon statistics and SPAD parameters.
    """
    photo_electron_count = (
        quantum_eff * scale_factor * (photon_count_tensor + photon_count_offset)
    )
    prob = 1 - np.exp(-photo_electron_count)
    return random() < prob


def inpaint_img(img, mask, **inpaint_kwargs):
    """
    Inpaint missing/corrupt pixels in an image using OpenCV's inpainting.
    """
    inpaint_kwargs.update({"inpaintRadius": 10})

    orig_dtype = img.dtype
    if img.dtype == np.float64:
        img = img.astype(np.float32)

    return cv2.inpaint(
        src=img,
        inpaintMask=mask,
        flags=cv2.INPAINT_TELEA,
        **inpaint_kwargs,
    ).astype(orig_dtype)


def nearest_neighbor_inpaint(array: torch.Tensor, hot_pixel_mask: np.ndarray):
    """
    Inpaint hot pixels using nearest neighbors (KD-tree spatial search).
    """
    i_ll, j_ll = np.where(1 - hot_pixel_mask)
    tree = spatial.KDTree(list(zip(i_ll.ravel(), j_ll.ravel())))
    query_i_ll, query_j_ll = np.where(hot_pixel_mask)
    query_ll = np.stack([query_i_ll, query_j_ll], axis=-1)

    dd, ii = tree.query(query_ll, workers=-1)
    nearest_ll = tree.data[ii].astype(int)
    if isinstance(array, torch.Tensor):
        nearest_ll = torch.from_numpy(nearest_ll).int().to(array.device)
    nearest_i_ll, nearest_j_ll = nearest_ll[:, 0], nearest_ll[:, 1]

    array[query_i_ll, query_j_ll] = array[nearest_i_ll, nearest_j_ll]
    return array

