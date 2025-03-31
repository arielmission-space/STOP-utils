"""Core functionality for Wavefront Error analysis."""

from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
from paos.classes.zernike import PolyOrthoNorm
from photutils.aperture import EllipticalAperture
from skimage.measure import label, regionprops

from .types import EllipticalParams, WFEResult


def load_wfe_data(file_path: Path) -> npt.NDArray[np.float64]:
    """Load WFE data from file and convert to nanometers."""
    data = np.loadtxt(file_path)
    return data * 1e9  # Convert to nm


def mask_to_elliptical_aperture(
    label_img: npt.NDArray[np.int_],
) -> Tuple[EllipticalAperture, EllipticalParams]:
    """Convert an elliptical mask to a photutils EllipticalAperture.

    Args:
        label_img: Labeled image array where each region has a unique integer value

    Returns:
        tuple: (EllipticalAperture object, EllipticalParams object)

    Raises:
        ValueError: If no regions found in mask
    """
    props = regionprops(label_img)
    if len(props) == 0:
        raise ValueError("No regions found in mask")

    prop = props[0]
    y0, x0 = prop.centroid
    a = prop.major_axis_length / 2
    b = prop.minor_axis_length / 2
    theta = prop.orientation  # in radians

    # Convert to photutils convention (angle in radians counter-clockwise from positive x-axis)
    theta = np.pi / 2 - theta

    params = EllipticalParams(x0=x0, y0=y0, a=a, b=b, theta=theta)
    aperture = EllipticalAperture((x0, y0), a, b, theta=theta)

    return aperture, params


def calculate_zernike(
    pupil_mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_zernike: int = 15,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate Zernike polynomials for given coordinates.

    Args:
        pupil_mask: Boolean mask array
        x: X coordinates array
        y: Y coordinates array
        n_zernike: Number of Zernike polynomials to calculate

    Returns:
        tuple: (Zernike polynomials array, Covariance matrix)
    """
    xx, yy = np.meshgrid(x, y)
    phi = np.arctan2(yy, xx)
    rho: npt.NDArray[np.float64] = np.ma.masked_array(
        data=np.sqrt(yy**2 + xx**2), mask=pupil_mask, fill_value=0.0
    )

    poly = PolyOrthoNorm(n_zernike, rho, phi, normalize=False, ordering="standard")
    zkm = poly()
    A = poly.cov()

    return zkm, A


def fit_zernike(
    errormap: npt.NDArray[np.float64],
    pupil_mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_zernike: int = 15,
) -> WFEResult:
    """Fit Zernike polynomials to WFE data.

    Args:
        errormap: Wavefront error data array
        pupil_mask: Boolean mask array
        x: X coordinates array
        y: Y coordinates array
        n_zernike: Number of Zernike polynomials to use

    Returns:
        WFEResult object containing:
        - coefficients: Fitted Zernike coefficients
        - PTTF component map
        - Complete model map
        - Residual error map
    """
    masked_error: npt.NDArray[np.float64] = np.ma.masked_array(
        errormap, mask=pupil_mask
    )

    # Calculate Zernike polynomials
    zkm, A = calculate_zernike(pupil_mask, x, y, n_zernike)
    B = np.ma.mean(zkm * masked_error, axis=(-2, -1))
    coeff = np.linalg.lstsq(A, B, rcond=-1)[0]

    # Calculate model using computed coefficients
    model = np.sum(coeff.reshape(-1, 1, 1) * zkm, axis=0)

    # Calculate PTTF (first 4 terms)
    pttf_zkm, _ = calculate_zernike(pupil_mask, x, y, 4)
    pttf = np.sum(coeff[:4].reshape(-1, 1, 1) * pttf_zkm, axis=0)

    # Calculate residual
    residual = masked_error - model

    return WFEResult(coefficients=coeff, pttf=pttf, model=model, residual=residual)


def analyze_wfe_data(
    wfe_file: Path, n_zernike: int = 15
) -> Tuple[WFEResult, EllipticalParams]:
    """Analyze WFE data file.

    Args:
        wfe_file: Path to WFE data file
        n_zernike: Number of Zernike polynomials to use

    Returns:
        tuple: (WFEResult object, EllipticalParams object)

    Raises:
        FileNotFoundError: If wfe_file does not exist
        ValueError: If data cannot be processed
    """
    if not wfe_file.exists():
        raise FileNotFoundError(f"WFE data file not found: {wfe_file}")

    # Load and preprocess data
    errormap = load_wfe_data(wfe_file)
    errormap_ma = np.ma.masked_where(errormap == 0, errormap)

    # Create mask and find elliptical aperture
    mask = label(~errormap_ma.mask)
    aperture, params = mask_to_elliptical_aperture(mask)

    # Create normalized coordinates
    shape = errormap.shape
    x = (np.arange(shape[1]) - params.x0) / params.a
    y = (np.arange(shape[0]) - params.y0) / params.a

    # Fit Zernike polynomials
    result = fit_zernike(
        errormap=errormap, pupil_mask=~mask.astype(bool), x=x, y=y, n_zernike=n_zernike
    )

    return result, params
