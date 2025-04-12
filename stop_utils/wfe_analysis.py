"""Core functionality for Wavefront Error analysis."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from paos.classes.zernike import PolyOrthoNorm
from photutils.aperture import EllipticalAperture
from skimage.measure import label, regionprops

from . import logger
from .converters import load_zemax_wfe
from .types import EllipticalParams, WFEResult

# Set logger context for analysis module
logger = logger.bind(context="wfe_analysis")


def load_wfe_data(
    file_path: Path, file_format: Optional[str] = None
) -> npt.NDArray[np.float64]:
    """Load WFE data from file and convert to nanometers.
    Handles different formats based on the provided format string or defaults to generic.
    """
    logger.debug(f"Loading WFE data from {file_path}, specified format: {file_format}")

    if file_format:
        format_lower = file_format.lower()
        if format_lower == "zemax":
            logger.info("Using specified Zemax file format.")
            data_nm = load_zemax_wfe(str(file_path))
        # elif format_lower == "codev":
        #     logger.info("Using specified CODEV file format.")
        #     data_nm = load_codev_wfe(str(file_path))
        # Add elif blocks here for other formats
        # elif format_lower == "some_other_format":
        #     data_nm = load_some_other_format(file_path)
        else:
            raise ValueError(f"Unsupported file format specified: '{file_format}'")
    else:
        # Default to generic format if not specified
        logger.warning(
            "File format not specified, defaulting to generic .dat format. "
            "Use --format to specify a different format (e.g., --format zemax)."
        )
        try:
            data = np.loadtxt(file_path)
            data_nm = data * 1e9  # Convert to nm
        except Exception as e:
            logger.error(f"Failed to load {file_path} as generic format: {e}")
            logger.error(
                "Check the file content or specify the correct format with --format."
            )
            raise

    logger.debug(
        f"Loaded data shape: {data_nm.shape}, range: [{data_nm.min():.2f}, {data_nm.max():.2f}] nm"
    )
    return data_nm


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


def calculate_polynomials(
    pupil_mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_polynomials: int = 15,
) -> Tuple[PolyOrthoNorm, npt.NDArray[np.float64]]:
    """Calculate orthonormal polynomials for given coordinates.

    Args:
        pupil_mask: Boolean mask array
        x: X coordinates array
        y: Y coordinates array
        n_polynomials: Number of polynomials to calculate

    Returns:
        tuple: (Orthonormal polynomials array, Covariance matrix)
    """
    xx, yy = np.meshgrid(x, y)
    phi = np.arctan2(yy, xx)
    rho: npt.NDArray[np.float64] = np.ma.masked_array(
        data=np.sqrt(yy**2 + xx**2), mask=pupil_mask, fill_value=0.0
    )

    logger.debug(f"Calculating {n_polynomials} orthonormal polynomials")
    poly = PolyOrthoNorm(n_polynomials, rho, phi, normalize=False, ordering="standard")
    A = poly.cov()
    logger.debug("Orthonormal polynomials calculated")

    return poly, A


def fit_polynomials(
    errormap: npt.NDArray[np.float64],
    pupil_mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_polynomials: int = 15,
) -> WFEResult:
    """Fit orthonormal polynomials to WFE data.

    Args:
        errormap: Wavefront error data array
        pupil_mask: Boolean mask array
        x: X coordinates array
        y: Y coordinates array
        n_polynomials: Number of orthonormal polynomials to use

    Returns:
        WFEResult object containing:
        - raw: Errormap to be fitted
        - coefficients: Fitted orthonormal polynomial coefficients
        - zernikes: Zernike coefficients (after removing first 3 terms from the orthonormal fit)
        - PTTF component map
        - Complete model map
        - Residual error map
    """
    masked_error: npt.NDArray[np.float64] = np.ma.masked_array(
        errormap, mask=pupil_mask
    )

    # Calculate orthonormal polynomials
    logger.debug(f"Fitting {n_polynomials} orthonormal polynomials to WFE data")
    poly, A = calculate_polynomials(pupil_mask, x, y, n_polynomials)
    polys = poly()
    B = np.ma.mean(polys * masked_error, axis=(-2, -1))
    coeff = np.linalg.lstsq(A, B, rcond=-1)[0]
    logger.debug("Orthonormal polynomial fitting completed")

    # Calculate model using computed coefficients
    model = np.sum(coeff.reshape(-1, 1, 1) * polys, axis=0)

    # Calculate PTTF (first 4 terms)
    pttf_poly, _ = calculate_polynomials(pupil_mask, x, y, 4)
    pttf_polys = pttf_poly()
    pttf = np.sum(coeff[:4].reshape(-1, 1, 1) * pttf_polys, axis=0)

    # Calculate residual
    residual = masked_error - model

    # Calculate Zernike coefficients
    zernikes = np.copy(coeff)
    zernikes[:3] = 0.0  # Set Piston, Tip, Tilt to zero
    zernikes = poly.toZernike(zernikes)

    return WFEResult(
        raw=masked_error,
        coefficients=coeff,
        zernikes=zernikes,
        pttf=pttf,
        model=model,
        residual=residual,
    )


def analyze_wfe_data(
    wfe_file: Path,
    n_polynomials: int = 15,
    file_format: Optional[str] = None,
) -> Tuple[WFEResult, EllipticalParams]:
    """Analyze WFE data file.

    Args:
        wfe_file: Path to WFE data file
        n_polynomials: Number of polynomials to use
        file_format: The format of the input file (e.g., 'zemax').

    Returns:
        tuple: (WFEResult object, EllipticalParams object)

    Raises:
        FileNotFoundError: If wfe_file does not exist
        ValueError: If data cannot be processed
    """
    if not wfe_file.exists():
        raise FileNotFoundError(f"WFE data file not found: {wfe_file}")

    # Load and preprocess data using the updated loader with format
    logger.info(f"Analyzing WFE data from {wfe_file}")
    errormap = load_wfe_data(wfe_file, file_format)  # Pass format here
    errormap_ma = np.ma.masked_where(errormap == 0, errormap)

    # Create mask and find elliptical aperture
    logger.debug("Creating elliptical aperture mask")
    mask = label(~errormap_ma.mask)
    aperture, params = mask_to_elliptical_aperture(mask)
    logger.debug(
        f"Elliptical aperture found: center=({params.x0:.1f}, {params.y0:.1f}), "
        f"axes=({params.a:.1f}, {params.b:.1f}), theta={params.theta:.3f}"
    )

    # Create normalized coordinates
    shape = errormap.shape
    x = (np.arange(shape[1]) - params.x0) / params.a
    y = (np.arange(shape[0]) - params.y0) / params.a

    # Fit orthonormal polynomials
    result = fit_polynomials(
        errormap=errormap,
        pupil_mask=~mask.astype(bool),
        x=x,
        y=y,
        n_polynomials=n_polynomials,
    )

    logger.info(f"Analysis complete - RMS error: {result.rms(result.residual):.2f} nm")
    return result, params
