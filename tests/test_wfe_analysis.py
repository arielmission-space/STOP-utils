"""Tests for WFE analysis functionality."""

from pathlib import Path
from typing import Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest
from paos.classes.zernike import PolyOrthoNorm
from photutils.aperture import EllipticalAperture

from stop_utils.types import WFEResult
from stop_utils.wfe_analysis import (
    analyze_wfe_data,
    calculate_zernike,
    fit_zernike,
    load_wfe_data,
    mask_to_elliptical_aperture,
)


def test_load_wfe_data(sample_wfe_file: Path) -> None:
    """Test loading WFE data from file."""
    data = load_wfe_data(sample_wfe_file)
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert not np.all(data == 0)
    assert np.any(data > 1e-9)  # Should be in nm


def test_mask_to_elliptical_aperture(sample_wfe_data: np.ndarray) -> None:
    """Test elliptical aperture creation from mask."""
    # Create mask
    mask = (sample_wfe_data != 0).astype(int)

    # Get aperture and parameters
    aperture, params = mask_to_elliptical_aperture(mask)

    # Check parameters are reasonable
    assert 0 <= params.x0 <= sample_wfe_data.shape[1]
    assert 0 <= params.y0 <= sample_wfe_data.shape[0]
    assert params.a > 0
    assert params.b > 0
    assert -np.pi <= params.theta <= np.pi


def test_calculate_zernike(sample_wfe_data: np.ndarray) -> None:
    """Test Zernike polynomial calculation."""
    # Setup
    mask = sample_wfe_data != 0
    size = sample_wfe_data.shape[0]
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)

    # Calculate Zernike polynomials
    zkm, A = calculate_zernike(~mask, x, y, n_zernike=15)

    # Check outputs
    assert zkm.shape[0] == 15  # Number of polynomials
    assert zkm.shape[1:] == sample_wfe_data.shape
    assert A.shape == (15, 15)  # Covariance matrix
    assert np.allclose(A, A.T)  # Should be symmetric


def test_fit_zernike(sample_wfe_data: np.ndarray) -> None:
    """Test Zernike polynomial fitting."""
    # Setup
    mask = sample_wfe_data != 0
    size = sample_wfe_data.shape[0]
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)

    # Fit Zernike polynomials
    result = fit_zernike(
        errormap=sample_wfe_data, pupil_mask=~mask, x=x, y=y, n_zernike=15
    )

    # Check results
    assert len(result.coefficients) == 15
    assert result.pttf.shape == sample_wfe_data.shape
    assert result.model.shape == sample_wfe_data.shape
    assert result.residual.shape == sample_wfe_data.shape

    # Check PTTF is first 4 terms
    pttf_zkm, _ = calculate_zernike(~mask, x, y, 4)
    pttf_direct = np.sum(result.coefficients[:4].reshape(-1, 1, 1) * pttf_zkm, axis=0)
    assert np.allclose(pttf_direct, result.pttf, rtol=1e-10)


def test_analyze_wfe_data(sample_wfe_file: Path) -> None:
    """Test complete WFE analysis pipeline."""
    # Run analysis
    result, params = analyze_wfe_data(sample_wfe_file, n_zernike=15)

    # Check results
    assert len(result.coefficients) == 15
    assert isinstance(result.rms_error(), float)
    assert isinstance(result.peak_to_valley(), float)

    # Check parameters
    assert params.a > 0
    assert params.b > 0
    assert -np.pi <= params.theta <= np.pi


def test_error_handling() -> None:
    """Test error handling in analysis functions."""
    # Test empty mask
    empty_mask: npt.NDArray[np.int_] = np.zeros((10, 10), dtype=np.int_)
    with pytest.raises(ValueError, match="No regions found in mask"):
        mask_to_elliptical_aperture(empty_mask)

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        analyze_wfe_data(Path("nonexistent.dat"))


def test_coefficient_stability(sample_wfe_file: Path) -> None:
    """Test stability of Zernike coefficient fitting."""
    # Run analysis twice
    result1, _ = analyze_wfe_data(sample_wfe_file, n_zernike=15)
    result2, _ = analyze_wfe_data(sample_wfe_file, n_zernike=15)

    # Results should be identical for same input
    assert np.allclose(result1.coefficients, result2.coefficients)


@pytest.mark.parametrize("n_zernike", [10, 15, 21])
@pytest.mark.parametrize("aspect", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("center_pos", [(0.5, 0.5), (0.4, 0.6), (0.3, 0.3)])
def test_zernike_fitting_roundtrip(
    n_zernike: int,
    aspect: tuple[float, float],
    center_pos: tuple[float, float],
) -> None:
    """Test if fitting a known Zernike combination recovers the coefficients.

    Args:
        n_zernike: Number of Zernike polynomials to fit
        aspect: Tuple of (a_factor, b_factor) for ellipse aspect ratio
        center_pos: Tuple of (x_frac, y_frac) for relative center position
    """
    # 1. Define parameters
    size = 512  # Grid size
    base_radius = size // 8  # Base radius for scaling
    a_factor, b_factor = aspect
    x_frac, y_frac = center_pos

    # Calculate ellipse parameters
    a = base_radius * a_factor
    b = base_radius * b_factor
    x0 = int(size * x_frac)
    y0 = int(size * y_frac)
    theta = np.pi / 6  # Fixed rotation for all tests

    # 2. Create coordinates and mask using photutils
    x_coords = np.arange(size)
    y_coords = np.arange(size)

    aperture = EllipticalAperture((x0, y0), a, b, theta=theta)
    photutils_mask = aperture.to_mask(method="exact")
    pupil_mask = ~photutils_mask.to_image((size, size)).astype(bool)

    # Normalized coordinates for Zernike calculation
    x = (x_coords - x0) / a
    y = (y_coords - y0) / a
    xx, yy = np.meshgrid(x, y)
    rho = np.ma.masked_array(data=np.sqrt(xx**2 + yy**2), mask=pupil_mask)
    phi = np.arctan2(yy, xx)

    # 3. Generate random coefficients with controlled magnitudes
    # Higher order terms typically have smaller magnitudes
    magnitudes = np.exp(-np.arange(n_zernike) / 5)  # Exponential decay
    input_coefficients = np.random.normal(0, 1, n_zernike) * magnitudes * 500

    # 4. Generate the "true" WFE map using PolyOrthoNorm
    poly = PolyOrthoNorm(
        n_zernike,
        rho,
        phi,
        mask=pupil_mask,
        normalize=False,
        ordering="standard",
    )
    zkm = poly()
    true_wfe_map = np.sum(input_coefficients.reshape(-1, 1, 1) * zkm, axis=0)
    true_wfe_map_masked = np.ma.masked_array(true_wfe_map, mask=pupil_mask)

    # 5. Fit the generated WFE map
    result = fit_zernike(
        errormap=true_wfe_map_masked.filled(np.nan),
        pupil_mask=pupil_mask,
        x=x,
        y=y,
        n_zernike=n_zernike,
    )

    # 6. Compare fitted coefficients to the original input coefficients
    print("\nTest configuration:")
    print(f"n_zernike: {n_zernike}")
    print(f"aspect ratio: {aspect}")
    print(f"center position: {center_pos}")
    print("\nResults:")
    print("Input Coeffs:", input_coefficients)
    print("Fitted Coeffs:", result.coefficients)
    print("Max diff:", np.max(np.abs(result.coefficients - input_coefficients)))

    assert np.allclose(
        result.coefficients, input_coefficients, atol=1e-9
    ), "Fitted coefficients do not match input coefficients"
