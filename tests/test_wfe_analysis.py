"""Tests for WFE analysis functionality."""

from pathlib import Path

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pytest
from paos.classes.zernike import Zernike, PolyOrthoNorm
from photutils.aperture import EllipticalAperture

from stop_utils.wfe_analysis import (
    analyze_wfe_data,
    calculate_polynomials,
    fit_polynomials,
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


def test_calculate_polynomials(sample_wfe_data: np.ndarray) -> None:
    """Test orthonormal polynomial calculation."""
    # Setup
    mask = sample_wfe_data != 0
    size = sample_wfe_data.shape[0]
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)

    # Calculate orthonormal polynomials
    poly, A = calculate_polynomials(~mask, x, y, n_polynomials=15)
    polys = poly()

    # Check outputs
    assert polys.shape[0] == 15  # Number of polynomials
    assert polys.shape[1:] == sample_wfe_data.shape
    assert A.shape == (15, 15)  # Covariance matrix
    assert np.allclose(A, A.T)  # Should be symmetric


def test_fit_polynomials(sample_wfe_data: np.ndarray) -> None:
    """Test orthonormal polynomial fitting."""
    # Setup
    mask = sample_wfe_data != 0
    size = sample_wfe_data.shape[0]
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)

    # Fit orthonormal polynomials
    result = fit_polynomials(
        errormap=sample_wfe_data, pupil_mask=~mask, x=x, y=y, n_polynomials=15
    )

    # Check results
    assert len(result.coefficients) == 15
    assert result.pttf.shape == sample_wfe_data.shape
    assert result.model.shape == sample_wfe_data.shape
    assert result.residual.shape == sample_wfe_data.shape

    # Check PTTF is first 4 terms
    pttf_poly, _ = calculate_polynomials(~mask, x, y, 4)
    pttf_polys = pttf_poly()
    pttf_direct = np.sum(result.coefficients[:4].reshape(-1, 1, 1) * pttf_polys, axis=0)
    assert np.allclose(pttf_direct, result.pttf, rtol=1e-10)


def test_analyze_wfe_data(sample_wfe_file: Path) -> None:
    """Test complete WFE analysis pipeline."""
    # Run analysis
    result, params = analyze_wfe_data(sample_wfe_file, n_polynomials=15)

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
    """Test stability of orthonormal polynomial coefficient fitting."""
    # Run analysis twice
    result1, _ = analyze_wfe_data(sample_wfe_file, n_polynomials=15)
    result2, _ = analyze_wfe_data(sample_wfe_file, n_polynomials=15)

    # Results should be identical for same input
    assert np.allclose(result1.coefficients, result2.coefficients)


@pytest.mark.parametrize("n_polynomials", [10, 15])
@pytest.mark.parametrize("aspect", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("center_pos", [(0.5, 0.5), (0.4, 0.6), (0.3, 0.3)])
def test_fitting_roundtrip(
    n_polynomials: int,
    aspect: tuple[float, float],
    center_pos: tuple[float, float],
) -> None:
    """Test if fitting a known orthonormal polynomial combination recovers the coefficients.

    Args:
        n_polynomials: Number of polynomials to fit
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

    # Normalized coordinates for polynomial calculation
    x = (x_coords - x0) / a
    y = (y_coords - y0) / a
    xx, yy = np.meshgrid(x, y)
    rho: ma.MaskedArray = np.ma.masked_array(
        data=np.sqrt(xx**2 + yy**2), mask=pupil_mask
    )
    phi = np.arctan2(yy, xx)

    # 3. Generate random coefficients
    input_coefficients = np.random.normal(0, 1, n_polynomials) * 500

    # 4. Generate the "true" WFE map using PolyOrthoNorm
    poly = PolyOrthoNorm(
        n_polynomials,
        rho,
        phi,
        mask=pupil_mask,
        normalize=False,
        ordering="standard",
    )
    zkm = poly()
    true_wfe_map = np.sum(input_coefficients.reshape(-1, 1, 1) * zkm, axis=0)
    true_wfe_map_masked: ma.MaskedArray = np.ma.masked_array(
        true_wfe_map, mask=pupil_mask
    )

    # 5. Fit the generated WFE map
    result = fit_polynomials(
        errormap=true_wfe_map_masked.filled(np.nan),
        pupil_mask=pupil_mask,
        x=x,
        y=y,
        n_polynomials=n_polynomials,
    )

    # 6. Compare fitted coefficients to the original input coefficients
    print("\nTest configuration:")
    print(f"n_polynomials: {n_polynomials}")
    print(f"aspect ratio: {aspect}")
    print(f"center position: {center_pos}")
    print("\nResults:")
    print("Input Coeffs:", input_coefficients)
    print("Fitted Coeffs:", result.coefficients)
    print("Max diff:", np.max(np.abs(result.coefficients - input_coefficients)))

    assert np.allclose(
        result.coefficients, input_coefficients, atol=1e-9
    ), "Fitted coefficients do not match input coefficients"


def test_zernike_to_orthonormal_conversion() -> None:
    """Test conversion between Zernike and orthonormal polynomials.

    This test:
    1. Creates a WFE map using standard Zernike polynomials on a circular aperture
    2. Fits the map with orthonormal polynomials
    3. Converts back to Zernike coefficients
    4. Verifies the final coefficients match the input
    """
    # Setup
    size = 512  # Grid size
    base_radius = size // 8  # Base radius for scaling
    n_polynomials = 21

    # Create elliptical aperture parameters
    a = base_radius * 2.0  # Semi-major axis
    b = base_radius * 1.5  # Semi-minor axis
    x0 = size // 2  # Center x
    y0 = size // 2  # Center y
    theta = 0.0  # np.deg2rad(30)  # rotation in degrees

    # Create coordinates
    x_coords = np.arange(size)
    y_coords = np.arange(size)

    # Create elliptical aperture and mask
    aperture = EllipticalAperture((x0, y0), a, b, theta=theta)
    photutils_mask = aperture.to_mask(method="exact")
    pupil_mask = ~photutils_mask.to_image((size, size)).astype(bool)

    # Create normalized coordinates
    x = (x_coords - x0) / a
    y = (y_coords - y0) / a
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)

    # Generate input Zernike coefficients
    input_coefficients = np.random.normal(0, 1, n_polynomials) * 500  # nm

    # Create WFE map using standard Zernikes
    circular_poly = Zernike(
        n_polynomials,
        rho,
        phi,
        ordering="standard",
        normalize=False,
    )
    circular_basis = circular_poly()
    wfe_map = np.sum(input_coefficients.reshape(-1, 1, 1) * circular_basis, axis=0)
    wfe_map_masked: np.ma.MaskedArray = np.ma.masked_array(wfe_map, mask=pupil_mask)

    # Create orthonormal polynomial base
    rho_e: np.ma.MaskedArray = np.ma.masked_array(
        data=np.sqrt(xx**2 + yy**2), mask=pupil_mask
    )
    poly = PolyOrthoNorm(
        n_polynomials,
        rho_e,
        phi,
        mask=pupil_mask,
        ordering="standard",
        normalize=False,
    )

    # Fit with orthonormal polynomials
    result = fit_polynomials(
        errormap=wfe_map_masked.filled(np.nan),
        pupil_mask=pupil_mask,
        x=x,
        y=y,
        n_polynomials=n_polynomials,
    )

    coeff = result.coefficients
    zernikes = poly.toZernike(coeff)

    # Compare coefficients (allowing for some numerical differences)
    print("\nConversion test:")
    print(f"Input Zernike coefficients: {np.round(input_coefficients, 6)}")
    print(f"Recovered Zernike coefficients: {np.round(zernikes, 6)}")
    print(f"Max difference: {np.max(np.abs(zernikes - input_coefficients)):.3e}")

    # The coefficients should match within reasonable tolerance
    assert np.allclose(
        zernikes, input_coefficients, atol=1e-9, rtol=1e-3
    ), "Recovered Zernike coefficients do not match input coefficients"
