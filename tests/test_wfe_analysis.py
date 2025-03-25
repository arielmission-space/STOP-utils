"""Tests for WFE analysis functionality."""

from pathlib import Path
from typing import Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest

from stop_utils.types import WFEResult
from stop_utils.wfe_analysis import (analyze_wfe_data, calculate_zernike,
                                     fit_zernike, load_wfe_data,
                                     mask_to_elliptical_aperture)


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
