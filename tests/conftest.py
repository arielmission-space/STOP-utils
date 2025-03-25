"""Test configuration and fixtures."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from stop_utils.types import AnalysisConfig


@pytest.fixture
def sample_wfe_data() -> np.ndarray:
    """Generate sample WFE data for testing."""
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Create circular aperture
    r = np.sqrt(xx**2 + yy**2)
    mask = r <= 0.25

    # Create sample wavefront with some aberrations
    z = np.zeros((size, size))
    z[mask] = (
        0.5 * xx[mask]  # tip
        + 0.3 * yy[mask]  # tilt
        + 0.2 * (2 * r[mask] ** 2 - 1)  # defocus
    )

    # Add some noise
    z[mask] += np.random.normal(
        loc=0.0, scale=0.1, size=int(np.sum(mask))  # Cast to int for proper typing
    )

    # Mask outside aperture
    z[~mask] = 0

    return cast(np.ndarray, z)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_wfe_file(tmp_path: Path, sample_wfe_data: np.ndarray) -> Path:
    """Create temporary WFE data file."""
    wfe_file = tmp_path / "test_wfe.dat"
    np.savetxt(wfe_file, sample_wfe_data)
    return wfe_file


@pytest.fixture
def analysis_config(temp_output_dir: Path) -> AnalysisConfig:
    """Create sample analysis configuration."""
    return AnalysisConfig(
        n_zernike=15,
        save_coeffs=True,
        generate_plots=True,
        plot_format="png",
        output_dir=temp_output_dir,
    )
