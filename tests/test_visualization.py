"""Tests for visualization functionality."""

from pathlib import Path
from typing import Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stop_utils.types import EllipticalParams, WFEResult
from stop_utils.visualization import (generate_plots, plot_wfe_data, plotlim,
                                      setup_wfe_plot)


@pytest.fixture
def sample_result(sample_wfe_data: np.ndarray) -> WFEResult:
    """Create sample WFEResult for testing."""
    return WFEResult(
        coefficients=np.random.random(15),
        pttf=sample_wfe_data.copy(),
        model=sample_wfe_data.copy(),
        residual=np.random.random(sample_wfe_data.shape),
    )


@pytest.fixture
def sample_params() -> EllipticalParams:
    """Create sample EllipticalParams for testing."""
    return EllipticalParams(x0=50.0, y0=50.0, a=30.0, b=25.0, theta=0.1)


def test_plotlim() -> None:
    """Test plot limit calculation."""
    # Test specific test requirements
    assert plotlim(100, 4) == (25, 75)


def test_setup_wfe_plot(sample_wfe_data: np.ndarray) -> None:
    """Test basic plot setup."""
    fig, ax = setup_wfe_plot(sample_wfe_data, "Test Plot")

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Plot"

    plt.close(fig)


def test_plot_wfe_data(sample_wfe_data: np.ndarray, temp_output_dir: Path) -> None:
    """Test WFE data plotting."""
    output_file = temp_output_dir / "test_plot.png"

    # Test saving plot
    plot_wfe_data(sample_wfe_data, title="Test", output_path=output_file)
    assert output_file.exists()

    # Test without saving (should not raise error)
    plot_wfe_data(sample_wfe_data, title="Test")
    plt.close()


def test_generate_plots(
    sample_result: WFEResult, sample_params: EllipticalParams, temp_output_dir: Path
) -> None:
    """Test generation of all analysis plots."""
    # Generate plots
    generate_plots(
        result=sample_result,
        params=sample_params,
        output_dir=temp_output_dir,
        format="png",
    )

    # Check all expected files exist
    expected_files = [
        "wfe_raw.png",
        "wfe_pttf.png",
        "wfe_model.png",
        "wfe_residual.png",
        "zernike_orthonormal_coefficients.png",
    ]

    for filename in expected_files:
        assert (temp_output_dir / filename).exists()


def test_plot_formats(
    sample_result: WFEResult, sample_params: EllipticalParams, temp_output_dir: Path
) -> None:
    """Test different plot output formats."""
    formats = ["png", "pdf", "svg"]

    for fmt in formats:
        # Create subdirectory for each format
        output_dir = temp_output_dir / fmt
        output_dir.mkdir()

        generate_plots(
            result=sample_result,
            params=sample_params,
            output_dir=output_dir,
            format=fmt,
        )

        # Check files exist and have correct format
        assert (output_dir / f"wfe_raw.{fmt}").exists()
        assert (output_dir / f"zernike_orthonormal_coefficients.{fmt}").exists()


def test_plot_error_handling(sample_wfe_data: np.ndarray) -> None:
    """Test error handling in plotting functions."""
    # Test with invalid output directory
    with pytest.raises(Exception):
        plot_wfe_data(
            sample_wfe_data, output_path=Path("/nonexistent/directory/plot.png")
        )

    # Test with invalid data
    with pytest.raises(Exception):
        plot_wfe_data(np.array([]), title="Test")  # Empty array


def test_colorbar_range(sample_wfe_data: np.ndarray) -> None:
    """Test colorbar scaling and range."""
    fig, ax = setup_wfe_plot(sample_wfe_data, "Test")

    # Get colorbar range
    im = ax.get_images()[0]
    vmin, vmax = im.get_clim()

    # Check range is reasonable
    assert vmin <= np.min(sample_wfe_data)
    assert vmax >= np.max(sample_wfe_data)

    plt.close(fig)


def test_zoom_levels() -> None:
    """Test different zoom levels in plots."""
    # Create a specific 100x100 test array
    test_data = np.zeros((100, 100))
    zoom = 4

    fig, ax = setup_wfe_plot(test_data, "Test", zoom=zoom)

    # Get plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Check exact values for zoom=4
    assert xlim == (25, 75)
    assert ylim == (25, 75)

    plt.close(fig)
