"""Tests for command-line interface."""

import json
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from stop_utils.cli import app

# Set up CLI runner with isolated filesystem
runner = CliRunner(mix_stderr=False)


def test_analyze_wfe_basic(sample_wfe_file: Path, temp_output_dir: Path) -> None:
    """Test basic WFE analysis command."""
    result = runner.invoke(
        app,
        ["analyze", str(sample_wfe_file), str(temp_output_dir)],
        standalone_mode=False,
    )

    assert result.exit_code == 0
    assert "Analysis complete!" in result.stdout

    # Check output files
    assert (temp_output_dir / "wfe_raw.png").exists()
    assert (temp_output_dir / "zernike_orthonormal_coefficients.json").exists()


def test_analyze_wfe_no_plots(sample_wfe_file: Path, temp_output_dir: Path) -> None:
    """Test analysis with plot generation disabled."""
    result = runner.invoke(
        app,
        ["analyze", str(sample_wfe_file), str(temp_output_dir), "--no-plots"],
        standalone_mode=False,
    )

    assert result.exit_code == 0
    assert (temp_output_dir / "zernike_orthonormal_coefficients.json").exists()
    assert not (temp_output_dir / "wfe_raw.png").exists()


def test_analyze_wfe_custom_format(
    sample_wfe_file: Path, temp_output_dir: Path
) -> None:
    """Test analysis with custom plot format."""
    result = runner.invoke(
        app,
        ["analyze", str(sample_wfe_file), str(temp_output_dir), "--plot-format", "pdf"],
        standalone_mode=False,
    )

    assert result.exit_code == 0
    assert (temp_output_dir / "wfe_raw.pdf").exists()


def test_analyze_wfe_custom_zernike(
    sample_wfe_file: Path, temp_output_dir: Path
) -> None:
    """Test analysis with custom number of Zernike polynomials."""
    result = runner.invoke(
        app,
        ["analyze", str(sample_wfe_file), str(temp_output_dir), "--nzernike", "20"],
        standalone_mode=False,
    )

    assert result.exit_code == 0

    # Check coefficient count
    coeff_file = temp_output_dir / "zernike_orthonormal_coefficients.json"
    with open(coeff_file) as f:
        data = json.load(f)
        assert len(data["coefficients"]) == 20


def test_analyze_wfe_no_coeffs(sample_wfe_file: Path, temp_output_dir: Path) -> None:
    """Test analysis without saving coefficients."""
    result = runner.invoke(
        app,
        ["analyze", str(sample_wfe_file), str(temp_output_dir), "--no-save-coeffs"],
        standalone_mode=False,
    )

    assert result.exit_code == 0
    assert not (temp_output_dir / "zernike_orthonormal_coefficients.json").exists()


def test_analyze_wfe_invalid_input() -> None:
    """Test error handling for invalid input file."""
    result = runner.invoke(app, ["analyze", "nonexistent.dat", "output/"])

    assert result.exit_code == 2  # Typer exit code for argument validation error
    assert "does not exist" in result.stderr


def test_analyze_wfe_invalid_format(
    sample_wfe_file: Path, temp_output_dir: Path
) -> None:
    """Test error handling for invalid plot format."""
    result = runner.invoke(
        app,
        [
            "analyze",
            str(sample_wfe_file),
            str(temp_output_dir),
            "--plot-format",
            "invalid",
        ],
    )

    assert result.exit_code == 2  # Typer exit code for invalid parameter
    assert "Plot format must be one of" in result.stderr


def test_analyze_wfe_invalid_nzernike(
    sample_wfe_file: Path, temp_output_dir: Path
) -> None:
    """Test error handling for invalid number of Zernike polynomials."""
    result = runner.invoke(
        app, ["analyze", str(sample_wfe_file), str(temp_output_dir), "--nzernike", "0"]
    )

    assert result.exit_code == 2  # Typer exit code for invalid parameter
    assert "is not in the range" in result.stderr


def test_help() -> None:
    """Test help output."""
    # Test root help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Wavefront Error Analysis Tools" in result.stdout
    assert "stop-utils" in result.stdout

    # Test command help
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "Analyze WFE data" in result.stdout
