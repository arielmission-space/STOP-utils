"""Type definitions for stop-utils package."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass
class EllipticalParams:
    """Parameters defining an elliptical aperture."""

    x0: float
    y0: float
    a: float
    b: float
    theta: float


@dataclass
class WFEResult:
    """Results from WFE analysis."""

    raw: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]
    zernikes: npt.NDArray[np.float64]
    pttf: npt.NDArray[np.float64]
    model: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]

    @staticmethod
    def rms(errormap: npt.NDArray[np.float64]) -> float:
        """Calculate RMS of the WFE errormap."""
        return float(np.ma.std(errormap))

    @staticmethod
    def rss(coefficients: npt.NDArray[np.float64]) -> float:
        """Calculate RSS of coefficients."""
        return float(np.sqrt(np.sum(coefficients**2)))

    @staticmethod
    def ptp(errormap: npt.NDArray[np.float64]) -> float:
        """Calculate Peak to Valley of the WFE errormap."""
        return float(np.ma.ptp(errormap))


@dataclass
class AnalysisConfig:
    """Configuration for WFE analysis."""

    n_polynomials: int
    save_coeffs: bool
    generate_plots: bool
    plot_format: str
    output_dir: Path
