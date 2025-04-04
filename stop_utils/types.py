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

    coefficients: npt.NDArray[np.float64]
    zernikes: npt.NDArray[np.float64]
    pttf: npt.NDArray[np.float64]
    model: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]

    def rms_error(self) -> float:
        """Calculate RMS of residual error."""
        return float(np.ma.std(self.residual))

    def peak_to_valley(self) -> float:
        """Calculate peak-to-valley of residual error."""
        return float(np.ma.ptp(self.residual))


@dataclass
class AnalysisConfig:
    """Configuration for WFE analysis."""

    n_polynomials: int
    save_coeffs: bool
    generate_plots: bool
    plot_format: str
    output_dir: Path
