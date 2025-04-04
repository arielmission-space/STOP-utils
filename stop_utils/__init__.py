"""Wavefront Error Analysis Utilities."""

import sys
from pathlib import Path

from loguru import logger

__version__ = "0.1.0"

# Configure loguru logger
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Remove default handler and add custom handlers
logger.remove()

# Add console handler with custom format
logger.add(
    sys.stderr,
    format=log_format,
    level="INFO",
    colorize=True,
)

# Add file handler for debug logs
log_dir = Path.home() / ".stop-utils" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    log_dir / "debug.log",
    format=log_format,
    level="DEBUG",
    rotation="1 week",
    retention="1 month",
    compression="gz",
)

from .types import AnalysisConfig, WFEResult
from .visualization import generate_plots
from .wfe_analysis import analyze_wfe_data

__all__ = [
    "WFEResult",
    "AnalysisConfig",
    "analyze_wfe_data",
    "generate_plots",
    "logger",
]
