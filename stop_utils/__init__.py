"""Wavefront Error Analysis Utilities."""
import importlib.metadata as metadata
import os
import sys
from datetime import date
from pathlib import Path

from loguru import logger

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

project = "stop-utils"

# load package info
__pkg_name__ = __title__ = metadata.metadata(project)["Name"].upper()
__version__ = metadata.version(project)
__url__ = metadata.metadata(project)["Project-URL"]
__author__ = metadata.metadata(project)["Author"]
__license__ = metadata.metadata(project)["License"]
__copyright__ = f"2025-{date.today().year:d}, {__author__}"
__summary__ = metadata.metadata(project)["Summary"]


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
