"""Wavefront Error Analysis Utilities."""
import importlib.metadata as metadata
import os
import sys
from datetime import date
from pathlib import Path

from loguru import logger

from .converters import load_zemax_wfe
from .types import AnalysisConfig, WFEResult
from .visualization import generate_plots
from .wfe_analysis import analyze_wfe_data

__all__ = [
    "WFEResult",
    "AnalysisConfig",
    "analyze_wfe_data",
    "generate_plots",
    "logger",
    "load_zemax_wfe",
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
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<10}</level> | "
    "<cyan>{name:<30}</cyan>:<cyan>{function:<30}</cyan>:<cyan>{line:<10}</cyan>"
    "<level>{message}</level>"
)

# Add announce logger level
logger.level("ANNOUNCE", no=100, color="<magenta>")
# Remove default handler and add custom handlers
logger.remove()

# Add console handler with custom format
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level="INFO",
    colorize=True,
)
