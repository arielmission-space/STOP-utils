import re
from pathlib import Path
from typing import Union

import numpy as np

from . import logger


def load_zemax_wfe(file_path: Union[str, Path]) -> np.ndarray:
    """
    Loads Wavefront Error (WFE) data from a Zemax text file.

    Parses the header to find wavelength and grid size, then reads the
    data grid and converts it from waves to nanometers.

    Args:
        file_path: path to the Zemax WFE map file (.txt). Can be a string or Path object.

    Returns:
        A 2D numpy array containing the WFE map in nanometers.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is incorrect or metadata is missing.
    """
    # Convert to Path if string is provided
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    wavelength_um = None
    grid_size = None
    data_lines = []
    header_lines_count = 0
    reading_data = False

    logger.info(f"Reading Zemax WFE file: {file_path}")

    with open(
        file_path, "r", encoding="utf-16le"
    ) as f:  # Zemax files often use utf-16le
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if not reading_data:
                header_lines_count += 1
                # Extract wavelength
                match_wl = re.search(r"(\d+\.\d+)\s+µm", line)
                if match_wl:
                    wavelength_um = float(match_wl.group(1))
                    logger.debug(f"Found wavelength: {wavelength_um} µm")

                # Extract grid size
                match_grid = re.search(r"Pupil grid size:\s+(\d+)\s+by\s+(\d+)", line)
                if match_grid:
                    grid_size = (int(match_grid.group(1)), int(match_grid.group(2)))
                    logger.debug(f"Found grid size: {grid_size}")

                # Check if the line looks like the start of data
                # Data lines typically start with a float in scientific notation
                if re.match(r"^\s*-?\d+\.\d+E[+-]\d+", line):
                    reading_data = True
                    logger.debug(f"Detected start of data at line {i + 1}")
                    # The current line is the first data line
                    data_lines.append(line)
            else:
                # Continue reading data lines
                if re.match(r"^\s*-?\d+\.\d+E[+-]\d+", line):
                    data_lines.append(line)
                else:
                    # Stop if a non-data line is encountered after data started
                    logger.debug(f"Detected end of data at line {i}")
                    break

    if wavelength_um is None:
        raise ValueError(f"Could not find wavelength in header of {file_path}")
    if grid_size is None:
        raise ValueError(f"Could not find grid size in header of {file_path}")
    if not data_lines:
        raise ValueError(f"Could not find data section in {file_path}")

    # Process the collected data lines
    try:
        # Create a single string with newline separators for loadtxt
        data_string = "\\n".join(data_lines)
        # Use np.loadtxt, specifying the correct number of header lines to skip
        # We load the data as a flat array first
        wfe_data_waves = np.fromstring(data_string.replace("\\n", " "), sep=" ")

        # Reshape the data according to the grid size found in the header
        wfe_data_waves = wfe_data_waves.reshape(grid_size)
        # flip the data upside down to match the expected orientation
        wfe_data_waves = np.flipud(wfe_data_waves)

    except Exception as e:
        raise ValueError(f"Error parsing data in {file_path}: {e}") from e

    # Convert from waves to nm
    wfe_data_nm = wfe_data_waves * wavelength_um * 1000.0

    logger.info(f"Successfully loaded and converted data from {file_path}")
    logger.debug(
        f"WFE data shape: {wfe_data_nm.shape}, Min: {wfe_data_nm.min():.3f} nm, Max: {wfe_data_nm.max():.3f} nm, Mean: {wfe_data_nm.mean():.3f} nm"
    )

    return wfe_data_nm
