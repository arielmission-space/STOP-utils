import glob
import os
from pathlib import Path
from typing import Optional

import typer

from stop_utils import __version__, logger, project
from stop_utils.zemax.wavefront_extractor import process_single_file


def batch_process_zmx(
    base_folder: str,
    output_dir: str = "WavefrontOutputs",
    surface_name: str = "EXPP",
    wavelength_um: Optional[float] = None,
):
    """
    Process all ZMX files in the specified directory.
    """
    logger.log("ANNOUNCE", "Analysis started.")
    logger.info(f"Running {project} v{__version__}")

    logger.info(f"Searching for .zmx files in: {base_folder}")
    zmx_files = glob.glob(os.path.join(base_folder, "*.zmx"))

    if not zmx_files:
        logger.error(f"No .zmx files found in {base_folder}")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(zmx_files)} .zmx files to process.")

    for i, zmx_file_path in enumerate(zmx_files):
        file_name = Path(zmx_file_path).stem
        logger.info(f"Processing file {i+1}/{len(zmx_files)}: {file_name}")
        try:
            process_single_file(
                zemax_file_path=zmx_file_path,
                base_folder=base_folder,
                output_dir=output_dir,
                surface_name=surface_name,
                wavelength_um=wavelength_um,
            )
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            continue

    logger.info("All files processed.")
    logger.log("ANNOUNCE", "Analysis ended.")
