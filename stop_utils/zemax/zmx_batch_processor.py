import glob
import os
import sys
from pathlib import Path
from typing import Optional

import typer


def batch_process_zmx(
    base_folder: str,
    output_dir: str = "WavefrontOutputs",
    surface_name: str = "EXPP",
    wavelength_um: Optional[float] = None,
):
    """
    Process all ZMX files in the specified directory.
    """
    typer.echo(
        typer.style(
            f"[bold cyan]Searching for .zmx files in:[/] {base_folder}", bold=True
        )
    )
    zmx_files = glob.glob(os.path.join(base_folder, "*.zmx"))

    if not zmx_files:
        typer.secho(
            f"No .zmx files found in {base_folder}", fg=typer.colors.RED, bold=True
        )
        raise typer.Exit(code=1)

    typer.secho(
        f"Found {len(zmx_files)} .zmx files to process.",
        fg=typer.colors.GREEN,
        bold=True,
    )

    output_path = os.path.join(base_folder, output_dir)
    os.makedirs(output_path, exist_ok=True)

    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from wavefront_extractor import process_single_file
    except ImportError as exc:
        typer.secho(
            "Error: Could not import the wavefront_extractor module.",
            fg=typer.colors.RED,
            bold=True,
        )
        typer.echo(
            "Make sure wavefront_extractor.py is in the same directory as this script."
        )
        raise typer.Exit(code=1) from exc

    for i, zmx_file_path in enumerate(zmx_files):
        file_name = Path(zmx_file_path).stem
        typer.echo(
            typer.style(
                f"\n[bold yellow]Processing file {i+1}/{len(zmx_files)}:[/] {file_name}",
                bold=True,
            )
        )
        try:
            process_single_file(
                zemax_file_path=zmx_file_path,
                base_folder=base_folder,
                output_dir=output_dir,
                surface_name=surface_name,
                wavelength_um=wavelength_um,
            )
        except Exception as e:
            typer.secho(f"Error processing {file_name}: {str(e)}", fg=typer.colors.RED)
            continue

    typer.secho("\nAll files processed.", fg=typer.colors.GREEN, bold=True)
