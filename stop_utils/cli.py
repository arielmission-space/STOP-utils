"""Command-line interface for WFE analysis."""

import json
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__, logger
from .types import AnalysisConfig
from .visualization import generate_plots
from .wfe_analysis import analyze_wfe_data

# Set logger context for CLI
logger = logger.bind(context="cli")

def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        logger.info(f"stop-utils version {__version__}")
        raise typer.Exit()

# Create console for output
console = Console()

def create_coefficients_table(coefficients: List[float]) -> Table:
    """Create a rich table displaying Zernike orthonormal coefficients."""
    table = Table(title="Zernike Orthonormal Coefficients")
    table.add_column("Mode", justify="center", style="cyan")
    table.add_column("Coefficient (nm)", justify="right", style="green")
    
    for i, coeff in enumerate(coefficients):
        table.add_row(str(i), f"{coeff:.3f}")
    
    return table

def save_coefficients(output_dir: Path, coefficients: List[float]) -> None:
    """Save Zernike orthonormal coefficients to JSON file."""
    coeff_file = output_dir / "zernike_orthonormal_coefficients.json"
    with open(coeff_file, "w") as f:
        json.dump(
            {"coefficients": [float(c) for c in coefficients], "units": "nm"},
            f,
            indent=2,
        )

def validate_plot_format(value: str) -> str:
    """Validate plot format option."""
    if value not in ["png", "pdf", "svg"]:
        raise typer.BadParameter("Plot format must be one of: png, pdf, svg")
    return value

def run_analysis(
    input_file: Path,
    output_dir: Path,
    n_zernike: int,
    plot_format: str,
    save_coeffs: bool,
    no_plots: bool,
) -> None:
    """Run WFE analysis with given parameters."""
    try:
        # Create output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise typer.Exit(1)

        # Create configuration
        config = AnalysisConfig(
            n_zernike=n_zernike,
            save_coeffs=save_coeffs,
            generate_plots=not no_plots,
            plot_format=plot_format,
            output_dir=output_dir,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn(" {task.description:<12}"),  # Left-aligned, fixed width, no dots
            console=console,
            transient=False,  # Keep tasks visible for better feedback
        ) as progress:
            # Analyze WFE data
            try:
                task_id = progress.add_task("Analyzing", total=None)
                result, params = analyze_wfe_data(
                    wfe_file=input_file, n_zernike=config.n_zernike
                )
                progress.remove_task(task_id)
            except FileNotFoundError:
                logger.error(f"Input file not found: {input_file}")
                raise typer.Exit(1)
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                raise typer.Exit(1)

            # Save coefficients if requested
            if config.save_coeffs:
                try:
                    task_id = progress.add_task("Saving", total=None)
                    coeff_list = [float(c) for c in result.coefficients]
                    save_coefficients(config.output_dir, coeff_list)
                    progress.remove_task(task_id)
                except Exception as e:
                    logger.error(f"Failed to save coefficients: {e}")
                    raise typer.Exit(1)

            # Generate plots if requested
            if config.generate_plots:
                try:
                    task_id = progress.add_task("Plotting", total=None)
                    generate_plots(
                        result=result,
                        params=params,
                        output_dir=config.output_dir,
                        format=config.plot_format,
                    )
                    progress.remove_task(task_id)
                except Exception as e:
                    logger.error(f"Failed to generate plots: {e}")
                    raise typer.Exit(1)

        # Display results summary (order matters for test compatibility)
        console.print("\n[green]Analysis complete![/green]")
        logger.success("Analysis completed successfully")

        # Display coefficients table
        coeff_list = [float(c) for c in result.coefficients]
        console.print(create_coefficients_table(coeff_list))
        
        # Log metrics and output locations
        logger.info(
            "Results:\n"
            f"  RMS residual error: {result.rms_error():.2f} nm\n"
            f"  PTP residual error: {result.peak_to_valley():.2f} nm"
        )
        
        if config.generate_plots or config.save_coeffs:
            outputs = []
            if config.generate_plots:
                outputs.append(f"  Plots: {output_dir}")
            if config.save_coeffs:
                outputs.append(f"  Coefficients: {output_dir}/zernike_orthonormal_coefficients.json")
            logger.info("Output files:\n" + "\n".join(outputs))

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise typer.Exit(1)

# Create the Typer app
app = typer.Typer(
    name="stop-utils",
    help="Wavefront Error Analysis Tools - Analyze and visualize wavefront error data",
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

@app.callback()
def callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    )
) -> None:
    """
    Wavefront Error Analysis Tools - Analyze and visualize wavefront error data.

    This tool provides functionality for analyzing wavefront error data using
    Zernike orthornormal polynomial decomposition and generating visualization outputs.
    """
    pass

@app.command()
def analyze(
    input_file: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, help="Input WFE data file"
    ),
    output_dir: Path = typer.Argument(
        ..., file_okay=False, dir_okay=True, help="Output directory for results"
    ),
    n_zernike: int = typer.Option(
        15, "--nzernike", "-n", min=1, help="Number of Zernike orthonormal polynomials"
    ),
    plot_format: str = typer.Option(
        "png",
        "--plot-format",
        "-f",
        callback=validate_plot_format,
        help="Plot output format (png, pdf, svg)",
    ),
    save_coeffs: bool = typer.Option(
        True,
        "--save-coeffs/--no-save-coeffs",
        help="Save Zernike orthonormal coefficients to JSON",
    ),
    no_plots: bool = typer.Option(False, help="Skip plot generation"),
) -> None:
    """Analyze WFE data and generate results."""
    run_analysis(
        input_file=input_file,
        output_dir=output_dir,
        n_zernike=n_zernike,
        plot_format=plot_format,
        save_coeffs=save_coeffs,
        no_plots=no_plots,
    )

def main() -> None:
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
