"""Command-line interface for WFE analysis."""

import json
from pathlib import Path
from typing import List, Optional

import typer

from . import LOG_FORMAT, __version__, logger, project
from .types import AnalysisConfig, EllipticalParams
from .visualization import generate_plots
from .wfe_analysis import analyze_wfe_data
from .zemax.zmx_batch_processor import batch_process_zmx

# Set logger context for CLI
logger = logger.bind(context="cli")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        logger.info(f"stop-utils version {__version__}")
        raise typer.Exit()


def print_coeffs(coefficients: List[float], zernikes: List[float]) -> None:
    """Logs polynomial coefficients line by line."""

    logger.info("Polynomial coefficients:")
    logger.info("ordering=standard, normalize=False")

    logger.info("Mode | Orthon. (nm) | Zernike (nm)")
    logger.info("----------------------------------")

    if len(coefficients) != len(zernikes):
        logger.warning(
            f"Mismatch in list lengths: {len(coefficients)} coefficients vs {len(zernikes)} Zernikes. "
            "Logging pairs up to the shorter list."
        )
        # zip will automatically stop at the shorter list, but the warning is good practice

    # Log each coefficient pair on its own line
    for i, (coeff, zern) in enumerate(zip(coefficients, zernikes)):
        # Format each line clearly identifying the mode and values
        logger.info(f"  {i:>2} | {coeff:>12.3f} | {zern:>12.3f}")
        # Adjust spacing (e.g., :>2, :>8.3f) as needed for alignment based on expected number ranges


def save_coefficients(
    output_dir: Path,
    orthonormal_coefficients: List[float],
    zernike_coefficients: List[float],
    params: EllipticalParams,
) -> None:
    """Save polynomial coefficients and ellipse parameters to JSON file."""
    coeff_file = output_dir / "polynomial_coefficients.json"
    with open(coeff_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "orthonormal_coefficients": [
                    float(c) for c in orthonormal_coefficients
                ],
                "zernike_coefficients": [float(c) for c in zernike_coefficients],
                "polynomials_definition": {
                    "ordering": "standard",
                    "normalize": False,
                },
                "units": {
                    "coefficients": "nm",
                    "center": "pixel",
                    "semi_axes": "pixel",
                    "angle": "rad",
                },
                "ellipse_parameters": {
                    "center": {"x": float(params.x0), "y": float(params.y0)},
                    "semi_axes": {"a": float(params.a), "b": float(params.b)},
                    "angle": float(params.theta),
                },
            },
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
    n_polynomials: int,
    plot_format: str,
    save_coeffs: bool,
    no_plots: bool,
    file_format: Optional[str] = None,
) -> None:
    """Run WFE analysis with given parameters."""
    try:
        # Create output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error(f"Failed to create output directory: {exc}")
            raise typer.Exit(1) from exc

        # Create configuration
        config = AnalysisConfig(
            n_polynomials=n_polynomials,
            save_coeffs=save_coeffs,
            generate_plots=not no_plots,
            plot_format=plot_format,
            output_dir=output_dir,
        )

        logger.log("ANNOUNCE", "Analysis started.")
        logger.info(f"Running {project} v{__version__}")

        # Analyze WFE data
        try:
            result, params = analyze_wfe_data(
                wfe_file=input_file,
                n_polynomials=config.n_polynomials,
                file_format=file_format,
            )
        except FileNotFoundError as exc:
            logger.error(f"Input file not found: {input_file}")
            raise typer.Exit(1) from exc
        except Exception as exc:
            logger.error(f"Analysis failed: {exc}")
            raise typer.Exit(1) from exc

        # Save coefficients if requested
        if config.save_coeffs:
            try:
                coeff_list = [float(c) for c in result.coefficients]
                zernikes_list = [float(c) for c in result.zernikes]
                save_coefficients(config.output_dir, coeff_list, zernikes_list, params)
            except Exception as exc:
                logger.error(f"Failed to save coefficients: {exc}")
                raise typer.Exit(1) from exc

        # Generate plots if requested
        if config.generate_plots:
            try:
                generate_plots(
                    result=result,
                    params=params,
                    output_dir=config.output_dir,
                    format=config.plot_format,
                )
            except Exception as exc:
                logger.error(f"Failed to generate plots: {exc}")
                raise typer.Exit(1) from exc

        logger.success("Analysis completed successfully")

        # Log computed coefficients
        coeff_list = [float(c) for c in result.coefficients]
        zernikes_list = [float(c) for c in result.zernikes]
        print_coeffs(coeff_list, zernikes_list)

        # Log global results
        logger.info("Global results:")

        logger.info(f"Raw WFE RMS: {result.rms(result.raw):.2f} nm")
        logger.info(f"Fit WFE RMS: {result.rms(result.model):.2f} nm")
        logger.info(f"Fit WFE RMS (-PTT): {result.rss(result.coefficients[3:]):.2f} nm")
        logger.info(
            f"Fit WFE RMS (-PTTF): {result.rss(result.coefficients[4:]):.2f} nm"
        )
        logger.info(f"Residual RMS: {result.rms(result.residual):.2f} nm")
        logger.info(f"Residual PTP: {result.ptp(result.residual):.2f} nm")

        # Ellipse fit bookkeeping
        logger.info("Ellipse parameters:")
        logger.info(f"Center: ({params.x0:.1f}, {params.y0:.1f}) pixel")
        logger.info(f"Semi-axes: ({params.a:.1f}, {params.b:.1f}) pixel")
        logger.info(f"Angle: {params.theta:.3f} rad")

        if config.generate_plots:
            logger.info(f"Plots saved to: {output_dir}/")
        if config.save_coeffs:
            logger.info(
                f"Coefficients saved to: {output_dir}/polynomial_coefficients.json"
            )

    except typer.Exit:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        raise typer.Exit(1) from exc

    logger.log("ANNOUNCE", "Analysis ended.")


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
    (orthonormal) polynomial decomposition on elliptical apertures and generating
    visualization outputs.
    """
    pass


@app.command()
def analyze(
    input_file: Path = typer.Option(
        ...,
        "--input-file",
        "-i",
        show_default=False,
        help="Input WFE data file",
        rich_help_panel="Required",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        show_default=False,
        help="Output directory for results",
        rich_help_panel="Required",
    ),
    n_polynomials: int = typer.Option(
        15,
        "--npolynomials",
        "-n",
        min=4,
        help="Number of polynomials",
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
        help="Save polynomial coefficients to JSON",
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Skip plot generation",
    ),
    file_format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Specify the file format (e.g., 'zemax'). "
        + "If not provided, expects a simple .dat file.",
    ),
) -> None:
    """Analyze WFE data and generate results."""
    # Add loguru file handler for debug.log in output_dir
    logger.add(
        output_dir / "debug.log",
        format=LOG_FORMAT,
        level="DEBUG",
        rotation="1 week",
        retention="1 month",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    run_analysis(
        input_file=input_file,
        output_dir=output_dir,
        n_polynomials=n_polynomials,
        plot_format=plot_format,
        save_coeffs=save_coeffs,
        no_plots=no_plots,
        file_format=file_format,
    )


@app.command("zmx-batch")
def zmx_batch(
    base_folder: str = typer.Option(
        ...,
        "--base-folder",
        "-b",
        help="Directory containing ZMX files.",
        show_default=False,
        rich_help_panel="Required",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        show_default=False,
        help="Output directory for results",
        rich_help_panel="Required",
    ),
    surface_name: str = typer.Option(
        "EXPP",
        "--surface-name",
        "-s",
        help="Surface name to look for.",
        show_default=True,
    ),
    wavelength_um: Optional[float] = typer.Option(
        None,
        "--wavelength-um",
        "-w",
        help="Custom wavelength in micrometers to use.",
        show_default=False,
    ),
) -> None:
    """
    Process all ZMX files in the specified directory.
    """
    # Add loguru file handler for debug.log in output_dir
    logger.add(
        str(Path(output_dir) / "debug.log"),
        format=LOG_FORMAT,
        level="DEBUG",
        rotation="1 week",
        retention="1 month",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    batch_process_zmx(
        base_folder=base_folder,
        output_dir=str(output_dir),
        surface_name=surface_name,
        wavelength_um=wavelength_um,
    )


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
