"""Visualization utilities for WFE analysis."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from photutils.aperture import EllipticalAperture

from . import logger
from .types import EllipticalParams, WFEResult

# Set logger context for visualization module
logger = logger.bind(context="visualization")


def plotlim(s: int, zoom: int) -> Tuple[int, int]:
    """Calculate plot limits for zoomed view.

    Args:
        s: Size of the dimension
        zoom: Zoom factor

    Returns:
        tuple: (min_limit, max_limit)
    """
    center = s // 2
    zoomed_range = (
        s // zoom // 2
    )  # Calculate zoomed_range based on zoom factor to match test expectations
    return (center - zoomed_range, center + zoomed_range + (s % 2))


def setup_wfe_plot(
    data: npt.NDArray[np.float64], title: str, zoom: int = 4, cmap: str = "jet"
) -> Tuple[plt.Figure, plt.Axes]:
    """Set up a basic WFE plot with common formatting.

    Args:
        data: 2D array to plot
        title: Plot title
        zoom: Zoom factor for display
        cmap: Matplotlib colormap name

    Returns:
        tuple: (Figure object, Axes object)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, origin="lower", cmap=cmap)
    plt.colorbar(im, ax=ax)

    # Set plot limits for proper zoom
    shape = data.shape
    xlims = plotlim(shape[1], zoom)
    ylims = plotlim(shape[0], zoom)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_title(title)
    plt.tight_layout()

    return fig, ax


def plot_wfe_data(
    data: npt.NDArray[np.float64],
    aperture: Optional[EllipticalAperture] = None,
    title: str = "Wavefront Error Map",
    output_path: Optional[Path] = None,
    zoom: int = 4,
    cmap: str = "gray_r",
) -> None:
    """Plot WFE data with optional aperture overlay.

    Args:
        data: WFE data array
        aperture: Optional EllipticalAperture to overlay
        title: Plot title
        output_path: Optional path to save plot
        zoom: Zoom factor for display
        cmap: Matplotlib colormap name
    """
    fig, ax = setup_wfe_plot(data, title, zoom=zoom, cmap=cmap)
    logger.debug(f"Created plot with zoom={zoom}, cmap={cmap}")

    if aperture is not None:
        aperture.plot(ax=ax, color="red", lw=2)
        logger.debug("Added aperture overlay")

    if output_path is not None:
        logger.debug(f"Saving plot to {output_path}")
        plt.savefig(output_path)
        plt.close(fig)
    else:
        logger.debug("Displaying plot")
        plt.show()


def generate_plots(
    result: WFEResult,
    params: EllipticalParams,
    output_dir: Path,
    format: str = "png",
    zoom: int = 1,
) -> None:
    """Generate and save all analysis plots.

    Args:
        result: WFEResult object containing analysis results
        params: EllipticalParams for the aperture
        output_dir: Directory to save plots
        format: Output format (png, pdf, or svg)
        zoom: Zoom factor for display
    """
    logger.info(f"Generating plots in {output_dir} with format={format}")
    output_dir.mkdir(parents=True, exist_ok=True)
    aperture = EllipticalAperture(
        (params.x0, params.y0), params.a, params.b, theta=params.theta
    )
    logger.debug(
        f"Created aperture: center=({params.x0:.1f}, {params.y0:.1f}), "
        f"axes=({params.a:.1f}, {params.b:.1f}), theta={params.theta:.3f}"
    )

    # Determine zoom factor based on the shape of the map and the size of the semi-axes
    map_height, map_width = result.model.shape
    aperture_radius = max(params.a, params.b)
    zoom_x = map_width / (2.0001 * aperture_radius)
    zoom_y = map_height / (2.0001 * aperture_radius)
    zoom = max(1, int(min(zoom_x, zoom_y)))

    logger.info(f"Calculated zoom factor: {zoom}")

    # Raw WFE map
    plot_wfe_data(
        result.raw,
        aperture=aperture,
        title="Raw Wavefront Error",
        output_path=output_dir / f"wfe_raw.{format}",
        zoom=zoom,
    )

    # PTTF component
    plot_wfe_data(
        result.pttf,
        aperture=aperture,
        title="PTTF Component",
        output_path=output_dir / f"wfe_pttf.{format}",
        zoom=zoom,
    )

    # Model fit
    plot_wfe_data(
        result.model,
        aperture=aperture,
        title="Orthonormal Polynomial Model Fit",
        output_path=output_dir / f"wfe_model.{format}",
        zoom=zoom,
    )

    # Residual error
    plot_wfe_data(
        result.residual,
        aperture=aperture,
        title=f"Residual Error (RMS: {result.rms(result.residual):.2f} nm)",
        output_path=output_dir / f"wfe_residual.{format}",
        zoom=zoom,
    )

    # Coefficient plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(result.coefficients)), result.coefficients)
    plt.xlabel("Orthonormal Polynomial Mode")
    plt.ylabel("Coefficient (nm)")
    plt.title("Orthonormal Polynomial Coefficients")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    coeff_plot = output_dir / f"polynomial_coefficients.{format}"
    logger.debug(f"Saving coefficient plot to {coeff_plot}")
    plt.savefig(coeff_plot)
    plt.close()

    logger.info("All plots generated successfully")
