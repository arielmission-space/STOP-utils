# wavefront_extractor.py
import os

import matplotlib.pyplot as plt
import numpy as np

from stop_utils import logger
from stop_utils.zemax.zmx_boilerplate import PythonStandaloneApplication


def process_single_file(
    zemax_file_path,
    base_folder,
    output_dir="WavefrontOutputs",
    surface_name="EXPP",
    wavelength_um=0.633,
):
    """
    Process a single Zemax file and extract wavefront data.

    Args:
        zemax_file_path (str): Path to the Zemax file
        base_folder (str): Base folder for output
        output_dir (str): Directory for output files
        surface_name (str): Name of the surface to analyze
        wavelength_um (float): Wavelength in micrometers to use
    """
    zos = PythonStandaloneApplication()

    try:
        # load local variables
        ZOSAPI = zos.ZOSAPI
        TheApplication = zos.TheApplication
        TheSystem = zos.TheSystem

        zemax_filename = os.path.splitext(os.path.basename(zemax_file_path))[0]

        # === Load the file ===
        zos.OpenFile(zemax_file_path, False)

        # === Setup the wavelength ===
        # Get the wavelength data
        wavelength_data = TheSystem.SystemData.Wavelengths

        # Process wavelength selection
        if wavelength_um is not None:
            logger.info(f"Using custom wavelength of {wavelength_um} micron")
            wavelength_data.GetWavelength(1).Wavelength = wavelength_um
        else:
            wavelength_um = wavelength_data.GetWavelength(1).Wavelength

        # === Setup Wavefront Map Analysis ===
        analysis = TheSystem.Analyses.New_Analysis(
            ZOSAPI.Analysis.AnalysisIDM.WavefrontMap
        )

        # === Now set the surface number
        lens_data = TheSystem.LDE
        surface_found = False
        surface_number = -1

        for i in range(1, lens_data.NumberOfSurfaces):
            surface = lens_data.GetSurfaceAt(i)
            comment = surface.Comment
            if comment.upper() == surface_name:
                surface_number = i
                surface_found = True
                logger.info(
                    f"Found surface with comment '{comment}' at surface number {surface_number}"
                )
                break

        if not surface_found:
            logger.error(f"Surface '{surface_name}' not found in {zemax_filename}")
            zos.CloseFile(False)
            return

        # Explicitly cast to IAS_WavefrontMap settings
        settings = analysis.GetSettings()
        wavefront_settings = ZOSAPI.Analysis.Settings.IAS_WavefrontMap(settings)

        # Now set the surface number on the properly cast object
        wavefront_settings.Surface.SetSurfaceNumber(surface_number)

        # Set the grid resolution to 512x512
        wavefront_settings.Sampling = ZOSAPI.Analysis.SampleSizes.S_512x512

        # === Apply the settings and run the analysis
        analysis.ApplyAndWaitForCompletion()

        # === Get results
        results = analysis.GetResults()
        data_grid = results.GetDataGrid(0)

        # Use get_Values() to retrieve the actual grid data
        grid_data = data_grid.get_Values()

        # For reshaping and processing the data
        x_size = grid_data.GetLength(0)  # X dimension
        y_size = grid_data.GetLength(1)  # Y dimension

        # Now, you can reshape the data if needed
        reshaped_data = zos.reshape(grid_data, x_size, y_size)
        wfe_map = np.asarray(reshaped_data)

        # === Save to txt file
        results.GetTextFile(f"{output_dir}\\{zemax_filename}.txt")

        # === Optional: create and save visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(
            wfe_map * wavelength_um * 1e3,
            cmap="Greys",
            origin="lower",
            interpolation="none",
        )
        plt.colorbar(label="Wavefront [nm]")
        title = "\n".join(
            [f"{zemax_filename}", f"Wavefront Map at Surface {surface_name}"]
        )
        plt.title(title)
        plt.xlabel("X Index")
        plt.ylabel("Y Index")

        # Save the plot
        plot_path = os.path.join(
            base_folder,
            output_dir,
            f"{zemax_filename} - {surface_name} - WFE.png",
        )
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free memory

        logger.info(f"Processed {zemax_filename} successfully")

    finally:
        # Always close the file and clean up
        if "TheSystem" in locals():
            zos.CloseFile(False)

        # Clean up the connection to OpticStudio
        del zos


if __name__ == "__main__":
    base_folder = (
        r"C:\Users\abocc\OneDrive - uniroma1.it\Andrea\work\Sap\Projects\zemax"
    )
    sim_config = "FC"
    case_number = "17"
    zemax_filename = f"ARIEL - STOP Analysis - {sim_config} - C{case_number}"
    zemax_file_path = rf"{base_folder}\{zemax_filename}.zmx"

    process_single_file(
        zemax_file_path=zemax_file_path,
        base_folder=base_folder,
        output_dir="WavefrontOutputs",
        surface_name="EXPP",
        wavelength_um=0.633,
    )
