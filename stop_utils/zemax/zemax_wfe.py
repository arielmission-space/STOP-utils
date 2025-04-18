import clr, os, winreg
from itertools import islice

# This boilerplate requires the 'pythonnet' module.
# The following instructions are for installing the 'pythonnet' module via pip:
#    1. Ensure you are running a Python version compatible with PythonNET. Check the article "ZOS-API using Python.NET" or
#    "Getting started with Python" in our knowledge base for more details.
#    2. Install 'pythonnet' from pip via a command prompt (type 'cmd' from the start menu or press Windows + R and type 'cmd' then enter)
#
#        python -m pip install pythonnet


class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self, path=None):
        # determine location of ZOSAPI_NetHelper.dll & add as reference
        aKey = winreg.OpenKey(
            winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER),
            r"Software\Zemax",
            0,
            winreg.KEY_READ,
        )
        zemaxData = winreg.QueryValueEx(aKey, "ZemaxRoot")
        NetHelper = os.path.join(
            os.sep, zemaxData[0], r"ZOS-API\Libraries\ZOSAPI_NetHelper.dll"
        )
        winreg.CloseKey(aKey)
        clr.AddReference(NetHelper)
        import ZOSAPI_NetHelper

        # Find the installed version of OpticStudio
        # if len(path) == 0:
        if path is None:
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize()
        else:
            # Note -- uncomment the following line to use a custom initialization path
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(path)

        # determine the ZOS root directory
        if isInitialized:
            dir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
        else:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to locate Zemax OpticStudio.  Try using a hard-coded path."
            )

        # add ZOS-API referencecs
        clr.AddReference(os.path.join(os.sep, dir, "ZOSAPI.dll"))
        clr.AddReference(os.path.join(os.sep, dir, "ZOSAPI_Interfaces.dll"))
        import ZOSAPI

        # create a reference to the API namespace
        self.ZOSAPI = ZOSAPI

        # create a reference to the API namespace
        self.ZOSAPI = ZOSAPI

        # Create the initial connection class
        self.TheConnection = ZOSAPI.ZOSAPI_Connection()

        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException(
                "Unable to initialize .NET connection to ZOSAPI"
            )

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application"
            )

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException(
                "License is not valid for ZOSAPI use"
            )

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system"
            )

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system"
            )
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException(
                "Unable to acquire Primary system"
            )
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException(
                "Unable to acquire ZOSAPI application"
            )

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if (
            self.TheApplication.LicenseStatus
            == self.ZOSAPI.LicenseStatusType.PremiumEdition
        ):
            return "Premium"
        elif (
            self.TheApplication.LicenseStatus
            == self.ZOSAPI.LicenseStatusType.EnterpriseEdition
        ):
            return "Enterprise"
        elif (
            self.TheApplication.LicenseStatus
            == self.ZOSAPI.LicenseStatusType.ProfessionalEdition
        ):
            return "Professional"
        elif (
            self.TheApplication.LicenseStatus
            == self.ZOSAPI.LicenseStatusType.StandardEdition
        ):
            return "Standard"
        elif (
            self.TheApplication.LicenseStatus
            == self.ZOSAPI.LicenseStatusType.OpticStudioHPCEdition
        ):
            return "HPC"
        else:
            return "Invalid"

    def reshape(self, data, x, y, transpose=False):
        """Converts a System.Double[,] to a 2D list for plotting or post processing

        Parameters
        ----------
        data      : System.Double[,] data directly from ZOS-API
        x         : x width of new 2D list [use var.GetLength(0) for dimension]
        y         : y width of new 2D list [use var.GetLength(1) for dimension]
        transpose : transposes data; needed for some multi-dimensional line series data

        Returns
        -------
        res       : 2D list; can be directly used with Matplotlib or converted to
                    a numpy array using numpy.asarray(res)
        """
        if type(data) is not list:
            data = list(data)
        var_lst = [y] * x
        it = iter(data)
        res = [list(islice(it, i)) for i in var_lst]
        if transpose:
            return self.transpose(res)
        return res

    def transpose(self, data):
        """Transposes a 2D list (Python3.x or greater).

        Useful for converting mutli-dimensional line series (i.e. FFT PSF)

        Parameters
        ----------
        data      : Python native list (if using System.Data[,] object reshape first)

        Returns
        -------
        res       : transposed 2D list
        """
        if type(data) is not list:
            data = list(data)
        return list(map(list, zip(*data)))


import numpy as np
from datetime import datetime


def save_wavefront_map_txt(
    filepath,
    wavefront_data,
    wavelength_um,
    field_x,
    field_y,
    peak_to_valley,
    rms,
    surface_number,
    surface_name,
    exit_pupil_diameter,
):
    # Get grid size
    Ny, Nx = wavefront_data.shape
    center_col = (Nx + 1) // 2
    center_row = (Ny + 1) // 2

    # Header
    lines = []
    lines.append("Listing of Wavefront Map Data\n")
    lines.append(f"File : {filepath}")
    lines.append("Title: ")
    lines.append(f"Date : {datetime.now().strftime('%d/%m/%Y')}\n\n")

    # Wavefront info
    lines.append("Wavefront Function")
    lines.append(f"{wavelength_um:.6f} µm at {field_x:.2f}, {field_y:.2f} (deg)")
    lines.append(f"Peak to valley = {peak_to_valley:.3f} waves, RMS = {rms:.3f} waves.")
    lines.append(f"Surface: {surface_number} ({surface_name})")
    lines.append(f"Exit Pupil Diameter: {exit_pupil_diameter:.6f} Millimeters\n")
    lines.append(f"Pupil grid size: {Nx} by {Ny}")
    lines.append(f"Center point is: Col {center_col}, Row {center_row}\n")

    # Format wavefront data
    for row in wavefront_data:
        line = " ".join(f"{val:13.6E}" for val in row)
        lines.append(line)

    # Write to file
    with open(filepath, "w", encoding="utf-16le") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wavefront map saved to {filepath}")


if __name__ == "__main__":
    zos = PythonStandaloneApplication()

    # load local variables
    ZOSAPI = zos.ZOSAPI
    TheApplication = zos.TheApplication
    TheSystem = zos.TheSystem

    # Insert Code Here

    # === USER CONFIGURATION ===
    base_folder = (
        rf"C:\Users\abocc\OneDrive - uniroma1.it\Andrea\work\Sap\Projects\zemax"
    )
    sim_config = "FC"
    case_number = "17"
    zemax_filename = f"ARIEL - STOP Analysis - {sim_config} - C{case_number}"
    zemax_file_path = (
        rf"{base_folder}\{zemax_filename}.zmx"  # Change to your .ZMX or .ZOS file path
    )
    # surface_number = 69  # Surface where you want the wavefront map

    # === Load the file ===
    zos.OpenFile(zemax_file_path, False)

    # === Setup Wavefront Map Analysis ===
    analysis = TheSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.WavefrontMap)

    # === Now set the surface number
    lens_data = TheSystem.LDE
    surface_found = False
    surface_number = -1

    surface_name = "expp"
    for i in range(1, lens_data.NumberOfSurfaces):
        surface = lens_data.GetSurfaceAt(i)
        comment = surface.Comment
        #print(comment)
        if comment.lower() == surface_name:
            surface_number = i
            surface_found = True
            print(
                f"Found surface with comment '{comment}' at surface number {surface_number}"
            )
            break

    # Explicitly cast to IAS_WavefrontMap settings
    settings = analysis.GetSettings()
    wavefront_settings = ZOSAPI.Analysis.Settings.IAS_WavefrontMap(settings)

    # Now set the surface number on the properly cast object
    wavefront_settings.Surface.SetSurfaceNumber(surface_number)

    # Set the grid resolution to 64x64
    wavefront_settings.Sampling = ZOSAPI.Analysis.SampleSizes.S_64x64

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

    # === You can proceed with your visualization code
    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(wfe_map, cmap="Greys", origin="lower", interpolation="none")
    plt.colorbar(label="Wavefront")
    plt.title(f"Wavefront Map at Surface {surface_number}")
    plt.xlabel("X Index")
    plt.ylabel("Y Index")
    plt.show()

    # === Save to txt file
    wavelength_data = TheSystem.SystemData.Wavelengths
    field_data = TheSystem.SystemData.Fields

    # Example: use primary wavelength and first field
    wavelength_um = wavelength_data.GetWavelength(1).Wavelength  # in microns
    field_x = field_data.GetField(1).X  # in degrees
    field_y = field_data.GetField(1).Y  # in degrees

    # Exit pupil diameter
    exit_pupil_diameter = TheSystem.SystemData.Aperture.ApertureValue  # in mm

    # data stats
    wfe_masked = np.ma.MaskedArray(wfe_map, mask=np.isnan(wfe_map))
    ptp = np.ma.ptp(wfe_masked)
    rms = np.ma.std(wfe_masked)

    out_dir = "WavefrontOutputs"

    save_wavefront_map_txt(
        filepath=f"{base_folder}/{out_dir}/{zemax_filename}.txt",
        wavefront_data=wfe_masked.filled(0.0),
        wavelength_um=wavelength_um,
        field_x=field_x,
        field_y=field_y,
        peak_to_valley=ptp,
        rms=rms,
        surface_number=surface_number,
        surface_name=surface_name,
        exit_pupil_diameter=exit_pupil_diameter,
    )

    # Optional: Close the file
    zos.CloseFile(False)

    #

    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zos
    zos = None
