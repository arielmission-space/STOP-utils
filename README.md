# STOP-utils

[![PyPI version](https://badge.fury.io/py/stop-utils.svg?icon=si%3Apython)](https://badge.fury.io/py/stop-utils)
[![GitHub version](https://badge.fury.io/gh/arielmission-space%2FSTOP-utils.svg?icon=si%3Agithub)](https://badge.fury.io/gh/arielmission-space%2FSTOP-utils)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/stop-utils/badge/?version=latest)](https://stop-utils.readthedocs.io/en/latest/?badge=latest)

Utilities for analyzing wavefront error data.

## Overview

This package provides utilities for Wavefront Error (WFE) analysis, implementing orthonormal polynomial decomposition and visualization tools. The core functionality is based on the `PAOS` package (see [readthedocs](https://paos.readthedocs.io/en/latest/)). The package supports both raw WFE data files and Zemax-exported Wavefront Map text files.

## Installation

<details>
<summary><strong>From PyPI</strong></summary>

Once the package is published on PyPI, you can install it using pip:

```bash
pip install stop-utils
```

</details>

<details>
<summary><strong>From GitHub</strong></summary>

You can install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/arielmission-space/STOP-utils.git
```

</details>

<details>
<summary><strong>Development Installation</strong></summary>

First, clone the repository:

```bash
git clone https://github.com/arielmission-space/STOP-utils.git
cd STOP-utils
```

Then, create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Finally, install dependencies with either make:

```bash
make install
```

Or poetry:

```bash
poetry install
```

</details>

## Package Structure

```bash
stop_utils/
├── pyproject.toml
├── README.md
├── stop_utils/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI implementation
│   ├── converters.py       # Input errormap data converters
│   ├── types.py            # Custom types and data classes
│   ├── visualization.py    # Plotting utilities
│   └── wfe_analysis.py     # Core analysis functionality
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_cli.py
    ├── test_converters.py
    ├── test_visualization.py
    └── test_wfe_analysis.py
```

### Zemax Integration

The `zemax/` submodule provides tools for interfacing with Zemax OpticStudio through its ZOS-API:

```bash
stop_utils/zemax/
├── __init__.py
├── zmx_boilerplate.py     # Core ZOS-API interface
├── wavefront_extractor.py # Wavefront map extraction
└── zmx_batch_processor.py # Batch processing utility
```

Key features:

- Direct integration with Zemax OpticStudio via ZOS-API
- Automated wavefront map extraction from Zemax files
- Batch processing capabilities for multiple Zemax files
- Support for custom wavelength and surface selection
- Automated visualization of wavefront maps

Requirements:

- Zemax OpticStudio (Premium, Professional, or Standard Edition)
- Python.NET (pythonnet) package
- Valid Zemax license for API use

#### Usage

A dedicated CLI entry point is provided for batch processing Zemax files:

```bash
stop-utils zmx-batch -b "C:\path\to\zemax\files" -o "C:\path\to\output\dir" -s "EXPP" -w 0.633
```

This command processes all `.zmx` files in the specified folder, extracting wavefront maps for the given surface and wavelength. Output files are saved in the specified output directory.

**Options:**

- `--base-folder`, `-b`: Directory containing `.zmx` files (required)
- `--output-dir`, `-o`: Directory to save output files (required)
- `--surface-name`, `-s`: Surface name to process (default: `EXPP`)
- `--wavelength-um`, `-w`: Custom wavelength in micrometers

## Module Descriptions

### types.py

Contains type definitions and data classes used throughout the package:

```python
@dataclass
class WFEResult:
    raw: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]
    zernikes: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]
    pttf: npt.NDArray[np.float64]
    model: npt.NDArray[np.float64]

@dataclass
class AnalysisConfig:
    n_zernike: int
    save_coeffs: bool
    generate_plots: bool
    plot_format: str
```

### Core Components

1. **wfe_analysis.py**: Core functionality for WFE analysis
   - Loading and preprocessing WFE data
   - Support for raw WFE data and Zemax Wavefront Map text files
   - Elliptical mask creation and fitting
   - Orthonormal polynomial decomposition
   - Coefficient calculation and fitting

2. **visualization.py**: Plotting utilities
   - WFE map visualization
   - Elliptical mask and aperture plotting
   - Residual error mapping
   - Results visualization

3. **cli.py**: Command-line interface using Typer
   - Main analysis command
   - Progress tracking
   - Result output handling

## Usage

Analyze a wavefront error data file:

```bash
stop-utils analyze input_file output_dir/
```

Supported input formats:

- Raw WFE data files (`.dat`)
- Zemax Wavefront Map exports (`.txt`) from the "Text" button in the Wavefront Map analysis window or from `zmx-batch` (see above).

Options:

- `--nzernike`, `-n`: Number of polynomials (default: 15)
- `--plot-format`, `-f`: Plot output format (png, pdf, svg) (default: png)
- `--save-coeffs/--no-save-coeffs`: Save coefficients to JSON
- `--no-plots`: Skip plot generation
- `--help`, `-h`: Show help message
- `--version`, `-v`: Show version information

Examples:

```bash
# Analyze a raw WFE data file
stop-utils analyze wfe.dat results/ --nzernike 21 --plot-format pdf --save-coeffs

# Analyze a Zemax Wavefront Map file
stop-utils analyze wavefront_map.txt results/ --plot-format png --format zemax
```

## Outputs

The tool generates several outputs in the specified directory (where {format} is determined by the --plot-format option):

- `wfe_raw.{format}`: Raw wavefront error data
- `wfe_pttf.{format}`: Piston, tip, tilt, and focus components
- `wfe_model.{format}`: Orthonormal polynomial model fit
- `wfe_residual.{format}`: Residual error after model fit
- `polynomial_coefficients.{format}`: Bar plot of polynomial coefficients
- `polynomial_coefficients.json`: JSON file containing coefficient values

## Requirements

- Python ≥ 3.10
- Dependencies in `pyproject.toml`

## Development

The project includes a Makefile to streamline common development tasks:

```bash
make help         # Show available commands
make install      # Install dependencies using poetry
make test         # Run the test suite
make check        # Run type checking with mypy
make format       # Format code with black and isort
make docs         # Build the documentation
make clean        # Remove Python cache files and build artifacts
```

Example workflow:

1. Set up development environment:

    ```bash
    make install
    ```

2. Make your changes and format the code:

    ```bash
    make format
    ```

3. Run type checking and tests:

    ```bash
    make check
    make test
    ```

4. Clean up before committing:

    ```bash
    make clean
    ```

### Building Documentation

To build the documentation locally:

1. Install the documentation dependencies:

    ```bash
    poetry install --with docs
    # or using make:
    # make install-docs
    ```

2. Build the HTML documentation:

    ```bash
    make docs
    # or manually:
    # sphinx-build -b html docs/source docs/build/html
    ```

The generated HTML files will be in the `docs/build/html` directory. Open `docs/build/html/index.html` in your browser to view the documentation.

## Implementation Notes

1. Data Flow:

   ```mermaid
   graph TD
      A[CLI/Input File] --> B[Load WFE Data]
      B --> C[Detect Elliptical Aperture]
      C --> D[Fit Orthonormal Polynomials]
      D --> E[Generate Results & Plots]
      E --> F[Output Directory/Logs]
   ```

2. Core Functions:
   - `mask_to_elliptical_aperture()`: Converts mask to elliptical aperture
   - `calculate_zernike()`: Computes polynomials decomposition
   - `fit_zernike()`: Performs polynomial fitting
   - `generate_plots()`: Creates visualization outputs

3. Error Handling:
   - Input validation
   - Graceful failure for invalid data
   - Clear error messages via Typer

4. Performance Considerations:
   - Efficient numpy operations
   - Progress tracking for long operations
   - Optional plot generation

## Cite as

```bibtex
@INPROCEEDINGS{2024SPIE13092E..4KB,
       author = {{Bocchieri}, Andrea and {Mugnai}, Lorenzo V. and {Pascale}, Enzo},
        title = "{PAOS: a fast, modern, and reliable Python package for physical optics studies}",
    booktitle = {Space Telescopes and Instrumentation 2024: Optical, Infrared, and Millimeter Wave},
         year = 2024,
       editor = {{Coyle}, Laura E. and {Matsuura}, Shuji and {Perrin}, Marshall D.},
       series = {Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series},
       volume = {13092},
        month = aug,
          eid = {130924K},
        pages = {130924K},
          doi = {10.1117/12.3018333},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024SPIE13092E..4KB},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
