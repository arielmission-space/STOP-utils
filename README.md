# STOP-utils

Utilities for analyzing wavefront error data.

## Overview

This package provides utilities for Wavefront Error (WFE) analysis, implementing orthonormal polynomial decomposition and visualization tools. The core functionality is based on the `PAOS` package (see [readthedocs](https://paos.readthedocs.io/en/latest/)).

## Installation

Using make:
```bash
make install
```

Or directly with poetry:
```bash
poetry install
```

## Package Structure

```bash
stop_utils/
├── pyproject.toml
├── README.md
├── stop_utils/
│   ├── __init__.py
│   ├── wfe_analysis.py     # Core analysis functionality
│   ├── visualization.py    # Plotting utilities
│   ├── types.py            # Custom types and data classes
│   └── cli.py              # Typer CLI implementation
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_wfe_analysis.py
    ├── test_visualization.py
    └── test_cli.py
```

## Module Descriptions

### types.py

Contains type definitions and data classes used throughout the package:

```python
@dataclass
class WFEResult:
    coefficients: npt.NDArray[np.float64]
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
   - Elliptical mask creation and fitting
   - Zernike polynomial decomposition
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
stop-utils analyze input_file.dat output_dir/
```

Options:

- `--nzernike`, `-n`: Number of Zernike polynomials (default: 15)
- `--plot-format`, `-f`: Plot output format (png, pdf, svg) (default: png)
- `--save-coeffs/--no-save-coeffs`: Save Zernike coefficients to JSON
- `--no-plots`: Skip plot generation
- `--help`, `-h`: Show help message
- `--version`, `-v`: Show version information

Example usage:

```bash
stop-utils analyze wfe.dat results/ --nzernike 21 --plot-format pdf --save-coeffs
```

## Outputs

The tool generates several outputs in the specified directory:

- `wfe_raw.{format}`: Raw wavefront error data
- `wfe_pttf.{format}`: Piston, tip, tilt, and focus components
- `wfe_model.{format}`: Zernike model fit
- `wfe_residual.{format}`: Residual error after model fit
- `zernike_coefficients.{format}`: Bar plot of Zernike coefficients
- `zernike_coefficients.json`: JSON file containing coefficient values

## Requirements

- Python ≥ 3.9
- Dependencies in `pyproject.toml`

## Development

The project includes a Makefile to streamline common development tasks:

```bash
make help         # Show available commands
make install      # Install dependencies using poetry
make test         # Run the test suite
make check        # Run type checking with mypy
make format       # Format code with black and isort
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

## Implementation Notes

1. Data Flow:

   ```mermaid
   graph TD
      A[Input WFE Data] --> B[Preprocessing]
      B --> C[Elliptical Fitting]
      C --> D[Zernike Analysis]
      D --> E[Results Generation]
      E --> F[Plot Output]
      E --> G[Coefficient Output]
   ```

2. Core Functions:
   - `mask_to_elliptical_aperture()`: Converts mask to elliptical aperture
   - `calculate_zernike()`: Computes Zernike polynomials
   - `fit_zernike()`: Performs polynomial fitting
   - `generate_plots()`: Creates visualization outputs

3. Error Handling:
   - Input validation
   - Graceful failure for invalid data
   - Clear error messages via Rich console

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
