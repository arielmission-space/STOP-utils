# STOP-utils

Utilities for analyzing wavefront error data.

## Overview

This package provides utilities for Wavefront Error (WFE) analysis, implementing Zernike orthonormal polynomial decomposition and visualization tools. The core functionality is based on the `PAOS` package (see [readthedocs](https://paos.readthedocs.io/en/latest/)).

## Installation

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

Example usage:

```bash
stop-utils analyze wfe.dat results/ --nzernike 20 --plot-format pdf --save-coeffs
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

1. Environment Setup:

   ```bash
   poetry install
   ```

2. Run tests:

   ```bash
   poetry run pytest
   ```

3. Check types:

   ```bash
   poetry run mypy .
   ```

4. Format code:

   ```bash
   poetry run black .
   poetry run isort .
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
