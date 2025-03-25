# Stop Utils Architecture

## Overview

This document outlines the architecture for the `stop_utils` package, which provides utilities for Wavefront Error (WFE) analysis.

## Package Structure

```
stop_utils/
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
├── stop_utils/
│   ├── __init__.py
│   ├── wfe_analysis.py     # Core analysis functionality
│   ├── visualization.py    # Plotting utilities
│   ├── types.py           # Custom types and data classes
│   └── cli.py             # Typer CLI implementation
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

### wfe_analysis.py

Core functionality for WFE analysis:
- Loading and preprocessing WFE data
- Elliptical mask creation and fitting
- Zernike polynomial decomposition
- Coefficient calculation and fitting

### visualization.py

Plotting utilities:
- WFE map visualization
- Elliptical mask and aperture plotting
- Residual error mapping
- Results visualization

### cli.py

Command-line interface implementation using Typer:
- Main analysis command
- Progress tracking
- Result output handling

## Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.8"
numpy = "*"
matplotlib = "*"
photutils = "*"
scikit-image = "*"
paos = "*"
typer = {extras = ["all"], version = "^0.9.0"}
rich = "^13.0.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0"
mypy = "^1.0.0"
black = "^23.0.0"
isort = "^5.0.0"
```

## CLI Interface

### Command: analyze-wfe

```bash
analyze-wfe INPUT_FILE OUTPUT_DIR [OPTIONS]
```

Options:
- `--nzernike, -n`: Number of Zernike polynomials (default: 15)
- `--plot-format, -f`: Plot output format (default: "png")
- `--save-coeffs/--no-save-coeffs`: Save Zernike coefficients (default: True)
- `--no-plots`: Skip plot generation

Example usage:
```bash
analyze-wfe wfe.dat results/ --nzernike 20 --plot-format pdf --save-coeffs
```

## Development Workflow

1. Environment Setup:
   ```bash
   poetry install
   ```

2. Running Tests:
   ```bash
   poetry run pytest
   ```

3. Code Formatting:
   ```bash
   poetry run black .
   poetry run isort .
   ```

4. Type Checking:
   ```bash
   poetry run mypy .
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