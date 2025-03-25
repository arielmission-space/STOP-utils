# STOP-Utils

Utilities for analyzing wavefront error data.

## Installation

```bash
poetry install
```

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
- Dependencies listed in pyproject.toml

## Development

Run tests:
```bash
poetry run pytest
```

Check types:
```bash
poetry run mypy .
```

Format code:
```bash
poetry run black .
poetry run isort .
```
