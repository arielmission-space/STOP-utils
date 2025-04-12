from pathlib import Path

import numpy as np
import pytest

from stop_utils import converters

# Get the directory of the current test file
TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"
ZEMAX_DIR = DATA_DIR / "zemax"


def test_load_zemax_wfe_valid() -> None:
    """Tests loading a valid Zemax WFE map file."""
    test_file = ZEMAX_DIR / "wfemap01fc.txt"
    assert test_file.exists(), f"Test file not found: {test_file}"

    wfe_data_nm = converters.load_zemax_wfe(str(test_file))

    # Check basic properties
    assert isinstance(wfe_data_nm, np.ndarray)
    assert wfe_data_nm.ndim == 2
    # Check shape based on header info in the sample file
    assert wfe_data_nm.shape == (512, 512)
    # Check data type
    assert wfe_data_nm.dtype == np.float64
    # Check if conversion to nm happened (values shouldn't be extremely small like waves)
    # Based on header: RMS = 0.0013 waves, Wavelength = 3.0 um
    # Expected RMS in nm = 0.0013 * 3.0 * 1000 = 3.9 nm
    # Let's check if the calculated RMS is in a reasonable range
    # Calculate RMS only for non-zero elements as zeros represent area outside pupil
    valid_data = wfe_data_nm[wfe_data_nm != 0]
    if valid_data.size > 0:
        rms_nm = np.std(valid_data, ddof=1)
        # Check if RMS is close to the expected value (allow some tolerance)
        assert (
            3.0 < rms_nm < 5.0
        ), f"Calculated RMS ({rms_nm:.3f} nm) not in expected range (3-5 nm)"
    else:
        # If all data is zero (unlikely for a real map), RMS is 0
        assert np.all(wfe_data_nm == 0)


def test_load_zemax_wfe_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent file."""
    non_existent_file = Path("non_existent_file.txt")
    with pytest.raises(FileNotFoundError):
        converters.load_zemax_wfe(non_existent_file)


def test_load_zemax_wfe_invalid_file(tmp_path: Path) -> None:
    """Test handling of invalid file formats."""
    # Create an empty file
    test_file = tmp_path / "empty.txt"
    test_file.write_text("")

    with pytest.raises(ValueError):
        converters.load_zemax_wfe(str(test_file))  # Convert Path to str


def test_load_zemax_wfe(tmp_path: Path) -> None:
    """Test successful loading of a Zemax WFE file."""
    # Create a test file with valid content
    test_file = tmp_path / "test_wfe.txt"
    content = (
        "ZYGO INTERFEROGRAM DATA\n"
        "Wavelength: 1.064900 µm\n"
        "Pupil grid size: 2 by 2\n"
        "\n"
        "-1.23456E-01 -2.34567E-01\n"
        "-3.45678E-01 -4.56789E-01\n"
    )
    test_file.write_text(content, encoding="utf-16le")

    wfe = converters.load_zemax_wfe(str(test_file))  # Convert Path to str
    assert isinstance(wfe, np.ndarray)
    assert wfe.shape == (2, 2)
    # Convert expected waves to nm (multiply by wavelength and 1000)
    expected = (
        np.array([[-0.123456, -0.234567], [-0.345678, -0.456789]]) * 1.0649 * 1000
    )
    np.testing.assert_array_almost_equal(wfe, expected)
