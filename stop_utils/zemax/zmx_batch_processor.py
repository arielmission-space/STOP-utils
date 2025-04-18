#!/usr/bin/env python
import argparse
import glob
import os
import sys
from pathlib import Path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process all ZMX files in a directory")
    parser.add_argument(
        "--base_folder", type=str, required=True, help="Directory containing ZMX files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="WavefrontOutputs",
        help="Directory to save output files (default: WavefrontOutputs)",
    )
    parser.add_argument(
        "--surface_name",
        type=str,
        default="EXPP",
        help="Surface name to look for (default: EXPP)",
    )
    parser.add_argument(
        "--wavelength_um", type=float, help="Custom wavelength in micrometers to use"
    )

    args = parser.parse_args()

    # Find all .zmx files in the directory
    zmx_files = glob.glob(os.path.join(args.base_folder, "*.zmx"))

    if not zmx_files:
        print(f"No .zmx files found in {args.base_folder}")
        return

    print(f"Found {len(zmx_files)} .zmx files to process")

    # Make sure output directory exists
    os.makedirs(os.path.join(args.base_folder, args.output_dir), exist_ok=True)

    # Import the wavefront extraction module
    try:
        # First, make sure the directory with your script is in sys.path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from wavefront_extractor import process_single_file
    except ImportError:
        print("Error: Could not import the wavefront_extractor module.")
        print(
            "Make sure wavefront_extractor.py is in the same directory as this script."
        )
        return

    # Process each file
    for i, zmx_file_path in enumerate(zmx_files):
        file_name = Path(zmx_file_path).stem
        print(f"\nProcessing file {i+1}/{len(zmx_files)}: {file_name}")

        try:
            # Call the function to process a single file
            process_single_file(
                zemax_file_path=zmx_file_path,
                base_folder=args.base_folder,
                output_dir=args.output_dir,
                surface_name=args.surface_name,
                wavelength_um=args.wavelength_um,
            )
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
