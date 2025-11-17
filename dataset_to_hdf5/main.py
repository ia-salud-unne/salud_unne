"""Command-line interface for DICOM to HDF5 pipeline."""

import argparse
import sys
from pathlib import Path
from . import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "DICOM -> CSV + HDF5 pipeline with ROI extraction and burner masking. "
            "Positional arguments: DICOM directory and output directory."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Base directory containing DICOM files"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where CSV and HDF5 files will be saved"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of DICOM files to process per batch (default: 1000)"
    )
    return parser


def main(argv=None):
    """Main entry point for CLI."""
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    csv_path, h5_path = run_pipeline(
        input_dir,
        output_dir,
        batch_size=args.batch_size
    )
    
    print("\n[OK] Generated artifacts:")
    print(f" - CSV : {csv_path}")
    print(f" - HDF5: {h5_path}")


if __name__ == "__main__":
    main()


# Example usage:
# python -m dicom_preprocessing.main \
#   "/path/to/dicom/dataset" \
#   "/path/to/hdf5/output" \
#   --batch-size 1000