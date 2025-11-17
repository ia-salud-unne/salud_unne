from pathlib import Path
from .export_dicom_metadata import export_metadata
from .dicom_videos_to_hdf5 import build_hdf5_from_dicom

__all__ = ["export_metadata", "build_hdf5_from_dicom", "run_pipeline"]
__version__ = "0.1.0"


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 1000
) -> tuple[Path, Path]:
    """Run complete pipeline: export CSV metadata then build HDF5.

    Args:
        input_dir: Root directory containing DICOM files
        output_dir: Directory where CSV and HDF5 will be saved
        batch_size: Number of DICOM files to process per batch

    Returns:
        Tuple of (csv_path, h5_path) with generated file paths
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_dir.name.strip() or "dataset"
    csv_path = output_dir / f"{base_name}_metadata.csv"
    h5_path = output_dir / f"{base_name}.h5"

    print(f"[1/2] Exporting metadata -> {csv_path}")
    export_metadata(input_dir, csv_path)

    print(f"[2/2] Building HDF5 -> {h5_path}")
    build_hdf5_from_dicom(
        input_dir=input_dir,
        output_h5=h5_path,
        csv_manifest=csv_path,
        min_frames=2,
        compression="lzf",
        gzip_level=4,
        verbose=True,
        strict_uid=False,
        use_csv_roi=True,
        mask_burners=True,
        save_mask=False,
        batch_size=batch_size,
    )
    return csv_path, h5_path