import csv
import math
from pathlib import Path
import numpy as np
import h5py
import pydicom
from pydicom.misc import is_dicom as pydicom_is_dicom


# ----------------------------- Constants -----------------------------

NON_DICOM_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
    ".htm", ".html", ".css", ".js", ".exe", ".bat", ".cmd", ".sh", ".jar",
    ".pdf", ".txt", ".xml", ".json", ".md", ".inf"
}


# ----------------------------- DICOM Utilities -----------------------------

def is_dicom(path: Path) -> bool:
    """Check if path points to a valid DICOM file.

    Args:
        path: File path to check

    Returns:
        True if file is a valid DICOM, False otherwise

    Note:
        - Filters common non-DICOM extensions to avoid unnecessary file operations
        - Uses pydicom.misc.is_dicom to check for 'DICM' prefix
        - Returns False on exceptions to avoid interrupting processing
    """
    if path.suffix.lower() in NON_DICOM_EXTENSIONS:
        return False
    try:
        return pydicom_is_dicom(str(path))
    except Exception:
        return False


def read_dicom(path: Path):
    """Read complete DICOM file including pixel data.

    Args:
        path: Path to DICOM file

    Returns:
        pydicom.Dataset with complete data
    """
    return pydicom.dcmread(str(path), force=True)


def to_native_frames(ds) -> np.ndarray:
    """Extract pixel array in native format.

    Args:
        ds: pydicom.Dataset

    Returns:
        Numpy array in format:
        - MONO: [T, H, W]
        - COLOR: [T, H, W, C]

    Note:
        Handles various DICOM pixel array layouts and converts to standard format
    """
    arr = ds.pixel_array
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # [1, H, W]
    elif arr.ndim == 3:
        if arr.shape[0] == n_frames:
            pass  # [T, H, W]
        elif arr.shape[-1] == n_frames and arr.dtype.kind in ("u", "i", "f"):
            arr = np.moveaxis(arr, -1, 0)  # [H, W, T] -> [T, H, W]
        else:
            if arr.shape[-1] in (3, 4):
                arr = arr[np.newaxis, ...]  # [H, W, C] -> [1, H, W, C]
    elif arr.ndim == 4:
        if arr.shape[0] == n_frames:
            pass
        elif arr.shape[2] == n_frames:
            arr = np.moveaxis(arr, 2, 0)  # [H, W, T, C] -> [T, H, W, C]
        else:
            if arr.shape[1] == n_frames:
                arr = np.moveaxis(arr, 1, 0)
    else:
        raise ValueError(f"Unsupported pixel_array dimensions: {arr.shape}")
    
    return arr


# ----------------------------- ROI Helpers (from CSV) -----------------------------

def to_int_or_none(value):
    """Convert value to integer, return None if conversion fails.

    Args:
        value: Value to convert

    Returns:
        Integer value or None
    """
    v = (value or "").strip()
    try:
        return int(v)
    except Exception:
        return None


def read_roi_from_row(row: dict):
    """Read ROI from CSV row.

    Args:
        row: CSV row dictionary

    Returns:
        Tuple (x0, y0, x1, y1) or None if ROI is missing or invalid

    Note:
        Looks for columns: USRegion_MinX0, USRegion_MinY0, USRegion_MaxX1, USRegion_MaxY1
    """
    x0 = to_int_or_none(row.get("USRegion_MinX0"))
    y0 = to_int_or_none(row.get("USRegion_MinY0"))
    x1 = to_int_or_none(row.get("USRegion_MaxX1"))
    y1 = to_int_or_none(row.get("USRegion_MaxY1"))
    if None in (x0, y0, x1, y1):
        return None
    return (x0, y0, x1, y1)


def clamp_roi_to_frame(roi, width, height):
    """Validate and clamp ROI to frame dimensions.

    Args:
        roi: Tuple (x0, y0, x1, y1)
        width: Frame width
        height: Frame height

    Returns:
        Clamped ROI tuple or None if invalid

    Note:
        Ensures ROI is within bounds [0..W-1], [0..H-1]
    """
    x0, y0, x1, y1 = roi
    x0c = max(0, min(x0, width - 1))
    x1c = max(0, min(x1, width - 1))
    y0c = max(0, min(y0, height - 1))
    y1c = max(0, min(y1, height - 1))
    if x1c < x0c or y1c < y0c:
        return None
    return (x0c, y0c, x1c, y1c)


def crop_frames(frames: np.ndarray, roi):
    """Crop frames to ROI (inclusive).

    Args:
        frames: Array [T, H, W] or [T, H, W, C]
        roi: Tuple (x0, y0, x1, y1)

    Returns:
        Cropped array
    """
    x0, y0, x1, y1 = roi
    if frames.ndim == 3:
        return frames[:, y0:y1+1, x0:x1+1]
    return frames[:, y0:y1+1, x0:x1+1, :]


# ----------------------------- Burner Mask (constant pixels) -----------------------------

def compute_constant_mask(frames: np.ndarray) -> np.ndarray:
    """Compute mask of pixels constant across all frames.

    Args:
        frames: Array [T, H, W] or [T, H, W, C]

    Returns:
        Boolean mask [H, W] where True indicates constant pixels

    Note:
        Vectorized implementation, no loops
        A pixel is constant if min == max across time dimension
    """
    min_val = frames.min(axis=0)
    max_val = frames.max(axis=0)
    if frames.ndim == 3:  # [T, H, W]
        return (max_val == min_val)  # [H, W]
    else:  # [T, H, W, C]
        return np.all(max_val == min_val, axis=-1)  # [H, W]


def apply_constant_mask_inplace(frames: np.ndarray, const_mask: np.ndarray, value=0):
    """Set masked pixels to value in all frames (in-place).

    Args:
        frames: Array [T, H, W] or [T, H, W, C]
        const_mask: Boolean mask [H, W]
        value: Value to set for masked pixels (default: 0)

    Note:
        Uses boolean indexing, doesn't require contiguous memory
    """
    if frames.ndim == 3:  # [T, H, W]
        frames[:, const_mask] = value  # -> [T, N]
    else:  # [T, H, W, C]
        frames[:, const_mask, :] = value  # -> [T, N, C]


# ----------------------------- CSV Manifest Loading -----------------------------

def load_manifest(csv_path: Path, min_frames: int):
    """Load CSV manifest and filter for video clips.

    Args:
        csv_path: Path to CSV file
        min_frames: Minimum number of frames required

    Returns:
        List of valid row dictionaries

    Note:
        Filters for:
        - is_multiframe == '1'
        - NumberOfFrames >= min_frames
        - Valid path_rel and clip_id/SOPInstanceUID_out
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                is_mf = str(row.get("is_multiframe", "")).strip() == "1"
                n_frames = int(row.get("NumberOfFrames") or 0)
                if not (is_mf and n_frames >= min_frames):
                    continue
                
                rel_path = (row.get("path_rel") or "").strip()
                if not rel_path:
                    continue
                
                clip_id = (row.get("clip_id") or row.get("SOPInstanceUID_out") or "").strip()
                if not clip_id:
                    continue
                
                rows.append(row)
            except Exception:
                pass
    
    if not rows:
        raise RuntimeError("CSV contains no valid rows (videos) matching criteria")
    
    return rows


def iter_paths_from_csv(input_root: Path, manifest_rows):
    """Generate (path, row) pairs from manifest.

    Args:
        input_root: Root directory containing DICOM files
        manifest_rows: List of CSV row dictionaries

    Yields:
        Tuples of (Path, dict) for each valid entry
    """
    base = input_root.resolve()
    for row in manifest_rows:
        rel_path = row["path_rel"].strip()
        yield (base / rel_path, row)


def chunked(seq, size: int):
    """Split sequence into chunks of given size.

    Args:
        seq: Sequence to chunk
        size: Chunk size

    Yields:
        Chunks of the sequence
    """
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


# ----------------------------- HDF5 Builder -----------------------------

def build_hdf5_from_dicom(
    input_dir: Path,
    output_h5: Path,
    csv_manifest: Path,
    min_frames: int = 2,
    compression: str = "lzf",
    gzip_level: int = 4,
    verbose: bool = True,
    strict_uid: bool = False,
    use_csv_roi: bool = True,
    mask_burners: bool = True,
    save_mask: bool = False,
    batch_size: int = 1000,
):
    """Build HDF5 dataset from DICOM videos with ROI and burner masking.

    Args:
        input_dir: Root directory containing DICOM files
        output_h5: Output HDF5 file path
        csv_manifest: CSV file with metadata (from export_dicom_metadata)
        min_frames: Minimum frames required for inclusion
        compression: Compression type: "lzf", "gzip", or "" (none)
        gzip_level: GZIP compression level (1-9)
        verbose: Print progress messages
        strict_uid: Validate DICOM UID matches CSV clip_id
        use_csv_roi: Apply ROI cropping from CSV
        mask_burners: Zero out constant pixels
        save_mask: Save burner mask as separate dataset
        batch_size: Process N DICOM files per batch

    Note:
        - Processes files in batches to manage memory
        - Skips duplicates (both in CSV and existing H5)
        - Creates hierarchical HDF5 structure: clips/<clip_id>/frames
        - Saves metadata as HDF5 attributes
    """
    batch_size = max(1, int(batch_size))
    input_dir = input_dir.resolve()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    # Load and sort manifest
    manifest_rows = load_manifest(csv_manifest, min_frames=min_frames)
    manifest_rows.sort(
        key=lambda r: (
            r.get("clip_id") or r.get("SOPInstanceUID_out") or "",
            r.get("path_rel") or ""
        )
    )

    # Statistics
    dup_count_csv = 0
    dup_count_h5 = 0
    seen_ids_csv = set()
    total_rows = 0
    total_clips_written = 0
    skipped = 0
    failed = 0

    # Load existing IDs from H5 to avoid duplicates
    with h5py.File(output_h5, "a") as f:
        clips_root = f.require_group("clips")
        existing_ids = set(clips_root.keys())

    total_batches = math.ceil(len(manifest_rows) / batch_size)
    if verbose:
        print(
            f"[INFO] Valid CSV rows: {len(manifest_rows)}  | "
            f"batch_size={batch_size}  | batches={total_batches}"
        )

    # Process in batches
    for batch_idx, batch_rows in enumerate(chunked(manifest_rows, batch_size), start=1):
        if verbose:
            print(f"\n======== BATCH {batch_idx}/{total_batches}  (rows: {len(batch_rows)}) ========")

        # Keep file open per batch (flushes on close)
        with h5py.File(output_h5, "a") as f:
            clips_root = f.require_group("clips")

            for src, row_csv in iter_paths_from_csv(input_dir, batch_rows):
                total_rows += 1
                try:
                    # Validate file exists and is DICOM
                    if not src.exists():
                        skipped += 1
                        if verbose:
                            print(f"[SKIP] File not found: {src}")
                        continue
                    
                    if not is_dicom(src):
                        skipped += 1
                        if verbose:
                            print(f"[SKIP] Not DICOM: {src}")
                        continue

                    # Extract clip ID
                    clip_id = (
                        row_csv.get("clip_id") or
                        row_csv.get("SOPInstanceUID_out") or
                        ""
                    ).strip()
                    if not clip_id:
                        skipped += 1
                        if verbose:
                            print(
                                f"[SKIP] Row missing clip_id/SOPInstanceUID_out: "
                                f"{row_csv.get('path_rel', '')}"
                            )
                        continue

                    # Check for duplicates in current CSV run
                    if clip_id in seen_ids_csv:
                        dup_count_csv += 1
                        if verbose:
                            print(f"[DUP-CSV] Repeated clip_id in CSV, skipping: {clip_id}")
                        continue
                    seen_ids_csv.add(clip_id)

                    # Check if already exists in H5
                    if clip_id in clips_root:
                        dup_count_h5 += 1
                        if verbose:
                            print(f"[DUP-H5] Already in HDF5, skipping: {clip_id}")
                        continue

                    # Read DICOM
                    ds = read_dicom(src)
                    
                    # Validate UID if strict mode
                    if strict_uid:
                        uid_dicom = str(ds.get((0x0008, 0x0018)).value or "")
                        if uid_dicom and clip_id and clip_id != uid_dicom:
                            skipped += 1
                            if verbose:
                                print(
                                    f"[SKIP] UID mismatch CSV vs DICOM: "
                                    f"{clip_id} != {uid_dicom} in {src}"
                                )
                            continue

                    # 1) Extract native frames
                    frames = to_native_frames(ds)  # [T, H, W] or [T, H, W, C]
                    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
                    orig_shape = tuple(frames.shape)

                    # 2) Apply ROI from CSV (optional)
                    roi_applied = (0, 0, W-1, H-1)
                    if use_csv_roi:
                        roi = read_roi_from_row(row_csv)
                        if roi:
                            roi = clamp_roi_to_frame(roi, W, H)
                            if roi:
                                frames = crop_frames(frames, roi)
                                x0, y0, x1, y1 = roi
                                roi_applied = (x0, y0, x1, y1)
                                H, W = frames.shape[1], frames.shape[2]

                    # 3) Compute burner mask (constant pixels)
                    mask_ratio = None
                    const_mask = None
                    if mask_burners:
                        const_mask = compute_constant_mask(frames)  # [H, W]
                        mask_ratio = float(const_mask.mean())
                        apply_constant_mask_inplace(frames, const_mask)

                    # 4) Write cleaned dataset to HDF5
                    spatial = frames.shape[1:]
                    chunks = (1,) + spatial
                    group = clips_root.create_group(clip_id)

                    if compression == "gzip":
                        dset = group.create_dataset(
                            "frames",
                            data=frames,
                            dtype=frames.dtype,
                            chunks=chunks,
                            compression="gzip",
                            compression_opts=int(gzip_level),
                            shuffle=True
                        )
                    elif compression == "lzf":
                        dset = group.create_dataset(
                            "frames",
                            data=frames,
                            dtype=frames.dtype,
                            chunks=chunks,
                            compression="lzf",
                            shuffle=True
                        )
                    else:
                        dset = group.create_dataset(
                            "frames",
                            data=frames,
                            dtype=frames.dtype,
                            chunks=chunks
                        )

                    # 5) Save attributes (copy from CSV, don't recalculate)
                    fps_str = (row_csv.get("fps") or "").strip()
                    if fps_str:
                        try:
                            dset.attrs["fps"] = float(fps_str)
                        except Exception:
                            pass

                    dset.attrs["orig_shape"] = np.array(orig_shape, dtype=np.int64)
                    dset.attrs["roi_applied"] = np.array(roi_applied, dtype=np.int64)
                    dset.attrs["mask_burners"] = int(bool(mask_burners))
                    if mask_ratio is not None:
                        dset.attrs["mask_ratio"] = float(mask_ratio)

                    # Optionally save burner mask
                    if save_mask and const_mask is not None:
                        group.create_dataset(
                            "burner_mask",
                            data=const_mask.astype(np.uint8),
                            dtype=np.uint8,
                            compression="lzf" if compression else None
                        )

                    total_clips_written += 1
                    if verbose and total_clips_written % 50 == 0:
                        print(f"[OK] Clips written (cumulative): {total_clips_written}")

                except Exception as e:
                    failed += 1
                    if verbose:
                        print(f"[ERROR] {src}: {e}")

        # End of batch - file is flushed automatically

    # Build index and global attributes
    with h5py.File(output_h5, "a") as f:
        clips_root = f.require_group("clips")
        all_ids = sorted(clips_root.keys())
        
        try:
            dt = h5py.string_dtype(encoding="utf-8")
            if "index_clip_ids" in f:
                del f["index_clip_ids"]
            f.create_dataset("index_clip_ids", data=np.array(all_ids, dtype=dt), dtype=dt)
        except Exception:
            pass

        f.attrs["schema"] = "clips/<clip_id>/frames -> cleaned, shape [T,H,W] or [T,H,W,C]"
        f.attrs["total_clips"] = len(all_ids)

    # Print summary
    print("\n===== SUMMARY =====")
    print(f"CSV rows processed         : {total_rows}")
    print(f"Clips written (new)        : {total_clips_written}")
    print(f"Skipped (not found/not D)  : {skipped}")
    print(f"Duplicates in CSV (same run): {dup_count_csv}")
    print(f"Duplicates in HDF5 (prev)  : {dup_count_h5}")
    print(f"Errors                     : {failed}")
    print(f"Output HDF5                : {output_h5}")