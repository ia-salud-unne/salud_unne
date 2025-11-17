import csv
from pathlib import Path
from datetime import datetime, date
import pydicom
from pydicom.misc import is_dicom as pydicom_is_dicom

NON_DICOM_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
    ".htm", ".html", ".css", ".js",
    ".exe", ".bat", ".cmd", ".sh", ".jar",
    ".pdf", ".txt", ".xml", ".json", ".md", ".inf"
}

SKIP_DIRECTORIES = {"IHE_PDI", "PLUGINS", "JRE", "HELP", "XTR_CONT"}


# ----------------------------- File Utilities -----------------------------

def is_dicom(path: Path) -> bool:
    """Check if path points to a valid DICOM file.

    Args:
        path: File path to check

    Returns:
        True if file is a valid DICOM, False otherwise

    Note:
        - Filters common non-DICOM extensions to avoid unnecessary file operations
        - Uses pydicom.misc.is_dicom to check for 'DICM' prefix and other heuristics
        - Returns False on exceptions to avoid interrupting directory traversal
    """
    if path.suffix.lower() in NON_DICOM_EXTENSIONS:
        return False
    try:
        return pydicom_is_dicom(str(path))
    except Exception:
        return False


def read_header(path: Path):
    """Read only DICOM header (without pixel data).

    Args:
        path: Path to DICOM file

    Returns:
        pydicom.Dataset with header information

    Note:
        - stop_before_pixels=True avoids loading image data, speeding up I/O
        - force=True attempts to read even with non-standard headers
    """
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


# ----------------------------- Date/Time Converters -----------------------------

def to_iso_date(dicom_date: str) -> str:
    """Convert DICOM DA format 'YYYYMMDD' to 'YYYY-MM-DD'.

    Args:
        dicom_date: DICOM date string in format YYYYMMDD

    Returns:
        ISO 8601 date string 'YYYY-MM-DD', or empty string if invalid
    """
    if not dicom_date:
        return ""
    s = str(dicom_date).strip()
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def to_iso_datetime(dicom_dt: str) -> str:
    """Convert DICOM DT to ISO-8601 format.

    Args:
        dicom_dt: DICOM datetime string in format 'YYYYMMDDHHMMSS.FFFFFF±ZZZZ'

    Returns:
        ISO 8601 datetime string 'YYYY-MM-DDTHH:MM:SS(.fff)?±HH:MM'

    Note:
        - Supports truncated values (date only or date+time)
        - Reformats timezone ±HHMM to ±HH:MM if present
    """
    if not dicom_dt:
        return ""
    s = str(dicom_dt).strip()

    # Extract timezone if present (±HHMM at the end)
    tz = ""
    for i in range(len(s) - 5, len(s)):
        if i >= 0 and s[i] in "+-" and s[i+1:i+5].isdigit():
            tz = s[i:i+5]  # ±HHMM
            s = s[:i]
            break

    # Parse date
    year = s[0:4] if len(s) >= 4 else "0000"
    month = s[4:6] if len(s) >= 6 else "01"
    day = s[6:8] if len(s) >= 8 else "01"
    iso = f"{year}-{month}-{day}"

    # Parse time if present
    if len(s) > 8:
        hour = s[8:10] if len(s) >= 10 else "00"
        minute = s[10:12] if len(s) >= 12 else "00"
        second = s[12:14] if len(s) >= 14 else "00"
        fraction = ""
        if len(s) > 14 and s[14] == ".":
            fraction = s[14:]  # includes decimal point and digits
        iso += f"T{hour}:{minute}:{second}{fraction}"

    if tz:
        iso += f"{tz[:3]}:{tz[3:]}"  # ±HH:MM
    return iso


# ----------------------------- ROI Extraction -----------------------------

def us_region_bbox(ds):
    """Extract ultrasound ROI from Ultrasound Region Sequence (0018,6011).

    Args:
        ds: pydicom.Dataset

    Returns:
        Tuple (x0, y0, x1, y1) as integers if present and complete, None otherwise

    Note:
        Does not raise exceptions if sequence is missing or incomplete
    """
    try:
        elem = ds.get((0x0018, 0x6011))  # Ultrasound Region Sequence (SQ)
        seq = getattr(elem, "value", None) if elem is not None else None
        if not seq or len(seq) == 0:
            return None

        item = seq[0]  # First item in sequence

        def get_int(tag):
            e = item.get(tag)
            if e is None or e.value in (None, ""):
                return None
            try:
                return int(str(e.value).strip())
            except Exception:
                return None

        x0 = get_int((0x0018, 0x6018))  # RegionLocationMinX0
        y0 = get_int((0x0018, 0x601A))  # RegionLocationMinY0
        x1 = get_int((0x0018, 0x601C))  # RegionLocationMaxX1
        y1 = get_int((0x0018, 0x601E))  # RegionLocationMaxY1

        if None in (x0, y0, x1, y1):
            return None
        return (x0, y0, x1, y1)
    except Exception:
        # Don't block pipeline if extraction fails
        return None


# ----------------------------- DICOM Element Helpers -----------------------------

def get_element_str(ds, tag) -> str:
    """Get DICOM element value as string.

    Args:
        ds: pydicom.Dataset
        tag: DICOM tag (tuple or keyword)

    Returns:
        Element value as string, or empty string if tag doesn't exist or is None
    """
    elem = ds.get(tag)
    return "" if elem is None or elem.value is None else str(elem.value)


def _float_list(val, expected_len=None) -> list[float]:
    """Convert value to list of floats.

    Args:
        val: Value to convert (can be list, tuple, or string with separators)
        expected_len: If specified, return empty list if length doesn't match

    Returns:
        List of floats, or empty list if parsing fails

    Note:
        Useful for tags like ImageOrientationPatient (6) or ImagePositionPatient (3)
        Accepts string with backslash, comma, or space separators
    """
    if val is None:
        return []
    try:
        if isinstance(val, (list, tuple)):
            arr = [float(x) for x in val]
        else:
            s = str(val).replace("\\", " ").replace(",", " ")
            arr = [float(x) for x in s.split() if x.strip() != ""]
        if expected_len and len(arr) != expected_len:
            return []
        return arr
    except Exception:
        return []


def _parse_date_any(date_str) -> date | None:
    """Parse date in DICOM format 'YYYYMMDD' or ISO format 'YYYY-MM-DD'.

    Args:
        date_str: Date string to parse

    Returns:
        datetime.date object, or None if format is not recognized

    Note:
        Used to calculate age when PatientAge is not directly available
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    
    # Try YYYYMMDD format
    if len(s) >= 8 and s[:8].isdigit():
        try:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except Exception:
            pass
    
    # Try YYYY-MM-DD format
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


# ----------------------------- Age Calculation -----------------------------

def parse_patient_age_to_years(age_str) -> int | None:
    """Convert DICOM PatientAge to integer years.

    Args:
        age_str: DICOM age string (e.g., '068Y', '010M', '021W', '003D')

    Returns:
        Age in integer years, or None if parsing fails

    Rules:
        - 'Y': years
        - 'M': months // 12
        - 'W': weeks // 52
        - 'D': days // 365
    
    If it cannot parse, returns None
    """
    if not age_str:
        return None
    s = str(age_str).strip().upper()
    
    if len(s) < 2:
        return None
    
    num, unit = s[:-1], s[-1]
    
    if not num.isdigit():
        return None
    
    suffix = s[-1]
    try:
        number = int(s[:-1])
    except ValueError:
        return None
    
    if unit == 'Y': return n
    if unit == 'M': return n // 12
    if unit == 'W': return n // 52
    if unit == 'D': return n // 365
    else:
        return None


def compute_age_years(birth_date_str: str | None, study_date_str: str | None) -> int | None:
    """Calculate age in years from birth date and study date.

    Args:
        birth_date_str: Birth date string (DICOM or ISO format)
        study_date_str: Study date string (DICOM or ISO format)

    Returns:
        Age in integer years, or None if calculation is not possible
    """
    if not birth_date_str or not study_date_str:
        return None
    
    bd = _parse_date_any(birth_date_str)
    sd = _parse_date_any(study_date_str)
    
    if not bd or not sd or sd < bd:
        return None
    
    age = sd.year - bd.year
    if (sd.month, sd.day) < (bd.month, bd.day):
        age -= 1
    
    return max(0, age)


# ----------------------------- Procedure Codes -----------------------------

def extract_procedure_code_summary(ds) -> str:
    """Extract summary of procedure codes from ProcedureCodeSequence (0008,1032).

    Args:
        ds: pydicom.Dataset

    Returns:
        Semicolon-separated string of procedure codes, or empty string if none found

    Format:
        Each procedure formatted as: "(CodeValue)[CodeMeaning]@CodingSchemeDesignator"
    """
    try:
        elem = ds.get((0x0008, 0x1032))
        if elem is None:
            return ""
        
        seq = getattr(elem, "value", None)
        if not seq:
            return ""
        
        codes = []
        for item in seq:
            try:
                code_value = get_element_str(item, (0x0008, 0x0100))
                code_meaning = get_element_str(item, (0x0008, 0x0104))
                coding_scheme = get_element_str(item, (0x0008, 0x0102))
                
                parts = []
                if code_value:
                    parts.append(f"({code_value})")
                if code_meaning:
                    parts.append(f"[{code_meaning}]")
                if coding_scheme:
                    parts.append(f"@{coding_scheme}")
                
                if parts:
                    codes.append("".join(parts))
            except Exception:
                continue
        
        return "; ".join(codes)
    except Exception:
        return ""


# ----------------------------- Main Export Function -----------------------------

def export_metadata(input_dir: Path, csv_output: Path):
    """Export DICOM metadata to CSV file.

    Args:
        input_dir: Root directory containing DICOM files
        csv_output: Path where CSV file will be written

    Note:
        - Recursively walks through directory tree
        - Skips directories in SKIP_DIRECTORIES
        - Validates SOPInstanceUID uniqueness
        - Exports comprehensive metadata including patient, study, series, and image information
    """
    input_root = Path(input_dir).resolve()
    
    # Define CSV columns
    fieldnames = [
        "path_rel", "DeviceSerialNumber", "SoftwareVersions", "ProtocolName",
        "InstitutionName", "ReferringPhysicianName", "StationName",
        "ProcedureCodeSequence", "OperatorsName",
        "patient_key", "study_key",
        "PatientID_out", "PatientName_out",
        "StudyID_out", "SeriesNumber", "InstanceNumber",
        "StudyInstanceUID_out", "SeriesInstanceUID_out", "SOPInstanceUID_out",
        "clip_id", "SOPClassUID_out",
        "PatientSex", "BirthDate_out", "PatientAge_out_years",
        "Height_m", "Weight_kg",
        "Modality", "StudyDescription_out", "SeriesDescription_out",
        "StudyDate_out", "SeriesDate_out", "ContentDate_out", "AcquisitionDateTime_out",
        "Rows", "Columns", "NumberOfFrames", "is_multiframe", "frame_type", "fps",
        "PixelSpacing_row_mm", "PixelSpacing_col_mm",
        "ImageOrientationPatient_r0", "ImageOrientationPatient_r1", "ImageOrientationPatient_r2",
        "ImageOrientationPatient_c0", "ImageOrientationPatient_c1", "ImageOrientationPatient_c2",
        "ImagePositionPatient_x_mm", "ImagePositionPatient_y_mm", "ImagePositionPatient_z_mm",
        "FrameOfReferenceUID",
        "BitsAllocated", "BitsStored", "PixelRepresentation",
        "USRegion_MinX0", "USRegion_MinY0", "USRegion_MaxX1", "USRegion_MaxY1"
    ]

    rows = []
    total_dicom = 0
    errors = 0

    print(f"Scanning DICOM files in: {input_root}")
    
    # Walk directory tree
    for root, dirs, files in input_root.walk():
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRECTORIES]
        
        for filename in files:
            src = root / filename
            
            if not is_dicom(src):
                continue
            
            total_dicom += 1
            
            try:
                ds = read_header(src)
                
                # Determine if multiframe
                n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
                is_multiframe = n_frames > 1
                
                # Extract frame type
                frame_type_elem = ds.get((0x0008, 0x0008))
                frame_type = ""
                if frame_type_elem:
                    ft_val = frame_type_elem.value
                    if isinstance(ft_val, (list, tuple)):
                        frame_type = "\\".join(str(x) for x in ft_val)
                    else:
                        frame_type = str(ft_val)
                
                # Extract FPS from various possible tags
                fps_str = ""
                if is_multiframe:
                    # Try (0018,0040) Cine Rate
                    cine_rate = ds.get((0x0018, 0x0040))
                    if cine_rate and cine_rate.value:
                        try:
                            fps_str = str(float(cine_rate.value))
                        except Exception:
                            pass
                    
                    # Try (0018,1063) Frame Time (ms)
                    if not fps_str:
                        frame_time = ds.get((0x0018, 0x1063))
                        if frame_time and frame_time.value:
                            try:
                                ft_ms = float(frame_time.value)
                                if ft_ms > 0:
                                    fps_str = str(1000.0 / ft_ms)
                            except Exception:
                                pass
                    
                    # Try (0018,1065) Frame Time Vector (ms per frame)
                    if not fps_str:
                        ftv = ds.get((0x0018, 0x1065))
                        if ftv and ftv.value:
                            try:
                                vals = ftv.value
                                if isinstance(vals, (list, tuple)) and len(vals) > 0:
                                    ft_ms = float(vals[0])
                                    if ft_ms > 0:
                                        fps_str = str(1000.0 / ft_ms)
                            except Exception:
                                pass
                
                # Extract pixel spacing
                ps_row_mm = ""
                ps_col_mm = ""
                
                # Try PixelSpacing (0028,0030)
                ps = ds.get((0x0028, 0x0030))
                if ps and ps.value:
                    arr = _float_list(ps.value, expected_len=2)
                    if len(arr) == 2:
                        ps_row_mm = str(arr[0])
                        ps_col_mm = str(arr[1])
                
                # If not found, try Physical Delta from Ultrasound Region Sequence
                if not ps_row_mm or not ps_col_mm:
                    try:
                        elem = ds.get((0x0018, 0x6011))
                        if elem:
                            seq = getattr(elem, "value", None)
                            if seq and len(seq) > 0:
                                item = seq[0]
                                dx_elem = item.get((0x0018, 0x602C))  # Physical Delta X
                                dy_elem = item.get((0x0018, 0x602E))  # Physical Delta Y
                                ux_elem = item.get((0x0018, 0x6024))  # Physical Units X Direction
                                uy_elem = item.get((0x0018, 0x6026))  # Physical Units Y Direction
                                
                                dx = float(str(dx_elem.value)) if dx_elem and str(dx_elem.value).strip() else None
                                dy = float(str(dy_elem.value)) if dy_elem and str(dy_elem.value).strip() else None
                                ux = int(str(ux_elem.value)) if ux_elem and str(ux_elem.value).strip() else None
                                uy = int(str(uy_elem.value)) if uy_elem and str(uy_elem.value).strip() else None
                                
                                def get_multiplier(unit):
                                    """Convert units to mm: 3=cm, 4=mm"""
                                    if unit == 3:
                                        return 10.0
                                    if unit == 4:
                                        return 1.0
                                    return None
                                
                                my = get_multiplier(uy)
                                mx = get_multiplier(ux)
                                
                                if dy is not None and my is not None:
                                    ps_row_mm = str(abs(dy) * my)
                                if dx is not None and mx is not None:
                                    ps_col_mm = str(abs(dx) * mx)
                    except Exception:
                        pass
                
                # Extract orientation and position
                iop = _float_list(
                    ds.get((0x0020, 0x0037)).value if ds.get((0x0020, 0x0037)) else None,
                    expected_len=6
                )
                ipp = _float_list(
                    ds.get((0x0020, 0x0032)).value if ds.get((0x0020, 0x0032)) else None,
                    expected_len=3
                )
                
                r0 = r1 = r2 = c0 = c1 = c2 = x = y = z = ""
                if iop:
                    r0, r1, r2, c0, c1, c2 = [str(v) for v in iop]
                if ipp:
                    x, y, z = [str(v) for v in ipp]
                
                frame_uid = get_element_str(ds, (0x0020, 0x0052))
                
                # Extract anonymized identifiers
                patient_id_out = get_element_str(ds, (0x0010, 0x0020))
                patient_name_out = get_element_str(ds, (0x0010, 0x0010))
                study_iuid_out = get_element_str(ds, (0x0020, 0x000D))
                study_id_out = get_element_str(ds, (0x0020, 0x0010))
                
                patient_key = patient_id_out or patient_name_out or ""
                study_key = study_iuid_out or study_id_out or "study"
                
                # Extract device and protocol information
                device_serial = get_element_str(ds, (0x0018, 0x1000))
                software_vers = get_element_str(ds, (0x0018, 0x1020))
                protocol = get_element_str(ds, (0x0018, 0x1030))
                institution = get_element_str(ds, (0x0008, 0x0080))
                referring_phys = get_element_str(ds, (0x0008, 0x0090))
                station = get_element_str(ds, (0x0008, 0x1010))
                proc_codes = extract_procedure_code_summary(ds)
                operators = get_element_str(ds, (0x0008, 0x1070))
                
                # Build row dictionary
                row = {
                    "path_rel": str(src.relative_to(input_root)),
                    "DeviceSerialNumber": device_serial,
                    "SoftwareVersions": software_vers,
                    "ProtocolName": protocol,
                    "InstitutionName": institution,
                    "ReferringPhysicianName": referring_phys,
                    "StationName": station,
                    "ProcedureCodeSequence": proc_codes,
                    "OperatorsName": operators,
                    "patient_key": patient_key,
                    "study_key": study_key,
                    "PatientID_out": patient_id_out,
                    "PatientName_out": patient_name_out,
                    "StudyID_out": study_id_out,
                    "SeriesNumber": get_element_str(ds, (0x0020, 0x0011)),
                    "InstanceNumber": get_element_str(ds, (0x0020, 0x0013)),
                    "StudyInstanceUID_out": study_iuid_out,
                    "SeriesInstanceUID_out": get_element_str(ds, (0x0020, 0x000E)),
                    "SOPInstanceUID_out": get_element_str(ds, (0x0008, 0x0018)),
                    "clip_id": get_element_str(ds, (0x0008, 0x0018)),
                    "SOPClassUID_out": get_element_str(ds, (0x0008, 0x0016)),
                    "PatientSex": get_element_str(ds, (0x0010, 0x0040)),
                    "BirthDate_out": to_iso_date(get_element_str(ds, (0x0010, 0x0030))),
                    "PatientAge_out_years": "",
                    "Height_m": get_element_str(ds, (0x0010, 0x1020)),
                    "Weight_kg": get_element_str(ds, (0x0010, 0x1030)),
                    "Modality": get_element_str(ds, (0x0008, 0x0060)),
                    "StudyDescription_out": get_element_str(ds, (0x0008, 0x1030)),
                    "SeriesDescription_out": get_element_str(ds, (0x0008, 0x103E)),
                    "StudyDate_out": to_iso_date(get_element_str(ds, (0x0008, 0x0020))),
                    "SeriesDate_out": to_iso_date(get_element_str(ds, (0x0008, 0x0021))),
                    "ContentDate_out": to_iso_date(get_element_str(ds, (0x0008, 0x0023))),
                    "AcquisitionDateTime_out": to_iso_datetime(get_element_str(ds, (0x0008, 0x002A))),
                    "Rows": get_element_str(ds, (0x0028, 0x0010)),
                    "Columns": get_element_str(ds, (0x0028, 0x0011)),
                    "NumberOfFrames": str(n_frames),
                    "is_multiframe": "1" if is_multiframe else "0",
                    "frame_type": frame_type,
                    "fps": fps_str,
                    "PixelSpacing_row_mm": ps_row_mm,
                    "PixelSpacing_col_mm": ps_col_mm,
                    "ImageOrientationPatient_r0": r0,
                    "ImageOrientationPatient_r1": r1,
                    "ImageOrientationPatient_r2": r2,
                    "ImageOrientationPatient_c0": c0,
                    "ImageOrientationPatient_c1": c1,
                    "ImageOrientationPatient_c2": c2,
                    "ImagePositionPatient_x_mm": x,
                    "ImagePositionPatient_y_mm": y,
                    "ImagePositionPatient_z_mm": z,
                    "FrameOfReferenceUID": frame_uid,
                    "BitsAllocated": get_element_str(ds, (0x0028, 0x0100)),
                    "BitsStored": get_element_str(ds, (0x0028, 0x0101)),
                    "PixelRepresentation": get_element_str(ds, (0x0028, 0x0103)),
                }
                
                # Extract ultrasound ROI
                roi = us_region_bbox(ds)
                rows_val = int(row["Rows"]) if row["Rows"].isdigit() else None
                cols_val = int(row["Columns"]) if row["Columns"].isdigit() else None
                
                if roi and rows_val and cols_val:
                    x0, y0, x1, y1 = roi
                    # Validate ROI is within image bounds
                    if not (0 <= x0 <= x1 < cols_val and 0 <= y0 <= y1 < rows_val):
                        print(
                            f"[WARNING] ROI out of bounds in {row['path_rel']}: "
                            f"(x0,y0,x1,y1)=({x0},{y0},{x1},{y1}) vs "
                            f"(cols,rows)=({cols_val},{rows_val})"
                        )
                
                if roi:
                    x0, y0, x1, y1 = roi
                    row["USRegion_MinX0"] = str(x0)
                    row["USRegion_MinY0"] = str(y0)
                    row["USRegion_MaxX1"] = str(x1)
                    row["USRegion_MaxY1"] = str(y1)
                else:
                    row["USRegion_MinX0"] = ""
                    row["USRegion_MinY0"] = ""
                    row["USRegion_MaxX1"] = ""
                    row["USRegion_MaxY1"] = ""
                
                # Calculate patient age
                age_years = None
                age_elem = ds.get((0x0010, 0x1010))
                if age_elem:
                    age_years = parse_patient_age_to_years(age_elem.value)
                
                if age_years is None:
                    birth_date = row["BirthDate_out"]
                    study_date = row["StudyDate_out"]
                    age_years = compute_age_years(
                        birth_date if birth_date else None,
                        study_date if study_date else None
                    )
                
                row["PatientAge_out_years"] = "" if age_years is None else str(age_years)
                
                rows.append(row)
                
            except Exception as e:
                errors += 1
                print(f"[ERROR] DICOM: {src} -> {e}")
    
    # Validate SOPInstanceUID uniqueness
    uids = [r.get("SOPInstanceUID_out", "") for r in rows if r.get("SOPInstanceUID_out")]
    if uids:
        seen = set()
        duplicates = set()
        for uid in uids:
            if uid in seen:
                duplicates.add(uid)
            else:
                seen.add(uid)
        
        if duplicates:
            print("\n[ALERT] Duplicate SOPInstanceUIDs detected:")
            for uid in list(duplicates)[:10]:  # Show first 10
                print(f"   - {uid}")
            print(f"Total duplicates: {len(duplicates)}\n")
        else:
            print("\n[OK] No duplicate SOPInstanceUIDs detected.\n")
    else:
        print("\n[WARNING] No valid SOPInstanceUID values found.\n")
    
    # Write CSV
    try:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV generated: {csv_output} (rows: {len(rows)})")
    except Exception as e:
        print(f"[ERROR] Writing CSV {csv_output} -> {e}")
    
    print(f"DICOM files read: {total_dicom}")
    if errors:
        print(f"Files with errors: {errors}")