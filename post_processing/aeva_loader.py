import os
import re
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional

# ---------- helpers ----------

def _natural_numeric_key(p: Path) -> Tuple[int, str]:
    """
    Sort key: primary by largest integer found in the filename (timestamp),
    secondary by full name to keep ordering stable if ties.
    """
    nums = re.findall(r"\d+", p.stem)  # robust even if filename had spaces
    ts = int(nums[-1]) if nums else -1
    return (ts, p.name)

def list_aeva_bins(aeva_dir: str | Path) -> List[Path]:
    aeva_dir = Path(aeva_dir)
    files = sorted(aeva_dir.glob("*.bin"), key=_natural_numeric_key)
    if not files:
        raise FileNotFoundError(f"No .bin files found in {aeva_dir}")
    return files

def parse_epoch_seconds_from_name(p: Path) -> float:
    """
    Extract a timestamp from the filename and convert to seconds.
    Handles common magnitudes (ns/us/ms/seconds).
    Example names: 1761061025490200.bin  or  176106102_5490200.bin
    """
    nums = re.findall(r"\d+", p.stem)
    if not nums:
        raise ValueError(f"No digits in filename: {p.name}")
    n = int("".join(nums))  # concatenate all digit runs
    # Heuristic by magnitude
    if n >= 1_000_000_000_000_000_000:      # >= 1e18 → nanoseconds
        return n / 1e9
    elif n >= 1_000_000_000_000:            # >= 1e12 → microseconds
        return n / 1e6
    elif n >= 1_000_000_000:                # >= 1e9  → milliseconds or seconds
        # If it's exactly Unix seconds (10 digits ~ 1e9), this branch also triggers.
        # Prefer ms if 13 digits, else seconds.
        digits = len(str(n))
        if digits >= 13:
            return n / 1e3
        else:
            return float(n)
    else:
        # Fallback: treat as seconds (e.g., 10-digit epoch)
        return float(n)

# ---------- core loader ----------

def load_aeva_frame(filepath: str | Path, frame_time_s: Optional[float] = None,
                    out_dtype=np.float32) -> np.ndarray:
    """
    Load a single Aeva frame from a .bin file.

    Returns an array of shape (N, 11) with columns:
      x, y, z, radial_velocity, intensity, signal_quality, reflectivity, time_sec,
      lineID, beamID, faceID
    """
    filepath = Path(filepath)
    raw = np.fromfile(filepath, dtype=np.float32)
    if raw.size % 10 != 0:
        raise ValueError(f"{filepath.name}: expected 10 float32 columns per point, got size {raw.size}")
    pts = raw.reshape((-1, 10)).astype(out_dtype)

    # Column meanings in your extractor:
    # 0:x 1:y 2:z 3:velocity 4:intensity 5:signal_quality 6:reflectivity
    # 7:time_offset_ns 8:point_flags_lsb 9:point_flags_msb

    # Absolute time = frame_time + per-point offset (s)
    if frame_time_s is None:
        frame_time_s = parse_epoch_seconds_from_name(filepath)
    pts[:, 7] = pts[:, 7] * 1e-9 + float(frame_time_s)

    # Flags decoding: lsb/msb were stored as floats but hold integer values.
    flags_lsb = pts[:, 8].astype(np.uint32)
    flags_msb = pts[:, 9].astype(np.uint32)
    flags = (flags_msb << 16) | flags_lsb

    # Unpack IDs (adjust masks/shifts if your mapping differs)
    lineID = ((flags >> 8) & 0xFF).astype(out_dtype)
    beamID = ((flags >> 16) & 0xF).astype(out_dtype)
    faceID = ((flags >> 22) & 0xF).astype(out_dtype)

    # Keep first 8 cols (up to absolute time) and append IDs
    pts8 = pts[:, :8]
    out = np.hstack([pts8, lineID[:, None], beamID[:, None], faceID[:, None]])
    return out

def load_aeva_sequence(aeva_dir: str | Path) -> List[np.ndarray]:
    """
    Convenience: load the whole directory, sorted by filename timestamp.
    Returns a list of (N_i, 11) arrays, one per frame.
    """
    files = list_aeva_bins(aeva_dir)
    frames = []
    for f in files:
        t0 = parse_epoch_seconds_from_name(f)
        frames.append(load_aeva_frame(f, frame_time_s=t0))
    return frames

# ---------- example usage ----------

if __name__ == "__main__":
    # Point this to your 'aeva' folder (the one in your screenshot)
    AEVA_DIR = "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/aeva"

    files = list_aeva_bins(AEVA_DIR)
    print(f"Found {len(files)} frames.")

    # Load first frame
    frame0 = load_aeva_frame(files[1])  # shape (N, 11)
    print("Frame 0: ", frame0)
    print("vel finite ratio:", np.isfinite(frame0[:,3]).mean())
    print("Frame 0 shape:", frame0.shape)
    print("Columns: x y z vel I SQ R time lineID beamID faceID")
    print("Velocity min/max:", np.nanmin(frame0[:,3]), np.nanmax(frame0[:,3]))
    print("X min/max:", np.nanmin(frame0[:,0]), np.nanmax(frame0[:,0]))
    print("Y min/max:", np.nanmin(frame0[:,1]), np.nanmax(frame0[:,1]))
    print("Z min/max:", np.nanmin(frame0[:,2]), np.nanmax(frame0[:,2]))
    print("Time (s) sample:", frame0[:3, 7])
    print("IDs sample (first 5):")
    print(frame0[:5, 8:11].astype(int))