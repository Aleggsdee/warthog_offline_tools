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
    nums = re.findall(r"\d+", p.stem)
    ts = int(nums[-1]) if nums else -1
    return (ts, p.name)

def list_ouster_bins(ouster_dir: str | Path) -> List[Path]:
    ouster_dir = Path(ouster_dir)
    # Filter for .bin files and sort them naturally by timestamp
    files = sorted(ouster_dir.glob("*.bin"), key=_natural_numeric_key)
    if not files:
        raise FileNotFoundError(f"No .bin files found in {ouster_dir}")
    return files

def parse_epoch_seconds_from_name(p: Path) -> float:
    """
    Extract a timestamp from the filename and convert to seconds.
    The extraction script saves filenames in microseconds (timestamp / 1000).
    """
    nums = re.findall(r"\d+", p.stem)
    if not nums:
        raise ValueError(f"No digits in filename: {p.name}")
    n = int("".join(nums))  # concatenate all digit runs
    
    # Heuristic by magnitude
    if n >= 1_000_000_000_000_000_000:      # >= 1e18 → nanoseconds
        return n / 1e9
    elif n >= 1_000_000_000_000:            # >= 1e12 → microseconds (Likely case for your script)
        return n / 1e6
    elif n >= 1_000_000_000:                # >= 1e9  → milliseconds or seconds
        digits = len(str(n))
        if digits >= 13:
            return n / 1e3
        else:
            return float(n)
    else:
        return float(n)

# ---------- core loader ----------

def load_ouster_frame(filepath: str | Path, frame_time_s: Optional[float] = None,
                      out_dtype=np.float32) -> np.ndarray:
    """
    Load a single Ouster frame from a .bin file created by the extraction script.

    The extraction script saves 9 floats per point:
    0: x
    1: y
    2: z
    3: intensity
    4: t (time offset in ns)
    5: reflectivity
    6: ring
    7: ambient
    8: range

    Returns an array of shape (N, 9) where the 't' column (idx 4) 
    is converted to Absolute Time (seconds).
    """
    filepath = Path(filepath)
    raw = np.fromfile(filepath, dtype=np.float32)
    
    # Check consistency: we expect 9 columns based on your extraction script
    if raw.size % 9 != 0:
        raise ValueError(f"{filepath.name}: expected 9 float32 columns per point, got total size {raw.size}")
        
    pts = raw.reshape((-1, 9)).astype(out_dtype)

    # Calculate Absolute Time
    # frame_time_s comes from the filename (microseconds -> seconds)
    # pts[:, 4] is 't' from Ouster, typically nanoseconds relative to frame start
    if frame_time_s is None:
        frame_time_s = parse_epoch_seconds_from_name(filepath)
    
    # Update column 4 to be absolute seconds
    pts[:, 4] = pts[:, 4] * 1e-9 + float(frame_time_s)

    return pts

def load_ouster_sequence(ouster_dir: str | Path) -> List[np.ndarray]:
    """
    Convenience: load the whole directory, sorted by filename timestamp.
    Returns a list of (N, 9) arrays.
    """
    files = list_ouster_bins(ouster_dir)
    frames = []
    for f in files:
        t0 = parse_epoch_seconds_from_name(f)
        frames.append(load_ouster_frame(f, frame_time_s=t0))
    return frames

# ---------- example usage ----------

if __name__ == "__main__":
    # Point this to your 'ouster' folder
    OUSTER_DIR = "/home/asrl/Documents/Research/vtr3/data/ouster_test/ouster"

    try:
        files = list_ouster_bins(OUSTER_DIR)
        print(f"Found {len(files)} frames.")

        # Load first frame
        frame0 = load_ouster_frame(files[0])  # shape (N, 9)
        
        print(f"Frame 0 shape: {frame0.shape}")
        print("-" * 30)
        print("Columns: x, y, z, intensity, time(abs), reflectivity, ring, ambient, range")
        
        # Basic Stats
        print(f"X min/max:         {np.nanmin(frame0[:,0]):.2f}, {np.nanmax(frame0[:,0]):.2f}")
        print(f"Y min/max:         {np.nanmin(frame0[:,1]):.2f}, {np.nanmax(frame0[:,1]):.2f}")
        print(f"Z min/max:         {np.nanmin(frame0[:,2]):.2f}, {np.nanmax(frame0[:,2]):.2f}")
        print(f"Intensity min/max: {np.nanmin(frame0[:,3]):.2f}, {np.nanmax(frame0[:,3]):.2f}")
        
        # Time check
        t_start = frame0[0, 4]
        t_end = frame0[-1, 4]
        print(f"Time (s) start:    {t_start:.6f}")
        print(f"Scan Duration (s): {t_end - t_start:.6f}")
        
        # Ring check (e.g. 0 to 127 for OS1-128)
        print(f"Ring ID min/max:   {np.nanmin(frame0[:,6]):.0f}, {np.nanmax(frame0[:,6]):.0f}")

    except Exception as e:
        print(f"Error: {e}")