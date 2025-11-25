# doppler_raster.py
# Rasterize Aeva Doppler frames from your .bin directory and (optionally) save a video.

from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # render off-screen, no Qt/Wayland

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.colors import Normalize


# ----- import your loader -----
# Make sure aeva_loader.py is in the same folder or on PYTHONPATH.
from aeva_loader import list_aeva_bins, load_aeva_frame, parse_epoch_seconds_from_name


# =========================
# Colormap & raster helpers
# =========================

def doppler_cmap() -> LinearSegmentedColormap:
    """Blue → cyan → white → yellow → red, with NaNs as black."""
    colors = [
        (0.00, (0.00, 0.00, 0.50)),  # deep blue
        (0.25, (0.00, 0.90, 1.00)),  # cyan
        (0.50, (1.00, 1.00, 1.00)),  # white
        (0.75, (1.00, 1.00, 0.00)),  # yellow
        (1.00, (0.80, 0.00, 0.00)),  # red
    ]
    cmap = LinearSegmentedColormap.from_list("doppler", colors)
    cmap.set_bad("black")
    return cmap


def make_raster(
    points: np.ndarray,
    xlim: Tuple[float, float] = (-10.0, 10.0),
    ylim: Tuple[float, float] = (0, 12),
    res: float = 0.20,
    min_points: int = 2,
    clip: Optional[float] = 5.0,
    gate_col: Optional[int] = None,          # 0=x, 1=y, 2=z (choose depth axis)
    gate_min: Optional[float] = None,
    gate_max: Optional[float] = None,
    value_col: int = 3,                      # 3 = velocity, 4 = intensity, 6 = reflectivity
    intensity_min: Optional[float] = None,   # filter by intensity before binning
    intensity_max: Optional[float] = None,
    reflectivity_min: Optional[float] = None,
    reflectivity_max: Optional[float] = None,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Generic front-view raster (we’re using y vs z for front view).
    value_col chooses which channel to rasterize (velocity, intensity, reflectivity).
    Optional gate on a third axis (e.g., depth), plus optional filters on intensity/reflectivity.
    """
    if points.size == 0:
        raise ValueError("Empty point cloud passed to make_raster")

    # ---- start with everything valid in the chosen value column ----
    mask = np.isfinite(points[:, value_col])

    # ---- optional gate on a third axis (e.g., depth) ----
    if gate_col is not None and gate_min is not None and gate_max is not None:
        g = points[:, gate_col]
        mask &= (g >= gate_min) & (g <= gate_max)

    # ---- optional filter by intensity (column 4) ----
    if intensity_min is not None or intensity_max is not None:
        intens = points[:, 4]
        if intensity_min is not None:
            mask &= intens >= intensity_min
        if intensity_max is not None:
            mask &= intens <= intensity_max

    # ---- optional filter by reflectivity (column 6) ----
    if reflectivity_min is not None or reflectivity_max is not None:
        refl = points[:, 6]
        if reflectivity_min is not None:
            mask &= refl >= reflectivity_min
        if reflectivity_max is not None:
            mask &= refl <= reflectivity_max

    # If nothing survives, return an all-NaN grid
    if not np.any(mask):
        W = int(np.ceil((xlim[1] - xlim[0]) / res))
        H = int(np.ceil((ylim[1] - ylim[0]) / res))
        grid = np.full((H, W), np.nan, dtype=np.float32)
        cnt = np.zeros((H, W), dtype=np.int32)
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        return grid, extent, cnt

    # ---- choose axes for the 2D raster (front view: y vs z) ----
    x = points[mask, 1]          # horizontal = y (left/right)
    y = points[mask, 2]          # vertical   = z (up)
    v = points[mask, value_col]  # chosen channel

    # ---- ROI filter on the chosen axes ----
    roi = (
        (x >= xlim[0]) & (x <= xlim[1]) &
        (y >= ylim[0]) & (y <= ylim[1]) &
        np.isfinite(v)
    )
    x, y, v = x[roi], y[roi], v[roi]

    if x.size == 0:
        W = int(np.ceil((xlim[1] - xlim[0]) / res))
        H = int(np.ceil((ylim[1] - ylim[0]) / res))
        grid = np.full((H, W), np.nan, dtype=np.float32)
        cnt = np.zeros((H, W), dtype=np.int32)
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        return grid, extent, cnt

    # ---- bin & average ----
    x_edges = np.arange(xlim[0], xlim[1] + res, res)
    y_edges = np.arange(ylim[0], ylim[1] + res, res)
    W, H = len(x_edges) - 1, len(y_edges) - 1

    ix = np.floor((x - xlim[0]) / res).astype(int)
    iy = np.floor((y - ylim[0]) / res).astype(int)
    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, v = ix[valid], iy[valid], v[valid]

    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.int32)
    np.add.at(acc, (iy, ix), v)
    np.add.at(cnt, (iy, ix), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = acc / np.maximum(cnt, 1)

    grid[cnt < min_points] = np.nan
    if clip is not None:
        grid = np.clip(grid, -clip, clip)

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    return grid, extent, cnt


def render_raster_png(
    grid,
    extent,
    vmax: float = 3.0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 200,
    channel: str = "velocity",   # "velocity", "intensity", or "reflectivity"
):
    if channel == "velocity":
        cmap = doppler_cmap()
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cbar_label = "Velocity (m/s)"
        cbar_ticks = [-vmax, 0, vmax]
        cbar_ticklabels = [f"-{int(vmax)} m/s", "0", f"{int(vmax)} m/s"]
    elif channel == "intensity":
        # Raw returned power; just use a standard colormap
        cmap = plt.get_cmap("viridis")
        norm = None
        cbar_label = "Intensity"
        cbar_ticks = None
        cbar_ticklabels = None
    else:  # "reflectivity"
        cmap = plt.get_cmap("viridis")
        vmin, vmax = 0.0, 50.0     # TODO: Tune
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "Reflectivity"
        cbar_ticks = None
        cbar_ticklabels = None

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("x (m)  left (-)  right (+)")
    ax.set_ylabel("y (m)  forward")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()  # optional, keeps labels from getting cut off
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


# =====================
# Batch & video helpers
# =====================

def rasterize_directory_to_pngs(
    aeva_dir: str | Path,
    out_dir: str | Path,
    xlim=(-10.0, 10.0),
    ylim=(0, 12),
    res=0.2,
    min_points=2,
    clip=3.0,
    vmax=3.0,
    every_k: int = 1,         # downsample frames (1 = use all)
    channel: str = "velocity",
    intensity_min: Optional[float] = None,
    intensity_max: Optional[float] = None,
    reflectivity_min: Optional[float] = None,
    reflectivity_max: Optional[float] = None,
) -> List[Path]:
    """
    Iterate through the .bin directory, save a PNG per frame.
    Returns list of saved PNG paths (sorted).

    channel: "velocity", "intensity", or "reflectivity"
    intensity_* and reflectivity_* filters are applied before rasterization.
    """
    aeva_dir = Path(aeva_dir)
    out_dir = Path(out_dir)
    files = list_aeva_bins(aeva_dir)
    saved: List[Path] = []

    # Map channel name to column index in (N, 11) array
    if channel == "velocity":
        value_col = 3
    elif channel == "intensity":
        value_col = 4
    else:  # "reflectivity"
        value_col = 6

    for i, f in enumerate(files):
        if i % every_k != 0:
            continue
        frame = load_aeva_frame(f)  # (N, 11)

        grid, extent, _ = make_raster(
            frame,
            xlim=xlim,
            ylim=ylim,
            res=res,
            min_points=min_points,
            clip=clip if channel == "velocity" else None,
            gate_col=0,       # gate by x (0th channel)
            gate_min=None,    # 2.0 m
            gate_max=None,    # 20.0 m
            value_col=value_col,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            reflectivity_min=reflectivity_min,
            reflectivity_max=reflectivity_max,
        )
        ts = parse_epoch_seconds_from_name(f)
        title = f"{f.name}  |  t={ts:.3f}s"
        png_path = out_dir / f"{f.stem}.png"
        render_raster_png(
            grid,
            extent,
            vmax=vmax,
            title=title,
            save_path=png_path,
            show=False,
            channel=channel,
        )
        saved.append(png_path)
    return saved


def pngs_to_video(png_paths: List[Path], out_mp4: str | Path, fps: int = 10) -> None:
    import imageio
    with imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=7) as w:
        for p in png_paths:
            w.append_data(imageio.imread(p))


# ================
# Simple CLI usage
# ================

if __name__ == "__main__":
    import argparse

    # TODO: may need to change x and y limits
    # VELOCITY EXAMPLE:
    # python doppler_raster.py \
    # "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/aeva" \
    # --out_dir "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/rasters_vel" \
    # --channel velocity \
    # --xlim -3 4 \
    # --ylim -1.5 2 \
    # --res 0.01 \
    # --vmax 3 \
    # --clip 3 \
    # --every_k 1 \
    # --video "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/vel_raster.mp4" \
    # --fps 10

    # INTENSITY EXAMPLE:
    # python doppler_raster.py \
    # "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/aeva" \
    # --out_dir "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/rasters_intensity" \
    # --channel intensity \
    # --xlim -3 4 \
    # --ylim -1.5 2 \
    # --res 0.01 \
    # --vmax 3 \
    # --clip 3 \
    # --every_k 1 \
    # --video "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/intensity_raster.mp4" \
    # --fps 10

    # REFLECTIVITY EXAMPLE:
    # python doppler_raster.py \
    # "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/aeva" \
    # --out_dir "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/rasters_reflectivity" \
    # --channel reflectivity \
    # --xlim -3 4 \
    # --ylim -1.5 2 \
    # --res 0.01 \
    # --every_k 1 \
    # --video "/home/asrl/Documents/Research/vtr3/data/rosbag2_copy/reflectivity_raster.mp4" \
    # --fps 10

    ap = argparse.ArgumentParser(description="Aeva rasterizer (velocity / intensity / reflectivity)")
    ap.add_argument("aeva_dir", type=str, help="Folder with Aeva .bin files")
    ap.add_argument("--out_dir", type=str, default="rasters", help="Where to write PNGs")
    ap.add_argument("--video", type=str, default="", help="Optional MP4 output path")
    ap.add_argument("--fps", type=int, default=10, help="Video FPS if --video is set")
    ap.add_argument("--xlim", type=float, nargs=2, default=[-30, 30])
    ap.add_argument("--ylim", type=float, nargs=2, default=[0, 60])
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--min_points", type=int, default=2)
    ap.add_argument("--clip", type=float, default=5.0, help="Velocity clip for color range")
    ap.add_argument("--vmax", type=float, default=5.0, help="Colorbar max (white at 0)")
    ap.add_argument("--every_k", type=int, default=1, help="Downsample frames (use every k-th)")
    ap.add_argument(
        "--channel",
        type=str,
        choices=["velocity", "intensity", "reflectivity"],
        default="velocity",
        help="Which channel to rasterize",
    )
    ap.add_argument(
        "--intensity_min",
        type=float,
        default=None,
        help="Drop points with intensity below this value",
    )
    ap.add_argument(
        "--intensity_max",
        type=float,
        default=None,
        help="Drop points with intensity above this value",
    )
    ap.add_argument(
        "--reflectivity_min",
        type=float,
        default=None,
        help="Drop points with reflectivity below this value",
    )
    ap.add_argument(
        "--reflectivity_max",
        type=float,
        default=None,
        help="Drop points with reflectivity above this value",
    )
    args = ap.parse_args()

    pngs = rasterize_directory_to_pngs(
        args.aeva_dir,
        args.out_dir,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        res=args.res,
        min_points=args.min_points,
        clip=args.clip,
        vmax=args.vmax,
        every_k=args.every_k,
        channel=args.channel,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        reflectivity_min=args.reflectivity_min,
        reflectivity_max=args.reflectivity_max,
    )
    print(f"Saved {len(pngs)} PNGs to {args.out_dir}")

    if args.video:
        try:
            pngs_to_video(pngs, args.video, fps=args.fps)
            print(f"Video written to {args.video}")
        except ModuleNotFoundError:
            print("imageio not installed. Install with: pip install imageio[ffmpeg]")
            print(
                "Alternatively, make a video with ffmpeg:\n"
                "  ffmpeg -framerate {fps} -pattern_type glob -i '{args.out_dir}/*.png' "
                "-c:v libx264 -pix_fmt yuv420p out.mp4"
            )
