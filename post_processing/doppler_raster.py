# doppler_raster.py
# Rasterize Aeva Doppler frames from your .bin directory and (optionally) save a video.

from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

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


def make_velocity_raster(
    points: np.ndarray,
    xlim: Tuple[float, float] = (-10.0, 10.0),
    ylim: Tuple[float, float] = (0, 12),
    res: float = 0.20,
    min_points: int = 2,
    clip: Optional[float] = 5.0,
    gate_col: Optional[int] = None,          # 0=x, 1=y, 2=z (choose depth axis)
    gate_min: Optional[float] = None,
    gate_max: Optional[float] = None,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Front-view raster (we’re using y vs z for front view below).
    Optional gate on a third axis (e.g., depth).
    """
    # ---- choose axes for the 2D raster (front view: y vs z) ----
    x = points[:, 1]  # horizontal = y (left/right)
    y = points[:, 2]  # vertical   = z (up)
    v = points[:, 3]  # radial velocity

    # ---- optional gate on a third axis (e.g., depth) ----
    if gate_col is not None and gate_min is not None and gate_max is not None:
        g = points[:, gate_col]
        keep = (g >= gate_min) & (g <= gate_max)
        x, y, v = x[keep], y[keep], v[keep]

    # ---- ROI filter on the chosen axes ----
    m = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1]) & np.isfinite(v)
    x, y, v = x[m], y[m], v[m]

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
    grid, extent, vmax=3.0, title=None, save_path=None, show=False, dpi=200
):
    from matplotlib.colors import TwoSlopeNorm
    cmap = doppler_cmap()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(np.ma.masked_invalid(grid),
                   origin="lower", extent=extent,
                   cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("x (m)  left (-)  right (+)")
    ax.set_ylabel("y (m)  forward")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("m/s")
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f"-{int(vmax)} m/s", "0", f"{int(vmax)} m/s"])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
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
) -> List[Path]:
    """
    Iterate through the .bin directory, save a PNG per frame.
    Returns list of saved PNG paths (sorted).
    """
    aeva_dir = Path(aeva_dir)
    out_dir = Path(out_dir)
    files = list_aeva_bins(aeva_dir)
    saved: List[Path] = []

    for i, f in enumerate(files):
        if i % every_k != 0:
            continue
        frame = load_aeva_frame(f)  # (N, 11)
        
        grid, extent, _ = make_velocity_raster(
            frame,
            xlim=xlim, ylim=ylim, res=res, min_points=min_points, clip=clip,
            gate_col=0, gate_min=2.0, gate_max=20.0    # <— gate by depth (x), e.g., 2–80 m
            # if you truly want to gate by z-as-“depth”, use gate_col=2 and set a z-range
        )
        ts = parse_epoch_seconds_from_name(f)
        title = f"{f.name}  |  t={ts:.3f}s"
        png_path = out_dir / f"{f.stem}.png"
        render_raster_png(grid, extent, vmax=vmax, title=title, save_path=png_path, show=False)
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

    ap = argparse.ArgumentParser(description="Aeva Doppler rasterizer")
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
    args = ap.parse_args()

    pngs = rasterize_directory_to_pngs(
        args.aeva_dir, args.out_dir,
        xlim=tuple(args.xlim), ylim=tuple(args.ylim),
        res=args.res, min_points=args.min_points, clip=args.clip,
        vmax=args.vmax, every_k=args.every_k,
    )
    print(f"Saved {len(pngs)} PNGs to {args.out_dir}")

    if args.video:
        try:
            pngs_to_video(pngs, args.video, fps=args.fps)
            print(f"Video written to {args.video}")
        except ModuleNotFoundError:
            print("imageio not installed. Install with: pip install imageio[ffmpeg]")
            print("Alternatively, make a video with ffmpeg:\n"
                  "  ffmpeg -framerate {fps} -pattern_type glob -i '{args.out_dir}/*.png' "
                  "-c:v libx264 -pix_fmt yuv420p out.mp4")
