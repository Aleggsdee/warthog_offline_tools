import os
import open3d as o3d
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt  # Import pyplot to get the colormap

from doppler_raster import intensity_cmap
from aeva_loader import list_aeva_bins, load_aeva_frame

intensity_min = -60
intensity_max = -20
p = 0.0  # keep top percentile of brightest points

# ------------------ Color options ------------------
# Choose: "intensity" | "depth" | "height" | "signal_quality"
COLOR_MODE = "signal_quality"

# For depth/height coloring ranges (meters). If None, use min/max of current cloud.
DEPTH_MIN, DEPTH_MAX = None, None   # depth = x axis
HEIGHT_MIN, HEIGHT_MAX = None, None # height = z axis

# Colormap for geometric coloring (depth/height)
GEOM_CMAP_NAME = "viridis"


def compute_colors(points_all, intens_all, signal_quality_all,
                   color_mode="intensity",
                   intensity_min=-60, intensity_max=-20,
                   depth_min=None, depth_max=None,
                   height_min=None, height_max=None,
                   geom_cmap_name="viridis"):
    """
    Returns colors_rgb as (M,3) float32, aligned with points_all order.
    color_mode:
      - "intensity": use intens_all (your current behavior)
      - "depth":     use x axis
      - "height":    use z axis
      - "signal_quality": use signal quality
    """
    if points_all.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if color_mode == "intensity":
        scalar = intens_all
        cmap = intensity_cmap()
        norm = Normalize(vmin=intensity_min, vmax=intensity_max, clip=True)

    elif color_mode == "depth":
        scalar = points_all[:, 0]  # x axis
        vmin = float(np.min(scalar)) if depth_min is None else float(depth_min)
        vmax = float(np.max(scalar)) if depth_max is None else float(depth_max)
        cmap = plt.get_cmap(geom_cmap_name)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    elif color_mode == "height":
        scalar = points_all[:, 2]  # z axis
        vmin = float(np.min(scalar)) if height_min is None else float(height_min)
        vmax = float(np.max(scalar)) if height_max is None else float(height_max)
        cmap = plt.get_cmap(geom_cmap_name)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    elif color_mode == "signal_quality":
        scalar = signal_quality_all  # signal quality
        vmin = float(np.min(scalar)) if height_min is None else float(height_min)
        vmax = float(np.max(scalar)) if height_max is None else float(height_max)
        cmap = plt.get_cmap(geom_cmap_name)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    else:
        raise ValueError(f"Unknown COLOR_MODE='{color_mode}'. Use 'intensity', 'depth', 'height' or 'signal_quality'.")

    colors_rgb = cmap(norm(scalar))[:, :3].astype(np.float32)
    return colors_rgb



if __name__ == "__main__":
    # AEVA_DIR = "/home/asrl/Documents/Research/vtr3/data/Dec_16_2025/calib/aeva"
    AEVA_DIR = "/home/asrl/Documents/Research/vtr3/data/alex/rosbag2_2026_01_12-17_08_51/aeva"
    SAVE_DIR = "/home/asrl/Documents/Research/warthog_offline_tools/post_processing/calib_pcd"

    files = list_aeva_bins(AEVA_DIR) # each .bin contains one full LiDAR frame
    print(f"Found {len(files)} frames.")

    # Define bounds
    x_min, x_max = 0.0, 10.0
    y_min, y_max = -3.0, 4.0
    z_min, z_max = -1.5, 2.0

    all_points = []
    all_intensities = []
    all_signal_qualities = []

    for file in files[:1]:
        frame = load_aeva_frame(file)

        mask = (
            (frame[:, 0] >= x_min) & (frame[:, 0] <= x_max) &
            (frame[:, 1] >= y_min) & (frame[:, 1] <= y_max) &
            (frame[:, 2] >= z_min) & (frame[:, 2] <= z_max) &
            np.isfinite(frame[:, 4]) &
            np.isfinite(frame[:, 0]) & np.isfinite(frame[:, 1]) & np.isfinite(frame[:, 2])
        )

        intensity_thr = np.percentile(frame[:, 4], p)
        keep = mask & (frame[:, 4] >= intensity_thr)

        filtered = frame[keep]
        if filtered.shape[0] == 0:
            continue

        all_points.append(filtered[:, 0:3].astype(np.float32))
        all_intensities.append(filtered[:, 4].astype(np.float32))
        all_signal_qualities.append(filtered[:, 5].astype(np.float32))

    # Stack lists into np arrays
    points_all = np.vstack(all_points) if all_points else np.zeros((0,3), dtype=np.float32)
    intens_all = np.hstack(all_intensities) if all_intensities else np.zeros((0,), dtype=np.float32)
    signal_quality_all = np.hstack(all_signal_qualities) if all_signal_qualities else np.zeros((0,), dtype=np.float32)

    pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(points_all, dtype=o3d.core.float32))

    # ------------------ Color selection (intensity/depth/height) ------------------

    # Color by intensity
    cmap = intensity_cmap()
    norm = Normalize(vmin=intensity_min, vmax=intensity_max, clip=True)
    colors_rgb = compute_colors(
        points_all=points_all,
        intens_all=intens_all,
        signal_quality_all=signal_quality_all,
        color_mode=COLOR_MODE,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        height_min=HEIGHT_MIN,
        height_max=HEIGHT_MAX,
        geom_cmap_name=GEOM_CMAP_NAME,
    )

    pcd.point["colors"] = o3d.core.Tensor(colors_rgb, dtype=o3d.core.float32)
    
    # save pcd
    filename = "test.pcd"
    full_save_path = os.path.join(SAVE_DIR, filename)
    o3d.t.io.write_point_cloud(full_save_path, pcd, write_ascii=True)
    print(f"Successfully saved to: {os.path.abspath(SAVE_DIR)}")


    # Load pcd
    pcd = o3d.t.io.read_point_cloud(full_save_path)
    axes = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Set camera pose
    center = [2, 0, 0]
    up_direction = [0, 0, 1]
    eye = [1, 0, 0]

    # # Draw both the cloud and the axes
    # o3d.visualization.draw(
    #     [pcd, axes], 
    #     lookat=center, 
    #     eye=eye,
    #     up=up_direction
    # )

    # Convert tensor point cloud -> legacy point cloud for picking
    pcd_legacy = pcd.to_legacy()

    # Add coordinate frame for context
    axes_legacy = axes.to_legacy()

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick points: Shift+LeftClick, then press Q")
    
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.0, 0.0, 0.0])
    render_opt.point_size = 1.0   # (default ≈ 5)

    vis.add_geometry(pcd_legacy)
    vis.add_geometry(axes_legacy)

    # Camera settings
    ctr = vis.get_view_control()
    ctr.set_lookat([2, 0, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_front([-1, 0, 0])

    print("Instructions:")
    print("  - Press 'P' to enable point picking")
    print("  - Shift + Left Click to pick points")
    print("  - Press 'Q' to close window")
    vis.run()
    vis.destroy_window()

    picked_idx = vis.get_picked_points()
    print("Picked indices:", picked_idx)

    # IMPORTANT: intensities must match the point order used to build the pcd
    points_xyz = np.asarray(pcd_legacy.points)          
    intensities = intens_all.astype(np.float64)      

    for k, idx in enumerate(picked_idx):
        x, y, z = points_xyz[idx]
        I = intensities[idx]
        print(f"[{k}] idx={idx:6d}  xyz=({x:.4f}, {y:.4f}, {z:.4f})  intensity={I:.3f}")