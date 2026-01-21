#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv

# -------------------------
# Geometry helpers
# -------------------------

def load_gt_map(gt_csv_path):
    gt = {}
    with open(gt_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["cluster_id"])
            d = float(row["distance_m"])
            gt[cid] = d
    return gt


def draw_axes(ax, origin=(0,0,0), L=1.0):
    ox, oy, oz = origin
    ax.plot([ox, ox+L], [oy, oy],   [oz, oz],   linewidth=3)  # +X
    ax.plot([ox, ox],   [oy, oy+L], [oz, oz],   linewidth=3)  # +Y
    ax.plot([ox, ox],   [oy, oy],   [oz, oz+L], linewidth=3)  # +Z
    ax.text(ox+L, oy,   oz, "X")
    ax.text(ox,   oy+L, oz, "Y")
    ax.text(ox,   oy,   oz+L, "Z")


def set_equal_axes_3d(ax, X):
    """
    Force equal scaling on x,y,z axes based on data X (N,3).
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    centers = (mins + maxs) / 2.0
    ranges = maxs - mins
    radius = 0.5 * np.max(ranges)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def point_to_line_dist(P: np.ndarray, p0: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Perpendicular distance from each point P (N,3) to line passing through p0 with direction v (unit).
    dist = || (P - p0) x v ||
    """
    v = v / np.linalg.norm(v)
    W = P - p0[None, :]
    return np.linalg.norm(np.cross(W, v[None, :]), axis=1)

def ransac_line_fit(P: np.ndarray,
                    max_iters: int = 2000,
                    dist_thresh: float = 0.02,
                    min_inliers: int = 30,
                    refine_pca: bool = True,
                    rng: np.random.Generator | None = None):
    """
    Fit a 3D line to points P using simple 2-point RANSAC.
    Returns (p0, v, inlier_idx) where v is unit direction.
    If refine_pca=True, p0 becomes centroid of inliers and v becomes PCA direction of inliers.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = P.shape[0]
    if N < 2:
        return None

    best_inliers = None
    best_p0 = None
    best_v = None
    best_count = -1

    for _ in range(max_iters):
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        p_i = P[i]
        p_j = P[j]
        v = p_j - p_i
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v = v / nv
        d = point_to_line_dist(P, p_i, v)
        inliers = np.where(d <= dist_thresh)[0]
        cnt = inliers.size
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_p0 = p_i
            best_v = v

    if best_inliers is None or best_count < min_inliers:
        return None

    if refine_pca:
        Pin = P[best_inliers]
        c = Pin.mean(axis=0)
        X = Pin - c[None, :]
        S = X.T @ X
        w, V = np.linalg.eigh(S)
        v = V[:, np.argmax(w)]
        v = v / np.linalg.norm(v)
        if v @ best_v < 0:
            v = -v
        best_p0 = c
        best_v = v

    return best_p0, best_v, best_inliers

def line_origin_distance(p0: np.ndarray, v: np.ndarray) -> float:
    """
    Shortest distance from origin to line p0 + t v.
    """
    v = v / np.linalg.norm(v)
    return np.linalg.norm(p0 - (v @ p0) * v)

def ray_pca_unit(P: np.ndarray) -> np.ndarray:
    """
    "Ray PCA": fit a direction u for a ray through the origin that best matches points P.
    Solve max_u u^T (sum p p^T) u  s.t. ||u||=1  => principal eigenvector of (P^T P).
    """
    S = P.T @ P  # 3x3
    w, V = np.linalg.eigh(S)
    u = V[:, np.argmax(w)]
    u = u / np.linalg.norm(u)

    # make sure ray is going outwards from origin to centroid
    centroid = np.mean(P, axis=0)
    if np.dot(u, centroid) < 0:
        u = -u
    return u

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True, help="Path to .pcd/.ply point cloud")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel size (m). 0 disables.")
    ap.add_argument("--eps", type=float, default=0.05, help="DBSCAN eps (m)")
    ap.add_argument("--min_points", type=int, default=50, help="DBSCAN min_points")
    ap.add_argument("--max_clusters", type=int, default=20, help="Max clusters to process (largest first)")
    ap.add_argument("--ransac_iters", type=int, default=3000)
    ap.add_argument("--ransac_thresh", type=float, default=0.02, help="RANSAC inlier dist (m)")
    ap.add_argument("--ransac_min_inliers", type=int, default=80)
    ap.add_argument("--plot_points_per_cluster", type=int, default=800)
    ap.add_argument("--line_length", type=float, default=1.0, help="Half-length to draw each line (m)")
    ap.add_argument("--ray_pca_thresh", type=float, default=0.02,
                    help="If dist(origin,line) < this (m), replace line with ray-PCA through origin.")
    ap.add_argument("--gt_csv", type=str, default=None,
                help="CSV with columns: cluster_id,distance_m (maps cluster id to ground-truth distance).")
    ap.add_argument("--plot_errors", action="store_true",
                    help="If set, plot range error histograms per cluster.")

    args = ap.parse_args()

    # Load
    pcd = o3d.io.read_point_cloud(args.pcd)
    if len(pcd.points) == 0:
        raise RuntimeError("Loaded point cloud has 0 points.")

    # Optional downsample
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    # Cluster
    labels = np.array(pcd.cluster_dbscan(eps=args.eps, min_points=args.min_points, print_progress=True))
    pts = np.asarray(pcd.points)

    cluster_ids = [cid for cid in np.unique(labels) if cid != -1]
    if len(cluster_ids) == 0:
        print("No clusters found (all noise). Try bigger eps or smaller min_points.")
        return

    # Sort clusters by size
    clusters = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        clusters.append((cid, idx))
    clusters.sort(key=lambda x: x[1].size, reverse=True)
    clusters = clusters[:args.max_clusters]

    print(f"\nFound {len(cluster_ids)} clusters, processing {len(clusters)} largest.\n")

    fitted = []  # (cid, p0, v, inliers_global, idx_all, used_ray_pca)
    rng = np.random.default_rng(0)

    for cid, idx_all in clusters:
        P = pts[idx_all]
        fit = ransac_line_fit(
            P,
            max_iters=args.ransac_iters,
            dist_thresh=args.ransac_thresh,
            min_inliers=args.ransac_min_inliers,
            refine_pca=True,
            rng=rng
        )
        if fit is None:
            print(f"Cluster {cid}: RANSAC FAILED (N={len(idx_all)})")
            continue

        p0, v, inliers_local = fit
        inliers_global = idx_all[inliers_local]
        d0 = line_origin_distance(p0, v)

        used_ray_pca = False

        # If line nearly passes through origin, replace with ray PCA through origin (using inliers)
        if d0 < args.ray_pca_thresh:
            Pin = pts[inliers_global]
            u = ray_pca_unit(Pin)
            p0 = np.zeros(3)  # ray goes through origin
            v = u
            used_ray_pca = True

        print(f"Cluster {cid}:")
        print(f"  points: {len(idx_all)}, inliers: {len(inliers_global)}")
        if used_ray_pca:
            print(f"  *** using RAY-PCA (through origin) because dist(origin,line)={d0:.4f} < {args.ray_pca_thresh:.4f} ***")
            print(f"  ray: x(t) = t u,  t>=0")
            print(f"  u  = [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}] (unit)")
        else:
            print(f"  line: x(t) = p0 + t v")
            print(f"  p0 = [{p0[0]: .4f}, {p0[1]: .4f}, {p0[2]: .4f}]")
            print(f"  v  = [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}] (unit)")
            print(f"  dist(origin, line) = {d0:.4f} m")
        print("")

        fitted.append((cid, p0, v, inliers_global, idx_all, used_ray_pca))

    if len(fitted) == 0:
        print("No clusters produced a valid line fit. Adjust DBSCAN/RANSAC params.")
        return
    
    # Compute distance errors
    gt_map = None
    if args.gt_csv is not None:
        gt_map = load_gt_map(args.gt_csv)
        print(f"Loaded {len(gt_map)} ground-truth distances from {args.gt_csv}")

    errors_by_cluster = {}  # cid -> errors (N,)
    ranges_by_cluster = {}  # cid -> ranges (N,)

    if gt_map is not None:
        for (cid, p0, v, inliers_global, idx_all, used_ray_pca) in fitted:
            if cid not in gt_map:
                print(f"[WARN] No GT distance for cluster {cid}; skipping error calc.")
                continue
            d_gt = gt_map[cid]
            P = pts[idx_all]
            r = np.linalg.norm(P, axis=1)
            e = r - d_gt
            errors_by_cluster[cid] = e
            ranges_by_cluster[cid] = r

            print(f"Cluster {cid}: GT={d_gt:.4f} m | "
                f"mean err={e.mean():+.4f} m | std={e.std():.4f} m | "
                f"rmse={np.sqrt(np.mean(e**2)):.4f} m | "
                f"p95={np.percentile(np.abs(e),95):.4f} m")

    if args.plot_errors and gt_map is not None and len(errors_by_cluster) > 0:
        n = len(errors_by_cluster)
        fig2, axes = plt.subplots(n, 1, figsize=(7, 2.2*n), sharex=True)
        if n == 1:
            axes = [axes]

        for axh, cid in zip(axes, sorted(errors_by_cluster.keys())):
            e = errors_by_cluster[cid]
            axh.hist(e, bins=20, rwidth=0.85, edgecolor="black", linewidth=0.5)
            axh.axvline(0.0, linewidth=2, color="red", linestyle='--')
            axh.set_title(f"Cluster {cid} range error: ||p|| - d_gt (N={len(e)})")
            axh.set_ylabel("count")

        axes[-1].set_xlabel("range error (m)")
        plt.tight_layout()
        plt.show()


    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # draw_axes(ax, origin=(0,0,0), L=0.75)

    colors = cm.get_cmap("tab10", len(fitted))  # distinct colors

    for k, (cid, p0, v, inliers_global, idx_all, used_ray_pca) in enumerate(fitted):
        color = colors(k)

        # sample points
        idx_use = idx_all
        if len(idx_use) > args.plot_points_per_cluster:
            idx_use = rng.choice(idx_use, size=args.plot_points_per_cluster, replace=False)
        Pplot = pts[idx_use]

        label = f"Cluster {cid}"
        if used_ray_pca:
            label += " (ray-PCA)"

        ax.scatter(
            Pplot[:, 0], Pplot[:, 1], Pplot[:, 2],
            s=4, color=color, alpha=0.6,
            label=label
        )

        # draw line / ray
        L = args.line_length
        a = p0 - L * v
        b = p0 + L * v
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color=color, linewidth=1, linestyle=':'
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Clusters + fitted lines (ray-PCA when near origin)")

    # Equal axis scaling
    set_equal_axes_3d(ax, pts)

    # Legend outside the plot (prevents clutter)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        markerscale=3.0
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# python cluster_line_fit.py --pcd /home/asrl/Documents/Research/warthog_offline_tools/post_processing/01_19_2026/calib_all_frames.pcd --voxel 0.01   --eps 0.03   --min_points 10   --ransac_thresh 0.02 --ransac_min_inliers 10   --line_length 3 --gt_csv /home/asrl/Documents/Research/warthog_offline_tools/post_processing/01_19_2026/gt_distances.csv --plot_errors