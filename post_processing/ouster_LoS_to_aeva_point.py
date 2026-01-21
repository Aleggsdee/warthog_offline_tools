# Computes transfrom between aeva and ouster, then uses aeva point distance and ouster LoS to compute intersection point in aeva frame

import numpy as np
import csv
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def solve_point_fusion(dist_aeva, los_ouster, T_aeva_ouster):
    """
    Solves for a 3D point given Aeva distance and Ouster line-of-sight.
    
    Args:
        dist_aeva (float): Distance from Aeva to point (r).
        los_ouster (np.array): 3D unit vector (x,y,z) in Ouster frame.
        T_aeva_ouster (np.array): 4x4 transform matrix from Ouster to Aeva.
        
    Returns:
        P_aeva (np.array): The solved [x, y, z] point in Aeva frame.
    """
    # 1. Extract Rotation (R) and Translation (t)
    R = T_aeva_ouster[:3, :3]
    t = T_aeva_ouster[:3, 3]

    # 2. Transform Ouster ray direction to Aeva frame (v)
    # Ensure los_ouster is normalized
    u = los_ouster / np.linalg.norm(los_ouster)
    v = R @ u
    
    # 3. Setup Quadratic: a*lambda^2 + b*lambda + c = 0
    # a = 1 (since v is unit vector)
    b = 2 * np.dot(v, t)
    c = np.dot(t, t) - dist_aeva**2
    
    # 4. Solve Discriminant
    delta = b**2 - 4*c
    
    if delta < 0:
        raise ValueError("No solution: Ouster line of sight never intersects Aeva range sphere.")
        
    # We only want the positive distance (ray shooting forward)
    lambda_dist = (-b + np.sqrt(delta)) / 2.0
    
    # 5. Compute final point
    # P_aeva = R * (lambda * u) + t
    P_aeva = lambda_dist * v + t
    
    return P_aeva

def get_aeva_distances(csv_path):
    # Get distance between every marker
    # IMPORTANT! DON'T FORGET TO ADD SENSOR Z OFFSET + VICON MARKER OFFSET
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"CSV is empty")

    cols = list(df.columns)
    points_matrix = df.values.reshape(-1, 3)
    dist_array = pdist(points_matrix, metric='euclidean')
    dist_matrix = squareform(dist_array)
    unsorted_marker_names = [c[:-2] for c in cols if c.endswith("_x") and (c[:-2] + "_y") in cols and (c[:-2] + "_z") in cols]
    dist_df = pd.DataFrame(
        dist_matrix,
        index = unsorted_marker_names,
        columns = unsorted_marker_names
    )
    print(dist_df.to_string(float_format="{:.4f}".format))
    aeva_distances = list(dist_df.loc['aeva', :])
    aeva_distances = aeva_distances[:-1]
    return aeva_distances


# 1. Define T_aeva_robot
T_aeva_robot = np.array([
    [9.999966697881730315e-01, -4.011860039981095054e-04, -2.549404313529212047e-03, -1.978520301731837294e-01],
    [4.030695525093277716e-04, 9.999996461883550181e-01, 7.383482245329893142e-04, 1.367627902960360945e-03],
    [2.549107196546519186e-03, -7.393733529328172863e-04, 9.999964776835693625e-01, -6.628822377478854611e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
])

# 2. Define T_ouster_robot
# Using the values you provided
T_ouster_robot = np.array([
    [-1, 1.97847e-31, 1.22465e-16, 0.025],
    [-1.97847e-31, 1, -3.23109e-15, 0.002],
    [-1.22465e-16, -3.23109e-15, -1, 0.84282],
    [0, 0, 0, 1]
])

# 3. Calculate Transformations
# T_aeva_ouster = T_aeva_robot * inv(T_ouster_robot)
T_aeva_ouster = T_aeva_robot @ np.linalg.inv(T_ouster_robot)

# T_ouster_aeva = inv(T_aeva_ouster)
T_ouster_aeva = np.linalg.inv(T_aeva_ouster)

# 4. Print Results
np.set_printoptions(precision=6, suppress=True)

print("T_aeva_ouster:")
print(T_aeva_ouster)
print("\nT_ouster_aeva:")
print(T_ouster_aeva)

# -----------------------------------------------------------------------------------

# Solve for intersection between aeva sphere and ouster line
aeva_distances_csv_path = "/home/asrl/Documents/Research/vicon_data_extraction/postprocessing/vicon_markers.csv"
distances_m = get_aeva_distances(aeva_distances_csv_path)
ouster_ray = np.array([[ -0.970694, 0.144083, 0.192334],
                       [ -0.989382,  -0.000081, 0.145342],
                       [ -0.998821, 0.037041, 0.031380],
                       [ -0.988910,  -0.146603, -0.023761],
                       [ -0.986029,  -0.146179, 0.079870],
                       [ -0.972817,  -0.144907, 0.180637]
                      ]) # line of sight in Ouster frame

# Loop and Solve
solved_points = []
point_names = []

for i in range(len(ouster_ray)):
    dist = distances_m[i]
    ray = ouster_ray[i]
    
    point_aeva = solve_point_fusion(dist, ray, T_aeva_ouster)
    
    solved_points.append(point_aeva)
    point_names.extend([f"aeva{i+1}_x", f"aeva{i+1}_y", f"aeva{i+1}_z"])

# Save to CSV
flat_data = [coord for point in solved_points for coord in point]
df_out = pd.DataFrame([flat_data], columns=point_names)

print("Calculated Points (First 3):")
print(df_out.iloc[:, :9].to_string())
df_out.to_csv('01_19_2026/calculated_aeva_points_from_ouster_LoS.csv', index=False)