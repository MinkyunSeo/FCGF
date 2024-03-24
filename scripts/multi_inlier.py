"""
A collection of unrefactored functions.
"""
import os
import sys
import numpy as np
import argparse
import open3d as o3d
import copy
import math

from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch
from lib.eval import find_nn_cpu



if __name__ == "__main__":
    
    src_fine_path = 'features/multi/3DMatch_0.025_fine/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_005.npz'
    src_mid_path = 'features/multi/3DMatch_0.025_mid64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_005.npz'
    src_coarse_path = 'features/multi/3DMatch_0.025_coarse64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_005.npz'
    
    tgt_fine_path = 'features/multi/3DMatch_0.025_fine/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_006.npz'
    tgt_mid_path = 'features/multi/3DMatch_0.025_mid64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_006.npz'
    tgt_coarse_path = 'features/multi/3DMatch_0.025_coarse64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_006.npz'

    fine_traj =[
      [0.932192051198, 0.086888881551, -0.351380565692, 0.372938956191],
      [-0.089297269725, 0.995960855403, 0.009379345602, 0.005967823860],
      [0.350776249628, 0.022633973735, 0.936185732603, -0.222355966132],
      [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ]
    
    mid_traj = [
      [0.932190714382, 0.091065094637, -0.350325021315, 0.372936161700],
      [-0.094090233024, 0.995528112983, 0.008414529666, 0.010424779096],
      [0.349524677341, 0.025118216469, 0.936590398804, -0.225725368776],
      [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ]

    coarse_traj = [
      [0.931182711032, 0.090344818387, -0.353180651317, 0.379097434821],
      [-0.091830612494, 0.995695110847, 0.012585104093, 0.003779641304],
      [0.352797246706, 0.020713764183, 0.935470492688, -0.219965124333],
      [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ]
    
    
    # Convert the list of lists to a NumPy array
    trans_fine = np.linalg.inv(fine_traj)
    trans_mid = np.linalg.inv(mid_traj)
    trans_coarse = np.linalg.inv(coarse_traj)

    src_fine = np.load(src_fine_path)
    src_mid = np.load(src_mid_path)
    src_coarse = np.load(src_coarse_path)
    
    tgt_fine = np.load(tgt_fine_path)
    tgt_mid = np.load(tgt_mid_path)
    tgt_coarse = np.load(tgt_coarse_path)
    
    coords_i, coords_j = src_fine['xyz'], tgt_fine['xyz']
    feats_fine_i, feats_fine_j = src_fine['feature'], tgt_fine['feature']
    feats_mid_i, feats_mid_j = src_mid['feature'], tgt_mid['feature']
    feats_coarse_i, feats_coarse_j = src_coarse['feature'], tgt_coarse['feature']

    pcd_fine_i = make_open3d_point_cloud(coords_i)
    pcd_fine_j = make_open3d_point_cloud(coords_j)
    pcd_mid_i = make_open3d_point_cloud(coords_i)
    pcd_coarse_i = make_open3d_point_cloud(coords_i)
    
    pcd_fine_i.transform(trans_fine)
    pcd_mid_i.transform(trans_mid)
    pcd_coarse_i.transform(trans_coarse)
  
    fine_inds = find_nn_cpu(feats_fine_i, feats_fine_j,return_distance=False)
    mid_inds = find_nn_cpu(feats_mid_i, feats_mid_j,return_distance=False)
    coarse_inds = find_nn_cpu(feats_coarse_i, feats_coarse_j,return_distance=False)  
    
    trans_fine_coords_i = np.asarray(pcd_fine_i.points)
    trans_mid_coords_i = np.asarray(pcd_mid_i.points)
    trans_coarse_coords_i = np.asarray(pcd_coarse_i.points)

    inlier_indices = []
    outlier_indices = []
    resolution_indices = []
    
    # threshold = 1.5 * 0.025  # 1.5 times the voxel size
    threshold = 0.1
    
    for i, point in enumerate(trans_fine_coords_i):
      dist_fine = np.sqrt(np.sum((trans_fine_coords_i[i] - coords_j[fine_inds[i]])**2))
      dist_mid = np.sqrt(np.sum((trans_mid_coords_i[i] - coords_j[mid_inds[i]])**2))
      dist_coarse = np.sqrt(np.sum((trans_coarse_coords_i[i] - coords_j[coarse_inds[i]])**2))
      
      min_dist = min(dist_fine, dist_mid, dist_coarse)
      min_idx = np.argmin([dist_fine, dist_mid, dist_coarse])
      inds = [fine_inds[i], mid_inds[i], coarse_inds[i]]
      
      if min_dist < threshold:
        inlier_indices.append((i, inds[min_idx]))
        resolution_indices.append((i, inds[min_idx], min_idx))
      else:
        outlier_indices.append((i, inds[min_idx]))
  
    # Convert the lists to NumPy arrays for further processing if needed
    inlier_indices = np.array(inlier_indices)
    outlier_indices = np.array(outlier_indices)
    resolution_indices = np.array(resolution_indices)
    
    # Save inlier indices to a text file
    inlier_file_path = 'features/multi/lab/inlier2.txt'
    np.savetxt(inlier_file_path, inlier_indices, fmt='%d')

    # Save outlier indices to a text file
    outlier_file_path = 'features/multi/lab/outlier2.txt'
    np.savetxt(outlier_file_path, outlier_indices, fmt='%d')
    
    # Save resolution indices to a text file
    resolution_file_path = 'features/multi/lab/resolution2.txt'
    np.savetxt(resolution_file_path, resolution_indices, fmt='%d')
    
    # print(f"{source_idx}, {target_idx} corr set saved. Inlier: {len(inlier_indices)}, Outlier: {len(outlier_indices)}")    
    # print(f"pcd_source: {len(pcd_i.points)}, pcd_target: {len(pcd_j.points)}")


  

