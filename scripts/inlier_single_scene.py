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
    
    src_global_path = 'features/multi/3DMatch_0.025_global/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_005.npz'
    src_mid_path = 'features/multi/3DMatch_0.025_mid64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_005.npz'
    src_local_path = 'features/multi/3DMatch_0.025_local64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_005.npz'
    
    tgt_global_path = 'features/multi/3DMatch_0.025_global/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_006.npz'
    tgt_mid_path = 'features/multi/3DMatch_0.025_mid64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_006.npz'
    tgt_local_path = 'features/multi/3DMatch_0.025_local64/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_006.npz'

    global_traj =[
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

    local_traj = [
      [0.931182711032, 0.090344818387, -0.353180651317, 0.379097434821],
      [-0.091830612494, 0.995695110847, 0.012585104093, 0.003779641304],
      [0.352797246706, 0.020713764183, 0.935470492688, -0.219965124333],
      [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ]
    
    # Convert the list of lists to a NumPy array
    trans_global = np.linalg.inv(global_traj)
    trans_mid = np.linalg.inv(mid_traj)
    trans_local = np.linalg.inv(local_traj)

    src_global = np.load(src_global_path)
    src_mid = np.load(src_mid_path)
    src_local = np.load(src_local_path)
    
    tgt_global = np.load(tgt_global_path)
    tgt_mid = np.load(tgt_mid_path)
    tgt_local = np.load(tgt_local_path)
    
    coords_i, coords_j = src_global['xyz'], tgt_global['xyz']
    feats_global_i, feats_global_j = src_global['feature'], tgt_global['feature']
    feats_mid_i, feats_mid_j = src_mid['feature'], tgt_mid['feature']
    feats_local_i, feats_local_j = src_local['feature'], tgt_local['feature']

    pcd_global_i = make_open3d_point_cloud(coords_i)
    pcd_global_j = make_open3d_point_cloud(coords_j)
    pcd_mid_i = make_open3d_point_cloud(coords_i)
    pcd_local_i = make_open3d_point_cloud(coords_i)
    
    pcd_global_i.transform(trans_global)
    pcd_mid_i.transform(trans_mid)
    pcd_local_i.transform(trans_local)
  
    global_inds = find_nn_cpu(feats_global_i, feats_global_j,return_distance=False)
    mid_inds = find_nn_cpu(feats_mid_i, feats_mid_j,return_distance=False)
    local_inds = find_nn_cpu(feats_local_i, feats_local_j,return_distance=False)  
    
    trans_global_coords_i = np.asarray(pcd_global_i.points)
    trans_mid_coords_i = np.asarray(pcd_mid_i.points)
    trans_local_coords_i = np.asarray(pcd_local_i.points)

    local_inlier_indices = []
    local_outlier_indices = []
    mid_inlier_indices = []
    mid_outlier_indices = []
    global_inlier_indices = []
    global_outlier_indices = []
    
    # threshold = 1.5 * 0.025  # 1.5 times the voxel size
    threshold = 0.1
    
    for i, point in enumerate(trans_global_coords_i):
      dist_mid = np.sqrt(np.sum((trans_mid_coords_i[i] - coords_j[mid_inds[i]])**2))
      dist_local = np.sqrt(np.sum((trans_local_coords_i[i] - coords_j[local_inds[i]])**2))
      dist_global = np.sqrt(np.sum((trans_global_coords_i[i] - coords_j[global_inds[i]])**2))
      
      if dist_mid < threshold:
        mid_inlier_indices.append((i, mid_inds[i]))
      else:
        mid_outlier_indices.append((i, mid_inds[i]))
        
      if dist_local < threshold:
        local_inlier_indices.append((i, local_inds[i]))
      else:
        local_outlier_indices.append((i, local_inds[i]))
      
      if dist_global < threshold:
        global_inlier_indices.append((i, global_inds[i]))
      else:
        global_outlier_indices.append((i, global_inds[i]))
        
    # Convert the lists to NumPy arrays for further processing if needed
    mid_inlier_indices = np.array(mid_inlier_indices)
    mid_outlier_indices = np.array(mid_outlier_indices)
    
    local_inlier_indices = np.array(local_inlier_indices)
    local_outlier_indices = np.array(local_outlier_indices)
    
    global_inlier_indices = np.array(global_inlier_indices)
    global_outlier_indices = np.array(global_outlier_indices)
    
    # Save inlier indices to a text file
    mid_inlier_file_path = 'features/multi/lab/mid_inlier2.txt'
    np.savetxt(mid_inlier_file_path, mid_inlier_indices, fmt='%d')

    # Save outlier indices to a text file
    mid_outlier_file_path = 'features/multi/lab/mid_outlier2.txt'
    np.savetxt(mid_outlier_file_path, mid_outlier_indices, fmt='%d')
    
    # Save inlier indices to a text file
    local_inlier_file_path = 'features/multi/lab/local_inlier2.txt'
    local_outlier_file_path = 'features/multi/lab/local_outlier2.txt'
    
    np.savetxt(local_inlier_file_path, local_inlier_indices, fmt='%d')
    np.savetxt(local_outlier_file_path, local_outlier_indices, fmt='%d')
    
    global_inlier_file_path = 'features/multi/lab/global_inlier2.txt'
    global_outlier_file_path = 'features/multi/lab/global_outlier2.txt'
    
    np.savetxt(global_inlier_file_path, global_inlier_indices, fmt='%d')
    np.savetxt(global_outlier_file_path, global_outlier_indices, fmt='%d')