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
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='features/KITTI_0.025', help='path to the root directory')
    parser.add_argument('--traj_dir', type=str, default='/root/dataset/threedmatch_test/7-scenes-redkitchen-evaluation/3dmatch.log', help='path to the root directory')
    parser.add_argument('--subject_name', type=str, default='7-scenes-redkitchen', help='subject name')
    parser.add_argument('--voxel_size', type=float, default=0.025, help='voxel size')
    parser.add_argument('--inlier_thresh', type=float, default=0.1, help='inlier threshold')
    args = parser.parse_args()
    
    root_dir = args.root_dir
    subject_name = args.subject_name
        
    # traj_path = os.path.join(root_dir, subject_name + ".log")
    traj_path = args.traj_dir
    traj_list = read_trajectory(traj_path)
    
    for m in range(len(traj_list)):
      traj_values = traj_list[m].pose

      # Convert the list of lists to a NumPy array
      traj = np.array(traj_values)
      trans_gth = np.linalg.inv(traj)
      
      source_idx = traj_list[m].metadata[0]
      source_path = os.path.join(root_dir, subject_name + "_%03d.npz" % source_idx)
      
      target_idx = traj_list[m].metadata[1]
      target_path = os.path.join(root_dir, subject_name + "_%03d.npz" % target_idx)

      source_data = np.load(source_path)
      target_data = np.load(target_path)
      
      coords_i, coords_j = source_data['xyz'], target_data['xyz']
      feats_i, feats_j = source_data['feature'], target_data['feature']
      pcd_i = make_open3d_point_cloud(coords_i)
      pcd_j = make_open3d_point_cloud(coords_j)
      
      pcd_i.transform(trans_gth)

      inds = find_nn_cpu(feats_i, feats_j, return_distance=False)    
      transformed_coords_i = np.asarray(pcd_i.points)

      inlier_indices = []
      outlier_indices = []
      # threshold = 1.5 * 0.025  # 1.5 times the voxel size
      threshold = 0.1
      
      for i, point in enumerate(transformed_coords_i):
        dist = np.sqrt(np.sum((point - coords_j[inds[i]])**2))
        if dist < threshold:
          inlier_indices.append((i, inds[i]))
        else:
          outlier_indices.append((i, inds[i]))
    
      # Convert the lists to NumPy arrays for further processing if needed
      inlier_indices = np.array(inlier_indices)
      outlier_indices = np.array(outlier_indices)
      
      # Save inlier indices to a text file
      corr_set_dir = os.path.join(root_dir, subject_name + "_corr_sets")
      inlier_file_path = os.path.join(corr_set_dir, "%d_%d_inlier_indices.txt" % (source_idx, target_idx))
      np.savetxt(inlier_file_path, inlier_indices, fmt='%d')

      # Save outlier indices to a text file
      outlier_file_path = os.path.join(corr_set_dir, "%d_%d_outlier_indices.txt" % (source_idx, target_idx))
      np.savetxt(outlier_file_path, outlier_indices, fmt='%d')
      
      print(f"{source_idx}, {target_idx} corr set saved. Inlier: {len(inlier_indices)}, Outlier: {len(outlier_indices)}")    
      print(f"pcd_source: {len(pcd_i.points)}, pcd_target: {len(pcd_j.points)}")


    

