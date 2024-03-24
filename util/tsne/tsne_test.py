import os
import numpy as np
import argparse
import open3d as o3d
import math
from urllib.request import urlretrieve
from util.visualization import get_colored_point_cloud_feature
from util.misc import extract_features

from model.resunet import ResUNetBN2C

import torch

def feature_to_mesh(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config.model)
    
    conv1_size_cube = checkpoint['state_dict']['conv1.kernel'].size()[0]
    conv1_size = round(math.pow(conv1_size_cube, 1/3))
    out_channel_size = checkpoint['state_dict']['final.kernel'].size()[-1]
    
    model = ResUNetBN2C(in_channels=1, out_channels=out_channel_size, normalize_feature=True, conv1_kernel_size=conv1_size, D=3)
    # model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    pcd_i = make_open3d_point_cloud(config.scene1)
    pcd_i = o3d.io.read_point_cloud(config.scene1)
    pcd_j = o3d.io.read_point_cloud(config.scene2)
    
    xyz_down_i, feature_i = extract_features(
        model,
        xyz=np.array(pcd_i.points),
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True)
    
    xyz_down_j, feature_j = extract_features(
        model,
        xyz=np.array(pcd_j.points),
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True)
    

    # Get the base directory name
    base_dir_name_i = os.path.basename(os.path.normpath(config.scene1))
    parent_dir_name = os.path.basename(os.path.normpath(os.path.dirname(config.scene1)))
    file_number_i = base_dir_name_i.split('_')[-1].split('.')[0]
    
    model_name = config.model_type
    file_name_i = f"{parent_dir_name}_{file_number_i}_{model_name}.ply"

    # Construct the full path
    path_i = os.path.join('visualization_concat_results', file_name_i)
    # o3d.io.write_triangle_mesh(path_i, vis_pcd_i)

    base_dir_name_j = os.path.basename(os.path.normpath(config.scene2))
    file_number_j = base_dir_name_j.split('_')[-1].split('.')[0]
    file_name_j = f"{parent_dir_name}_{file_number_j}_{model_name}.ply"
    path_j = os.path.join('visualization_concat_results', file_name_j)
    # o3d.io.write_triangle_mesh(path_j, vis_pcd_j)

    
    np.savez_compressed(
        os.path.join('visualization_concat_results', f"{parent_dir_name}_{file_number_i}_{model_name}.npz"),
        xyz=xyz_down_i,
        feature=feature_i.detach().cpu().numpy(),
        points=pcd_i.points
    )
    
    np.savez_compressed(
        os.path.join('visualization_concat_results', f"{parent_dir_name}_{file_number_j}_{model_name}.npz"),
        xyz=xyz_down_j,
        feature=feature_j.detach().cpu().numpy(),
        points=pcd_j.points
    )

        
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-s1',
      '--scene1',
      default='/root/dataset/threedmatch_test/7-scenes-redkitchen/cloud_bin_0.ply',
      type=str,
      help='path to a pointcloud file')
  parser.add_argument(
      '-s2',
      '--scene2',
      default='/root/dataset/threedmatch_test/7-scenes-redkitchen/cloud_bin_2.ply',
      type=str,
      help='path to a pointcloud file')
  parser.add_argument(
      '-m',
      '--model',
      default='ResUNetBN2C-16feat-3conv.pth',
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '-mt',
      '--model_type',
      default='3DMatch',
      type=str,
      choices=['3DMatch', 'KITTI'],  # Restrict choices to '3DMatch' or 'KITTI'
      help='type of the model (default: 3DMatch)')
  parser.add_argument(
      '--voxel_size',
      default=0.025,
      type=float,
      help='voxel size to preprocess point cloud')

  config = parser.parse_args()
  feature_to_mesh(config)
