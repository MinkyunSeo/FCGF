import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from util.visualization import get_colored_point_cloud_feature
from util.misc import extract_features

from model.resunet import ResUNetBN2C

import torch

if not os.path.isfile('ResUNetBN2C-16feat-3conv.pth'):
  print('Downloading weights...')
  urlretrieve(
      "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
      'ResUNetBN2C-16feat-3conv.pth')

if not os.path.isfile('redkitchen-20.ply'):
  print('Downloading a mesh...')
  urlretrieve("https://node1.chrischoy.org/data/publications/fcgf/redkitchen-20.ply",
              'redkitchen-20.ply')


def feature_to_mesh(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config.model)
    
    # 3D Match Pretrained
    # model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
    # KITTI Pretrained
    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=5, D=3)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    pcd = o3d.io.read_point_cloud(config.input)
    xyz_down, feature = extract_features(
        model,
        xyz=np.array(pcd.points),
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True)

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(xyz_down)

    vis_pcd = get_colored_point_cloud_feature(vis_pcd,
                                            feature.detach().cpu().numpy(),
                                            config.voxel_size)
    
    # Get the base directory name
    base_dir_name = os.path.basename(os.path.normpath(config.input))
    parent_dir_name = os.path.basename(os.path.normpath(os.path.dirname(config.input)))
    file_number = base_dir_name.split('_')[-1].split('.')[0]
    
    model_name = config.model_type
    file_name = f"{parent_dir_name}_{file_number}_{model_name}.ply"

    # Construct the full path
    path = os.path.join('visualization_results', file_name)
    o3d.io.write_triangle_mesh(path, vis_pcd)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--input',
      default='redkitchen-20.ply',
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
