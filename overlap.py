import argparse
import numpy as np
import open3d as o3d
from util.pointcloud import compute_overlap_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s1',
        '--scene1',
        default='/root/dataset/threedmatch_test/7-scenes-redkitchen/cloud_bin_4.ply',
        type=str,
        help='path to a pointcloud file')
    parser.add_argument(
        '-s2',
        '--scene2',
        default='/root/dataset/threedmatch_test/7-scenes-redkitchen/cloud_bin_11.ply',
        type=str,
        help='path to a pointcloud file')
    
    config = parser.parse_args()

    traj_values = [
        [0.9835029850, -0.1286556230, -0.1271597750, 0.6103462380],
        [0.1358681670, 0.9894775360, 0.0497397823, 0.2013524550],
        [0.1194224380, -0.0661961900, 0.9906343150, -0.0446159691],
        [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]
    ]

    traj = np.array(traj_values)
    traj = np.linalg.inv(traj)

    pcd0 = o3d.io.read_point_cloud(config.scene1)
    pcd1 = o3d.io.read_point_cloud(config.scene2)
    overlap_ratio = compute_overlap_ratio(pcd0, pcd1, traj, 0.025)
    print(overlap_ratio)