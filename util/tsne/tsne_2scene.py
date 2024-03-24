import numpy as np
import open3d as o3d
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import time
from util.visualization import embed_tsne, get_color_map, mesh_sphere
from util.pointcloud import make_open3d_point_cloud

def align_pcd(pcd, pose, translate):
    traj = np.linalg.inv(pose)
    pcd.transform(traj)
    pcd.translate(translate)
    return np.asarray(pcd.points)


if __name__=='__main__' :
    vis_type= 'both'
    voxel_size = 0.025

    ref_path = 'npz/mid_7-scenes-redkitchen_004.npz'
    src_path = 'npz/mid_7-scenes-redkitchen_011.npz'
    pose_path = 'npz/0_2_gt_pose.npz'
    ref_file = np.load(ref_path)
    src_file = np.load(src_path)
    pose_file = np.load(pose_path)
    ref_points, src_points = ref_file['xyz'], src_file['xyz']
    ref_feats, src_feats = ref_file['feature'], src_file['feature']
    # gt_pose = pose_file['pose']
    translate = [4., 0., 0.]
    
    gt_pose = [
        [0.9835029850, -0.1286556230, -0.1271597750, 0.6103462380],
        [0.1358681670, 0.9894775360, 0.0497397823, 0.2013524550],
        [0.1194224380, -0.0661961900, 0.9906343150, -0.0446159691],
        [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]
    ]
        
    src_pcd = make_open3d_point_cloud(src_points)
    src_points_trans = align_pcd(src_pcd, gt_pose, translate)

    points = {
        'ref': ref_points,
        'src': src_points,
        'both' : np.concatenate([ref_points, src_points_trans])
    }[vis_type]

    features = {
        'ref': ref_feats,
        'src': src_feats,
        'both': np.concatenate([ref_feats, src_feats])
    }[vis_type]

    points_tsne = embed_tsne(features)
    color = get_color_map(points_tsne)

    pcd_o3d = make_open3d_point_cloud(points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    mesh_o3d = mesh_sphere(pcd_o3d, voxel_size)
    # o3d.visualization.draw_geometries([pcd_o3d])
    o3d.io.write_triangle_mesh('visualization_concat_results/mid_7-scenes-redkitchen_4_11_3DMatch.ply', mesh_o3d)
    # o3d.visualization.draw_geometries([mesh_o3d])
    
