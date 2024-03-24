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

def trans_pcd(pcd, translate):
    pcd.translate(translate)
    return np.asarray(pcd.points)


if __name__=='__main__' :
    vis_type= 'all'
    voxel_size = 0.025

    local_path = 'npz/local16_7-scenes-redkitchen_004.npz'
    mid_path = 'npz/mid16_7-scenes-redkitchen_004.npz'
    global_path = 'npz/global_7-scenes-redkitchen_004.npz'
    local_file = np.load(local_path)
    mid_file = np.load(mid_path)
    global_file = np.load(global_path)
    local_points, mid_points, global_points = local_file['xyz'], mid_file['xyz'], global_file['xyz']
    local_feats, mid_feats, global_feats = local_file['feature'], mid_file['feature'], global_file['feature']
    
    translate_local = [-4., 0., 0.]
    translate_global = [4., 0., 0.]
    
    local_pcd = make_open3d_point_cloud(local_points)
    global_pcd = make_open3d_point_cloud(global_points)
    
    local_points_trans = trans_pcd(local_pcd, translate_local)
    global_points_trans = trans_pcd(global_pcd, translate_global)


    points = {
        'local': local_points,
        'mid': mid_points,
        'global': global_points,
        'all' : np.concatenate([local_points_trans, mid_points, global_points_trans])
    }[vis_type]

    features = {
        'local': local_feats,
        'mid': mid_feats,
        'global': global_feats,
        'all': np.concatenate([local_feats, mid_feats, global_feats])
    }[vis_type]

    points_tsne = embed_tsne(features)
    color = get_color_map(points_tsne)

    pcd_o3d = make_open3d_point_cloud(points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    mesh_o3d = mesh_sphere(pcd_o3d, voxel_size)
    # o3d.visualization.draw_geometries([pcd_o3d])
    o3d.io.write_triangle_mesh('visualization_concat_results/multi_7-scenes-redkitchen_4.ply', mesh_o3d)
    # o3d.visualization.draw_geometries([mesh_o3d])
    
