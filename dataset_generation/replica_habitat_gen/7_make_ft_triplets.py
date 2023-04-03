import numpy as np
import tqdm, random
from PIL import Image
from utils import OccupancyGridMultiDim
import torch
import os, glob, sys

# Validation
# frl_apartment_0: 19 20 21
# apartment_1: 13 14
# office_2: 42

if __name__ == '__main__':

    stage = 'easy'
    n_scene = 48
    voxel_size = 0.1
    spatial_size = [144, 64, 160] # center
    spatial_size = [item + 1 for item in spatial_size]
    offset = np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], np.int)
    bboxes = np.loadtxt(f'ReplicaGen/bboxes.txt')

    ndim = 16
    
    a, b = int(sys.argv[1]), int(sys.argv[2])
    # ckpt = np.load('ReplicaGen/multi_scene_nsvf.npz', allow_pickle=True)
    # ckpt = np.load('ReplicaGen/reg_multi_scene_nsvf_dim32.npz', allow_pickle=True)
    # ckpt = np.load('ReplicaGen/reg_multi_scene_nsvf_dim4in.npz', allow_pickle=True)
    # ckpt = np.load('ReplicaGen/multi_scene_nsvf_rgba_field_zero_init.npz', allow_pickle=True)
    ckpt = np.load('ReplicaGen/multi_scene_nsvf_rgba_init.npz', allow_pickle=True)
    for sid in range(a, b):
        if sid not in [13, 14, 19, 20, 21, 42]:
            continue

        mapping = np.load(f'ReplicaGenRelation/scene{sid:02d}.npz')['mapping']
        triplets = np.load(f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz')[stage]

        center_point = ckpt[f'center_point_{sid:02d}']
        center_point_np = np.loadtxt(f'ReplicaGen/scene{sid:02d}/init_voxel/center_points.txt')
        assert(np.abs(center_point_np - center_point).mean() < 5e-6)

        center_keep = ckpt[f'center_keep_{sid:02d}']
        center_to_vertex = ckpt[f'center_to_vertex_{sid:02d}']
        vertex_feature = ckpt[f'vertex_feature_{sid:02d}']

        # Compute vertex points
        vertex_point = np.repeat(center_point, 8, axis=0) + np.tile(offset * 0.5 * voxel_size, (center_point.shape[0], 1))
        vertex_point_x10 = np.round(vertex_point * 10).astype(np.int32)
        assert(np.abs(vertex_point - vertex_point_x10 / 10).mean() < 5e-6)
        vertex_point_x10 = np.concatenate([center_to_vertex.reshape((-1, 1)), vertex_point_x10], axis=-1)
        vertex_point_x10 = np.unique(vertex_point_x10, axis=0)
        assert(vertex_point_x10.shape[0] == vertex_feature.shape[0])
        vertex_point = np.zeros((vertex_point_x10.shape[0], 3), np.float32)
        vertex_point[vertex_point_x10[:, 0]] = vertex_point_x10[:, 1:].astype(np.float32)
        vertex_point /= 10

        # vertex_index_raw = (vertex_point - bboxes[sid][:3]) / voxel_size
        # vertex_index = np.round(vertex_index_raw).astype(np.int32)
        # assert(np.abs(vertex_index - vertex_index_raw).mean() < 1e-6)

        ft_grid = OccupancyGridMultiDim(spatial_size, ndim, voxel_size, bboxes[sid][:3])
        ft_grid.set_occupancy_by_coord(vertex_point, vertex_feature)
        
        for k, i, j in tqdm.tqdm(triplets):

            basename = f'{sid:02d}_{k:03d}_{i:03d}_{j:03d}'

            has_i = mapping[i] > 0
            has_j = mapping[j] > 0
            has_k = mapping[k] > 0

            center_pts = center_point[has_i | has_j | has_k]
            center_idx = (center_pts - bboxes[sid][:3]) / voxel_size # .5
            vertex_idx_raw = np.repeat(center_idx, 8, axis=0) + np.tile(offset * 0.5, (center_idx.shape[0], 1))
            vertex_idx = np.round(vertex_idx_raw).astype(np.int32)
            assert(np.abs(vertex_idx - vertex_idx_raw).mean() < 5e-6)
            vertex_idx, c2v = np.unique(vertex_idx, axis=0, return_inverse=True)
            c2v = c2v.reshape((-1, 8))

            ij_vertex_mask = np.unique(center_to_vertex[has_i | has_j].flatten())
            ft_grid_ij = OccupancyGridMultiDim(spatial_size, ndim+1, voxel_size, bboxes[sid][:3])
            ft_grid_ij.set_occupancy_by_coord(
                vertex_point[ij_vertex_mask],
                np.concatenate([vertex_feature, vertex_feature[:,:1]*0+1], axis=-1)[ij_vertex_mask],
            )

            input_pc = ft_grid_ij.get_occupancy_status_by_index(vertex_idx)
            output_pc = ft_grid.get_occupancy_status_by_index(vertex_idx)
            assert((np.abs(output_pc).sum(axis=-1) < 1e-6).sum() == 0)
            mask = input_pc[..., -1].astype(np.bool)
            assert(np.abs(input_pc[mask, :-1] - output_pc[mask]).mean() < 1e-6)
            assert(np.abs(input_pc[~mask, :-1]).mean() < 1e-6)

            # src_rgb = f'ReplicaGen/scene{sid:02d}/rgb/{k:03d}.png'
            # tar_rgb = f'ReplicaGenFtTriplets/{stage}/rgb/{basename}.png'
            # src_pose = f'ReplicaGen/scene{sid:02d}/pose/{k:03d}.txt'
            # tar_pose = f'ReplicaGenFtTriplets/{stage}/pose/{basename}.txt'
            # os.system(f'cp {src_rgb} {tar_rgb}')
            # os.system(f'cp {src_pose} {tar_pose}')
            np.savez_compressed(
                f'ReplicaGenFtTriplets/{stage}/npz_rgba_init/{basename}.npz',
                center_pts=center_pts.astype(np.float32),
                center_to_vertex=c2v.astype(np.int32),
                vertex_idx=vertex_idx.astype(np.int32),
                vertex_input=input_pc.astype(np.float32), # last channel mask
                vertex_output=output_pc.astype(np.float32),
                reference=bboxes[sid][:3].astype(np.float32),
            )

            if False:
                with open(f'pts/center_pts.txt', 'w') as f:
                    for pt in center_pts:
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                with open(f'pts/vertex_pts_0.txt', 'w') as f:
                    for pt in vertex_pts[input_pc[:,-1] < 0.1]:
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [127,127,127]))
                with open(f'pts/vertex_pts_1.txt', 'w') as f:
                    for pt, (r, g, b, _) in zip(vertex_pts[input_pc[:,-1] > 0.1], input_pc[input_pc[:,-1] > 0.1]):
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
                quit()
            
            
