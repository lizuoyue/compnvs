import os, h5py, tqdm
import numpy as np
import torch

if __name__ == '__main__':

    voxel_size = 0.1
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

    ckpt_path = 'multi_scene_nsvf_basev1_10cm/checkpoint_47_234000.pt'
    # ckpt_path = 'multi_scene_nsvf_basev1_10cm_refine/checkpoint_last.pt'
    ckpt = torch.load(ckpt_path)
    tar_ckpt_path = 'multi_scene_nsvf_basev1_10cm_refine/save/checkpoint_last.pt'
    tar_ckpt = torch.load(tar_ckpt_path)

    for scene_idx in range(45):

        center_point = np.loadtxt(f'scene{scene_idx:02d}/init_voxel/fine_points_0.1.txt')
        reference = center_point.min(axis=0)
        center_index = np.round((center_point - reference) / voxel_size) + 0.5

        vertex_index = np.repeat(center_index, 8, axis=0) + np.tile(offset * 0.5, (center_index.shape[0], 1))
        vertex_index = vertex_index.astype(np.int)

        vertex_index, center_to_vertex = np.unique(vertex_index, return_inverse=True, axis=0)
        center_to_vertex = center_to_vertex.reshape((center_index.shape[0], 8))

        vertex_ft = np.zeros((vertex_index.shape[0], 32), np.float32)
        vertex_ft_mapping = -np.ones((vertex_index.shape[0]), np.int32)

        center_keep = np.ones((center_to_vertex.shape[0]), np.int64)

        # li = ['points', 'feats', 'keep', 'values.weight', 'keys', 'num_keys']
        # print(f'scene{scene_idx:02d}')
        # for key in li:
        #     print(key.ljust(15), tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.{key}'].shape)
        
        diff = center_to_vertex - tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.feats'].numpy()
        assert(np.abs(diff).max() == 0)

        # 
        center_point_ckpt = ckpt['model'][f'encoder.all_voxels.{scene_idx}.points'].numpy()
        center_to_vertex_ckpt = ckpt['model'][f'encoder.all_voxels.{scene_idx}.feats'].numpy()
        vertex_ft_ckpt = ckpt['model'][f'encoder.all_voxels.{scene_idx}.values.weight'].numpy()
        center_keep_ckpt = ckpt['model'][f'encoder.all_voxels.{scene_idx}.keep'].numpy().astype(np.int64)
        center_index_ckpt = np.round((center_point_ckpt - reference) / voxel_size).astype(np.int)
        d = {}
        for i, (x, y, z) in enumerate(center_index_ckpt):
            d[(x, y, z)] = i

        # 
        assert(center_index.shape[0] == center_to_vertex.shape[0])
        num_no_point = 0
        for iii, (ci, c2v) in enumerate(zip(center_index, center_to_vertex)):
            x, y, z = (ci - 0.5).astype(np.int)
            if (x, y, z) in d:
                ci_ckpt = d[(x, y, z)]
                c2v_ckpt = center_to_vertex_ckpt[ci_ckpt]
                vertex_ft[c2v] = vertex_ft_ckpt[c2v_ckpt]
                for c2vi, c2vi_ckpt in zip(c2v, c2v_ckpt):
                    if vertex_ft_mapping[c2vi] < 0:
                        vertex_ft_mapping[c2vi] = c2vi_ckpt
                    else:
                        assert(vertex_ft_mapping[c2vi] == c2vi_ckpt)
                center_keep[iii] = center_keep_ckpt[ci_ckpt]
            else:
                num_no_point += 1

        tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.values.weight'] *= 0
        tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.values.weight'] += torch.from_numpy(vertex_ft)
        tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.keep'] = torch.from_numpy(center_keep)
        tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.max_hits'] += 30
        print(f'scene{scene_idx:02d}')
        print('    center', 1 - num_no_point / center_index.shape[0])
        print('    vertex', (vertex_ft_mapping >= 0).mean())
        print('    keep  ', tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.keep'].numpy().mean())
        print('    step  ', tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.step_size'].numpy())
        print('    hit   ', tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.max_hits'].numpy())
        print('    vs    ', tar_ckpt['model'][f'encoder.all_voxels.{scene_idx}.voxel_size'].numpy())

        # with open(f'center_point_00.txt', 'w') as f:
        #     for pt in center_point:
        #         f.write('%.6lf %.6lf %.6lf\n' % tuple(list(pt)))
        
        # with open(f'center_point_ckpt_00.txt', 'w') as f:
        #     for pt in center_point_ckpt:
        #         f.write('%.6lf %.6lf %.6lf\n' % tuple(list(pt)))
    
    for key in ckpt['model'].keys():
        if key.startswith('field'):
            tar_ckpt['model'][key] = ckpt['model'][key]
    
    torch.save(tar_ckpt, 'multi_scene_nsvf_basev1_10cm_refine/checkpoint_last.pt')
