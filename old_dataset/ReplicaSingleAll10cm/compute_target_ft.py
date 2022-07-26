import os
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import pytorch3d

def write_pc(filename, coord, color):
    with open(filename, 'w') as f:
        for p, c in zip(coord, color):
            x, y, z = p
            r, g, b = (c * 255).astype(np.int)
            f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
    return

if __name__ == '__main__':

    # a0 = \
    # torch.nn.functional.grid_sample(
    #     input=torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
    #     grid=2/3*torch.Tensor([[[[-1, -1], [-1, 0], [-1, 1]], [[0, -1], [0, 0], [0, 1]], [[1, -1], [1, 0], [1, 1]]]]),
    # mode='bilinear', padding_mode='zeros', align_corners=False)

    # a1 = \
    # torch.nn.functional.grid_sample(
    #     input=torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
    #     grid=torch.Tensor([[[[-1, -1], [-1, 0], [-1, 1]], [[0, -1], [0, 0], [0, 1]], [[1, -1], [1, 0], [1, 1]]]]),
    # mode='bilinear', padding_mode='zeros', align_corners=True)

    # print(a0)
    # print(a1)

    # quit()

    os.system('mkdir all')
    os.system('mkdir all/target_ft')

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
    ckpt = torch.load(f'../ReplicaMultiScene/multi_scene_nsvf_basev1_10cm_refine/checkpoint_45_224000.pt')

    for sid in tqdm.tqdm(list(range(45))):

        voxel = {
            'points': ckpt['model'][f'encoder.all_voxels.{sid}.points'],
            'feats': ckpt['model'][f'encoder.all_voxels.{sid}.feats'],
            'values.weight': ckpt['model'][f'encoder.all_voxels.{sid}.values.weight'],
        }
        
        # ['points', 'feats', 'keep', 'values.weight']
        center_point = voxel['points'].numpy()
        reference = center_point.min(axis=0)
        center_index = np.round((center_point - reference) / voxel_size) + 0.5
        vertex_index = np.repeat(center_index, 8, axis=0) + np.tile(offset * 0.5, (center_index.shape[0], 1))
        vertex_index = vertex_index.astype(np.int)
        x, y, z = vertex_index.max(axis=0) + 1

        lb = reference - 0.5 * voxel_size
        lk = 2.0 / (vertex_index.max(axis=0) * voxel_size)

        vertex_index = vertex_index.reshape((-1, 8, 3))
        center_to_vertex = voxel['feats'].numpy()

        mapping = {}
        scene_pt = voxel['values.weight'].numpy()[:, :3] * 0
        for idx3d, idx1d in zip(vertex_index, center_to_vertex):
            for i in range(8):
                k = int(idx1d[i])
                if k in mapping:
                    assert(mapping[k] == tuple(list(idx3d[i])))
                else:
                    mapping[k] = tuple(list(idx3d[i]))
                    scene_pt[k] = (idx3d[i] - 0.5) * voxel_size + reference
        
        feature_list = voxel['values.weight']
        c = feature_list.shape[-1]
        grid = torch.zeros(x, y, z, c + 1).float()
        for k in mapping:
            xx, yy, zz = mapping[k]
            grid[xx, yy, zz, :c] = feature_list[k]
            grid[xx, yy, zz, c] = 1
        grid = grid.permute([3, 2, 1, 0]).unsqueeze(0)

        # pca = PCA(n_components=3)
        # pca.fit(feature_list)

        # vis = pca.transform(feature_list)
        # vmin = vis.min(axis=0)
        # vmax = vis.max(axis=0)
        # vis = (vis - vmin) / (vmax - vmin)

        # write_pc(f'source_{sid:02d}.txt', scene_pt, vis)

        for i in range(sid * 250, (sid + 1) * 250):

            pose = np.loadtxt(f'all/pose_global/0_{i:05d}.txt')
            data = np.load(f'all/npz/0_{i:05d}.npz')

            # print(data['voxel_pts'].shape)
            # print(data['voxel_to_vertex'].shape)
            # print(data['vertex_info'].shape)
            vertex_pt = data['vertex_idx'] * voxel_size + data['reference']
            global_vertex_pt = np.concatenate([vertex_pt, vertex_pt[:, 0:1] * 0 + 1], axis=1).dot(pose.T)[:, :3]
            global_vertex_pt = vertex_pt

            # print(global_vertex_pt.min(axis=0))
            # print(global_vertex_pt.max(axis=0))

            global_vertex_pt_sample = (global_vertex_pt - lb) * lk - 1
            # print(global_vertex_pt_sample.min(axis=0))
            # print(global_vertex_pt_sample.max(axis=0))

            global_vertex_pt_sample = torch.from_numpy(global_vertex_pt_sample).float()
            global_vertex_pt_sample = global_vertex_pt_sample.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # print(global_vertex_pt_sample.shape)

            ft = F.grid_sample(grid, global_vertex_pt_sample, mode='bilinear', padding_mode='border', align_corners=True)
            ft = ft.numpy()[0, :, 0, 0].transpose()
            ft_valid = ft[:, -1:] > 1e-9
            ft = ft[:, :-1] / np.maximum(ft[:, -1:], 1e-9)

            np.savez_compressed(
                f'all/target_ft/0_{i:05d}.npz',
                vertex_feature=ft.astype(np.float32),
                vertex_feature_valid=ft_valid,
            )
            continue

            ft_vis = pca.transform(ft)
            ft_vis = (ft_vis - vmin) / (vmax - vmin)

            write_pc(f'target_{sid:02d}_{i:02d}_1.txt', global_vertex_pt, ft_vis)

            quit()
        
        quit()

        


