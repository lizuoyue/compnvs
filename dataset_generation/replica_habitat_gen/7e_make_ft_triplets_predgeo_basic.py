import numpy as np
import tqdm, random
from PIL import Image
import torch
import os, glob, sys
import MinkowskiEngine as ME
sys.path.append('/home/lzq/lzy/minkowski_3d_completion')
from minkowski_utils import replace_features
from utils import project_depth
import scipy.ndimage as mor

# Validation
# frl_apartment_0: 19 20 21
# apartment_1: 13 14
# office_2: 42

offset_np = np.array([
    [-1,-1,-1],
    [-1,-1, 1],
    [-1, 1,-1],
    [-1, 1, 1],
    [ 1,-1,-1],
    [ 1,-1, 1],
    [ 1, 1,-1],
    [ 1, 1, 1],
], np.int)
voxel_size = 0.1

def load(idx, npz_type):
    npz = np.load(f'ReplicaGenFt/all/{npz_type}/{idx}.npz', allow_pickle=True)
    pts = npz['pts'] / voxel_size + 0.5
    rgb = npz['rgb'].astype(np.float32)
    coords, feats = ME.utils.sparse_collate([pts], [rgb])
    x = ME.SparseTensor(
        coordinates=coords,
        features=feats,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
    )
    npz = np.load(f'ReplicaGenEncFtTriplets/easy/npz_single/{idx}.npz', allow_pickle=True)
    idx = torch.from_numpy(npz['vertex'])
    idx = torch.cat([idx[:,:1] * 0, idx], dim=-1)
    return x, idx


def load_pred(idx, scene_min, scene_max):
    pred_geo = np.load(f'/home/lzq/lzy/minkowski_3d_completion/mink_comp_pred_w1/{idx}.npz')['pred_geo']
    # pred_coords = (pred_geo + 0.5) * 0.1
    # mask = ((scene_min - 0.2) <= pred_coords) & (pred_coords <= (scene_max + 0.2))
    # mask = mask[:,0] & mask[:,1] & mask[:,2]
    # pred_geo = pred_geo[mask]

    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    kernel_x = kernel[np.newaxis]
    kernel_y = kernel[:,np.newaxis]
    kernel_z = kernel[:,:,np.newaxis]

    pred_min, pred_max = pred_geo.min(axis=0) - 5, pred_geo.max(axis=0) + 5
    pred_biased = pred_geo - pred_min
    dense_grid = np.zeros(pred_max - pred_min + 1, np.bool)
    dense_grid[pred_biased[:,0], pred_biased[:,1], pred_biased[:,2]] = True
    dense_grid_x = mor.binary_closing(dense_grid, structure=kernel_x, iterations=3)
    dense_grid_y = mor.binary_closing(dense_grid, structure=kernel_y, iterations=3)
    dense_grid_z = mor.binary_closing(dense_grid, structure=kernel_z, iterations=3)
    dense_grid = dense_grid_x | dense_grid_y | dense_grid_z

    # np.savetxt(f'closing_test/{idx}_pre.txt', pred_geo, '%d')
    # np.savetxt(f'closing_test/{idx}_postx.txt', np.stack(dense_grid_x.nonzero(), axis=1)+pred_min, '%d')
    # np.savetxt(f'closing_test/{idx}_posty.txt', np.stack(dense_grid_y.nonzero(), axis=1)+pred_min, '%d')
    # np.savetxt(f'closing_test/{idx}_postz.txt', np.stack(dense_grid_z.nonzero(), axis=1)+pred_min, '%d')
    # input()
    pred_geo = np.stack(dense_grid.nonzero(), axis=1) + pred_min

    vertex_idx = np.repeat(pred_geo * 2 + 1, 8, axis=0) + np.tile(offset_np, (pred_geo.shape[0], 1)) # all voxel vertices index, always even
    vertex_idx = (np.unique(vertex_idx, axis=0) / 2).astype(np.int32)
    idx = torch.from_numpy(vertex_idx)
    idx = torch.cat([idx[:,:1] * 0, idx], dim=-1)
    return idx






if __name__ == '__main__':

    stage = 'easy'
    n_scene = 48
    voxel_size = 0.1
    bboxes = np.loadtxt(f'ReplicaGen/bboxes.txt')
    a, b = int(sys.argv[1]), int(sys.argv[2])

    pool = ME.MinkowskiSumPooling(kernel_size=2, stride=1, dimension=3)

    offset = torch.from_numpy(np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], np.int))

    for sid in range(a, b):

        scene_pts = np.loadtxt(f'ReplicaGen/scene{sid:02d}/init_voxel/center_points.txt')
        scene_min = scene_pts.min(axis=0)
        scene_max = scene_pts.max(axis=0)

        triplets = np.load(f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz')[stage]
        
        for k, i, j in tqdm.tqdm(triplets):

            basename = f'{sid:02d}_{k:03d}_{i:03d}_{j:03d}'

            x_i, idx_i = load(f'{sid:02d}_{i:03d}', 'npz')
            x_j, idx_j = load(f'{sid:02d}_{j:03d}', 'npz')
            x_ij = ME.SparseTensor(
                coordinates=torch.cat([x_i.C, x_j.C], dim=0),
                features=torch.cat([x_i.F, x_j.F], dim=0),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            )
            # np.savetxt('tst.txt',np.concatenate([
            #     x_ij.C.numpy().astype(np.int32)[:,1:],
            #     x_ij.F.numpy().astype(np.int32),
            # ],axis=1), '%d')

            idx_k = load_pred(f'{sid:02d}_{k:03d}_{i:03d}_{j:03d}', scene_min, scene_max)
            idx_ijk = torch.cat([idx_i, idx_j, idx_k], dim=0)

            input_info = ME.SparseTensor(
                coordinates=idx_ijk,
                features=torch.zeros_like(idx_ijk).float(),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            )
            # np.savetxt('tst1.txt',np.concatenate([
            #     input_info.C.numpy().astype(np.int32)[:,1:],
            #     input_info.C.numpy().astype(np.int32)[:,1:]*0,
            # ],axis=1), '%d')

            mink_ones = replace_features(input_info, input_info.F[:,:1] * 0 + 1)
            center_nums = pool(mink_ones)
            center_idx = center_nums.C[torch.round(center_nums.F[:,0]) == 8, 1:].float() + 0.5 # half stride

            # TODO: Here batch size do not support
            vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + 0.5 * offset.repeat(center_idx.shape[0], 1)
            vertex_idx = torch.round(vertex_idx).int()
            vertex_idx, center_to_vertex = torch.unique(vertex_idx, dim=0, return_inverse=True)
            
            center_to_vertex = center_to_vertex.reshape(-1, 8)
            center_pts = center_idx * 0.1

            locations = torch.cat([vertex_idx[:,:1]*0, vertex_idx], dim=1).float()
            vertex_input = x_ij.features_at_coordinates(locations)
            # vertex_input = vertex_input / torch.maximum(vertex_input[:,-1:], vertex_input[:,-1:]*0+1)

            # np.savetxt('tst2.txt',np.concatenate([
            #     vertex_idx.numpy().astype(np.int32),
            #     vertex_input.numpy().astype(np.int32),
            # ],axis=1), '%d')
            # input()

            # src_rgb = f'ReplicaGenFt/all/rgb/{sid:02d}_{k:03d}.png'
            # tar_rgb = f'ReplicaGenEncFtTriplets/{stage}/rgb/{basename}.png'
            # src_dep = f'ReplicaGenFt/all/depth/{sid:02d}_{k:03d}.npz'
            # tar_dep = f'ReplicaGenEncFtTriplets/{stage}/depth/{basename}.npz'
            # src_pose = f'ReplicaGenFt/all/pose/{sid:02d}_{k:03d}.txt'
            # tar_pose = f'ReplicaGenEncFtTriplets/{stage}/pose/{basename}.txt'
            # os.system(f'cp {src_rgb} {tar_rgb}')
            # os.system(f'cp {src_dep} {tar_dep}')
            # os.system(f'cp {src_pose} {tar_pose}')

            np.savez_compressed(
                f'ReplicaGenEncFtTriplets/{stage}/npz_basic_pred/{basename}.npz',
                center_pts=center_pts.numpy().astype(np.float32),
                center_to_vertex=center_to_vertex.numpy().astype(np.int32),
                vertex_idx=vertex_idx.numpy().astype(np.int32),
                vertex_input=vertex_input.numpy().astype(np.float32), # last channel mask
            )

