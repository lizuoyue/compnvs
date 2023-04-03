import numpy as np
import tqdm, random
from PIL import Image
import torch
import os, glob, sys
import MinkowskiEngine as ME
sys.path.append('/home/lzq/lzy/minkowski_3d_completion')
from minkowski_utils import replace_features

# Validation
# frl_apartment_0: 19 20 21
# apartment_1: 13 14
# office_2: 42

def load(idx, npz_type):
    npz = np.load(f'ReplicaGenEncFtTriplets/easy/{npz_type}/{idx}.npz', allow_pickle=True)
    # normed = npz['values'].copy()
    # normed[:,:8] = (normed[:,:8] + 200) / 400
    # normed[:,8:] = (normed[:,8:] + 50) / 100
    # normed = np.round(np.clip(normed, 0, 1) * 255.0).astype(np.uint8)
    # for i in range(4):
    #     with open(f'pts/{idx}_{i}.txt', 'w') as f:
    #         for pt, rgb in zip(npz['vertex'], normed[:,8*i:8*i+3]):
    #             f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + list(rgb)))
    idx = torch.from_numpy(npz['vertex'])
    idx = torch.cat([idx[:,:1] * 0, idx], dim=-1)
    valid = (np.abs(npz['values']).sum(axis=-1) > 1e-6).astype(np.float32)
    val = np.concatenate([npz['values'], valid[:,np.newaxis]], axis=-1)
    val = torch.from_numpy(val)
    return idx, val

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

        triplets = np.load(f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz')[stage]
        
        for k, i, j in tqdm.tqdm(triplets):

            basename = f'{sid:02d}_{k:03d}_{i:03d}_{j:03d}'

            idx_i, val_i = load(f'{sid:02d}_{i:03d}', 'npz_single')
            idx_j, val_j = load(f'{sid:02d}_{j:03d}', 'npz_single')
            idx_k, val_k = load(f'{sid:02d}_{k:03d}', 'npz_single')
            idx_tri, val_tri = load(f'{sid:02d}_{k:03d}_{i:03d}_{j:03d}', 'npz_triple')

            idx_ijk = torch.cat([idx_i, idx_j, idx_k], dim=0)
            # val_ijk = torch.cat([val_i, val_j, val_k], dim=0)
            val_ij0 = torch.cat([val_i, val_j, val_k*0], dim=0)
            # val_ij0ijk = torch.cat([val_ij0, val_ijk], dim=1)

            input_info = ME.SparseTensor(
                coordinates=idx_ijk,
                features=val_ij0,#val_ij0ijk,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            )

            output_info = ME.SparseTensor(
                coordinates=idx_tri,
                features=val_tri,
                coordinate_manager=input_info.coordinate_manager,
            )

            output_ft = output_info.features_at_coordinates(input_info.C.float())
            if input_info.C.shape[0] == idx_tri.shape[0] and torch.round(1-output_ft[:,-1]).int().sum().item() == 0:
                pass
            else:
                print(sid, i, j, k)

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
            vertex_input = input_info.features_at_coordinates(locations)
            vertex_input = vertex_input / torch.maximum(vertex_input[:,-1:], vertex_input[:,-1:]*0+1)
            vertex_output = output_info.features_at_coordinates(locations)[:, :-1]

            # src_rgb = f'ReplicaGenFt/all/rgb/{sid:02d}_{k:03d}.png'
            # tar_rgb = f'ReplicaGenEncFtTriplets/{stage}/rgb/{basename}.png'
            # src_dep = f'ReplicaGenFt/all/depth/{sid:02d}_{k:03d}.npz'
            # tar_dep = f'ReplicaGenEncFtTriplets/{stage}/depth/{basename}.npz'
            # src_pose = f'ReplicaGenFt/all/pose/{sid:02d}_{k:03d}.txt'
            # tar_pose = f'ReplicaGenEncFtTriplets/{stage}/pose/{basename}.txt'
            # os.system(f'cp {src_rgb} {tar_rgb}')
            # os.system(f'cp {src_dep} {tar_dep}')
            # os.system(f'cp {src_pose} {tar_pose}')

            #
            np.savez_compressed(
                f'ReplicaGenEncFtTriplets/{stage}/npz/{basename}.npz',
                center_pts=center_pts.numpy().astype(np.float32),
                center_to_vertex=center_to_vertex.numpy().astype(np.int32),
                vertex_idx=vertex_idx.numpy().astype(np.int32),
                vertex_input=vertex_input.numpy().astype(np.float32), # last channel mask
                vertex_output=vertex_output.numpy().astype(np.float32),
                reference=bboxes[sid][:3].astype(np.float32),
            )

