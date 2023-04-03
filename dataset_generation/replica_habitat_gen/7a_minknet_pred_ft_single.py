import MinkowskiEngine as ME
import os, sys, tqdm
sys.path.append('/home/lzq/lzy/minkowski_3d_completion')
import resnet
import torch
import numpy as np
from minkowski_utils import replace_features

def read(filename):
    data = np.load(filename, allow_pickle=True)
    res = {}
    res['batch'] = torch.from_numpy((data['rgb'][:,0:1] * 0).astype(np.int32)).cuda()
    res['rgb'] = torch.from_numpy(data['rgb'].astype(np.float32) / 256.0).cuda()
    res['pts'] = torch.from_numpy(data['pts']).cuda()
    return res

def read_ckpt(filename):
    ckpt = torch.load(filename)
    d = {}
    for key in ckpt['model']:
        if key.startswith('encoder.mink_net.'):
            d[key.replace('encoder.mink_net.', '')] = ckpt['model'][key]
    return d


if __name__ == '__main__':

    # params
    offset = torch.from_numpy(np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], np.int)).cuda()
    voxel_size = 0.1
    stage = 'easy'
    a, b = int(sys.argv[1]), int(sys.argv[2])

    # network
    ckpt = read_ckpt('/home/lzq/lzy/NSVF/ReplicaGenFt/all/minkpc_nsvf_rgbasep_field_basev1/checkpoint60.pt')
    mink_net = resnet.MinkResNetEncoder10cm(3, 32, D=3).cuda()
    # ckpt = read_ckpt('/home/lzq/lzy/NSVF/ReplicaGenFt/all/minkpc_nsvf_basev1/checkpoint60.pt')
    # mink_net = resnet.MinkResNetBackbone(3, 32, D=3).cuda()

    mink_net.load_state_dict(ckpt)
    pool = ME.MinkowskiSumPooling(kernel_size=2, stride=1, dimension=3).cuda()

    for sid in range(a, b):
        for fid in tqdm.tqdm(list(range(300))):
            data = read(f'/home/lzq/lzy/NSVF/ReplicaGenFt/all/crop_npz/{sid:02d}_{fid:03d}.npz')

            init_pts = data['pts']
            init_rgb = data['rgb']
            init_batch = data['batch']

            center_idx = torch.floor(init_pts / voxel_size * 16).int() * 2 + 1 # always odd
            vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + offset.repeat(center_idx.shape[0], 1) # always even
            assert((vertex_idx % 2).sum() == 0)
            vertex_idx //= 2
            vertex_idx = torch.cat([torch.repeat_interleave(init_batch, 8, dim=0), vertex_idx], dim=1)
            vertex_info = torch.repeat_interleave(init_rgb, 8, dim=0)

            in_field = ME.TensorField(
                features=vertex_info,
                coordinates=vertex_idx,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=vertex_info.device,
            )
            sparse_input = in_field.sparse()
            
            with torch.no_grad():
                mink_values = mink_net(sparse_input) # stride = 16

                #
                mink_ones = replace_features(mink_values, mink_values.F[:,:1] * 0 + 1)
                center_nums = pool(mink_ones)
                center_idx = center_nums.C[torch.round(center_nums.F[:,0]) == 8, 1:] + 8 # half stride

                # TODO: Here batch size do not support
                vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + 8 * offset.repeat(center_idx.shape[0], 1)
                vertex_idx, feats = torch.unique(vertex_idx, dim=0, return_inverse=True)
                
                feats = feats.reshape(-1, 8)
                points = center_idx * voxel_size / 16
                values = mink_values.features_at_coordinates(torch.cat([vertex_idx[:,:1]*0, vertex_idx], dim=1).float())
                vertex = torch.round(vertex_idx / 16).int()

            np.savez_compressed(
                f'ReplicaGenEncFtTriplets/{stage}/npz_singlecrop/{sid:02d}_{fid:03d}.npz',
                feats=feats.detach().cpu().numpy().astype(np.int32),
                points=points.detach().cpu().numpy().astype(np.float32),
                values=values.detach().cpu().numpy().astype(np.float32),
                vertex=vertex.detach().cpu().numpy().astype(np.int32),
            )
