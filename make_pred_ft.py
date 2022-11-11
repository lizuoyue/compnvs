import MinkowskiEngine as ME
import os, sys, tqdm
import geo_completion.resnet as resnet
import torch, glob
import numpy as np
from geo_completion.minkowski_utils import replace_features

def read(filename):
    data = np.load(filename, allow_pickle=True)
    res = {}
    select = np.arange(data['rgb'].shape[0])
    # np.random.shuffle(select)
    # select = select[:int(data['rgb'].shape[0]*0.3)]
    res['batch'] = torch.from_numpy((data['rgb'][select,0:1] * 0).astype(np.int32)).cuda()
    res['rgb'] = torch.from_numpy(data['rgb'][select].astype(np.float32) / 256.0).cuda()
    res['pts'] = torch.from_numpy(data['pts'][select]).cuda()
    return res

def read_ckpt(filename):
    ckpt = torch.load(filename)
    d = {}
    for key in ckpt['model']:
        if key.startswith('encoder.mink_net.'):
            d[key.replace('encoder.mink_net.', '')] = ckpt['model'][key]
    return d

def set_bn_track_stats_false(net):
    for name, child in net.named_children():
        if 'bn' in name:
            child.track_running_stats = False
        else:
            set_bn_track_stats_false(child)
    return


if __name__ == '__main__':

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

    # network
    ckpt = read_ckpt('ckpt/ckpt_encoder.pt')
    mink_net = resnet.MinkResNetEncoder10cm(3, 32, D=3).cuda()
    mink_net.load_state_dict(ckpt)
    set_bn_track_stats_false(mink_net)

    pool = ME.MinkowskiSumPooling(kernel_size=2, stride=1, dimension=3).cuda()

    files = sorted(glob.glob('ExampleScenesFused/*/npz/00000.npz'))
    
    with torch.no_grad():
        for file in tqdm.tqdm(files[18:]):

            d = read(file)
            center_idx = torch.floor(d['pts'] / voxel_size * 16).long() * 2 + 1 # always odd

            vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + offset.repeat(center_idx.shape[0], 1) # always even
            assert((vertex_idx % 2).sum() == 0)
            vertex_idx //= 2
            vertex_idx = torch.cat([torch.repeat_interleave(d['batch'], 8, dim=0), vertex_idx], dim=1)
            vertex_info = torch.repeat_interleave(d['rgb'], 8, dim=0)

            in_field = ME.TensorField(
                features=vertex_info,
                coordinates=vertex_idx,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=vertex_info.device,
            )
            sparse_input = in_field.sparse()

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
                file.replace('/npz/', '/npz_fts/'),
                feats=feats.detach().cpu().numpy().astype(np.int32),
                points=points.detach().cpu().numpy().astype(np.float32),
                values=values.detach().cpu().numpy().astype(np.float32),
                vertex=vertex.detach().cpu().numpy().astype(np.int32),
            )

