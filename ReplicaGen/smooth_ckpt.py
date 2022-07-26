import torch
import numpy as np
import sparseconvnet as scn
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_3d_gaussian_kernel(k, sigma):
    # k is a odd number
    h = k//2
    d = np.arange(-h, h + 1)
    x, y, z = np.meshgrid(d, d, d)
    kernel = np.exp(-(x**2 + y**2 + z**2)/(2*sigma**2)).astype(np.float32) * 100
    print(kernel)
    return kernel

class SCNSmoother(object):

    def __init__(self, num_channels, filter_size, sigma):

        self.dim = 3
        self.spatial = [1024, 1024, 1024]
        self.conv = scn.SubmanifoldConvolution(self.dim, num_channels, num_channels, filter_size=filter_size, bias=False, groups=num_channels)
        self.conv_weight = torch.from_numpy(
            np.concatenate([
                get_3d_gaussian_kernel(filter_size, sigma).reshape((filter_size ** 3, 1, 1, 1))
            ] * num_channels, axis=1)
        )
        self.conv.weight = torch.nn.Parameter(self.conv_weight)
        self.conv.weight.requires_grad = False
        self.scn_smoother = scn.Sequential() \
            .add(scn.InputLayer(self.dim, self.spatial, mode=4)) \
            .add(self.conv) \
            .add(scn.OutputLayer(self.dim)) \
            .cuda()
    
    def __call__(self, locations, features):
        counts = torch.ones_like(features)
        s = self.scn_smoother((locations, features, 1))
        sw = self.scn_smoother((locations, counts, 1))
        return s / sw


if __name__ == '__main__':

    k, s = 3, 0.6
    scn_smoother = SCNSmoother(32, k, s)

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

    ckpt_train = torch.load('multi_scene_nsvf_basev1_train/checkpoint40.pt')
    ckpt_val = torch.load('multi_scene_nsvf_basev1_val/checkpoint6.pt')

    val_sid = [13,14,19,20,21,42]
    train_sid = [sid for sid in range(48) if sid not in val_sid]

    for sid_li, ckpt in zip([train_sid, val_sid], [ckpt_train, ckpt_val]):
        for ckpt_id, sid in enumerate(sid_li):
            print(ckpt_id, sid)
            center_point = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.points'].numpy().astype(np.float32)
            center_to_vertex = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.feats'].numpy().astype(np.int32)
            vertex_feature = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.values.weight'].numpy().astype(np.float32)

            vertex_point = np.repeat(center_point, 8, axis=0) + np.tile(offset * 0.05, (center_point.shape[0], 1))
            vertex_point_x10 = np.round(vertex_point * 10).astype(np.int32)
            assert(np.abs(vertex_point - vertex_point_x10 / 10).max() < 1e-6)
            vertex_point = np.zeros((center_to_vertex.max() + 1, 3), np.int32)
            vertex_point_x10 = np.concatenate([
                center_to_vertex.flatten()[:, np.newaxis],
                vertex_point_x10,
            ], axis=-1)
            vertex_point_x10 = np.unique(vertex_point_x10, axis=0)
            assert(vertex_point.shape[0] == vertex_point_x10.shape[0])
            vertex_point[vertex_point_x10[:,0]] = vertex_point_x10[:,1:]
            
            locations = torch.from_numpy(vertex_point - vertex_point.min(axis=0)).long()
            original_feature = torch.from_numpy(vertex_feature).float().cuda()
            smoothed_feature = scn_smoother(locations, original_feature).cpu()

            # to_see = np.hstack([original_feature[::100].numpy(), smoothed_feature[::100].numpy()])
            # plt.imshow(to_see.T, vmin=-0.5, vmax=0.5)
            # plt.colorbar()
            # plt.savefig(f'{sid:02d}.png')
            ckpt['model'][f'encoder.all_voxels.{ckpt_id}.values.weight'] = smoothed_feature

        torch.save(ckpt, f'multi_scene_nsvf_basev1_train_smooth_{k}_{s}/checkpoint40.pt')
        torch.save(ckpt, f'multi_scene_nsvf_basev1_train_smooth_{k}_{s}/checkpoint_last.pt')
        quit()
    




