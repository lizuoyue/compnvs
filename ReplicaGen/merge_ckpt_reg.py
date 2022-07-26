import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

class SparseConvUNet(nn.Module):

    def __init__(self, input_dim, output_dim, spatial, reps=1, n_planes=[i * 32 for i in [1, 2, 3, 4, 5]], bn_momentum=0.99):
        # input_dim: number of input feature channels
        # spatial: spatial size of the voxel grid
        # In case of repetition in coords:
        # mode == 0 if the input is guaranteed to have no duplicates
        # mode == 1 to use the last item at each spatial location
        # mode == 2 to keep the first item at each spatial location
        # mode == 3 to sum feature vectors sharing one spatial location
        # mode == 4 to average feature vectors at each spatial location
        nn.Module.__init__(self)
        self.dim = 3 # 3D
        self.reps = reps # conv block repetition factor
        self.n_planes = n_planes # unet number of features per level
        self.unet = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4)).add(
            scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0], filter_size=3, bias=False)).add(
            scn.UNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2], bn_momentum=bn_momentum)).add(
            scn.BatchNormReLU(self.n_planes[0], momentum=bn_momentum)).add(
            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(self.n_planes[0], output_dim)

        # (1-momentum) * x_saved + momentum * x_t
        return

    def forward(self, x):
        x = self.unet(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':

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

    ckpt_train = torch.load('/home/lzq/lzy/NSVF/ReplicaGen/reg_multi_scene_nsvf_basev1_dim32/checkpoint_last.pt')
    ckpt_val = None#torch.load('/home/lzq/lzy/NSVF/ReplicaGen/multi_scene_nsvf_basev1_val/checkpoint6.pt')

    val_sid = [13,14,19,20,21,42]
    train_sid = [sid for sid in range(48) if sid not in val_sid]

    scn_model = SparseConvUNet(32, 32, torch.Tensor([1024] * 3).long(), 3, [i * 64 for i in [1, 2, 3, 4, 5]], bn_momentum=0.0)
    scn_model_dict = {}
    for key in ckpt_train['model'].keys():
        if key.startswith('encoder.scn_model.'):
            scn_model_dict[key.replace('encoder.scn_model.', '')] = ckpt_train['model'][key]
    scn_model.load_state_dict(scn_model_dict)
    scn_model.cuda()
    scn_model.eval()

    d = {}
    for sid_li, ckpt in zip([train_sid, val_sid], [ckpt_train, ckpt_val]):
        if ckpt is None:
            continue
        for ckpt_id, sid in enumerate(sid_li):
            print(ckpt_id, sid)
            
            d[f'center_point_{sid:02d}'] = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.points'].numpy().astype(np.float32)
            d[f'center_keep_{sid:02d}'] = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.keep'].numpy().astype(np.bool)
            d[f'center_to_vertex_{sid:02d}'] = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.feats'].numpy().astype(np.int32)

            center_point = d[f'center_point_{sid:02d}']
            vertex_point = np.repeat(center_point, 8, axis=0) + np.tile(offset * 0.05, (center_point.shape[0], 1))
            vertex_point_x10 = np.round(vertex_point * 10).astype(np.int32)
            assert(np.abs(vertex_point - vertex_point_x10 / 10).mean() < 5e-6)
            vertex_point = np.zeros((d[f'center_to_vertex_{sid:02d}'].max() + 1, 3), np.int32)
            vertex_point_x10 = np.concatenate([
                d[f'center_to_vertex_{sid:02d}'].flatten()[:, np.newaxis],
                vertex_point_x10,
            ], axis=-1)
            vertex_point_x10 = np.unique(vertex_point_x10, axis=0)
            assert(vertex_point.shape[0] == vertex_point_x10.shape[0])
            vertex_point[vertex_point_x10[:,0]] = vertex_point_x10[:,1:]
            d[f'vertex_point_x10_{sid:02d}'] = vertex_point

            with torch.no_grad():
                locations = torch.from_numpy(vertex_point - vertex_point.min(axis=0)).long().contiguous()
                init_values = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.values.weight'].cuda()

                print(init_values.shape)
                print(init_values)

                init_values = nn.Tanh()(init_values)

                print(locations.shape)
                print(locations)
                values = scn_model((locations, init_values, 1))

                values = nn.Tanh()(values)

                print(values.shape)
                print(values)

                d[f'vertex_feature_{sid:02d}'] = values.cpu().numpy().astype(np.float32)


    np.savez_compressed('reg_multi_scene_nsvf_dim32.npz', **d)
