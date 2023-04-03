import torch
import torch.nn as nn
import sparseconvnet as scn
import numpy as np
import glob

class SparseConvUNet(nn.Module):

    def __init__(self, input_dim, output_dim, spatial, reps=1, n_planes=[i * 32 for i in [1, 2, 3, 4, 5]]):
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
            scn.UNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2])).add(
            scn.BatchNormReLU(self.n_planes[0])).add(
            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(self.n_planes[0], output_dim)
        self.final = nn.Tanh()
        self.final_scale = 3
    
    def forward(self, x):
        x = self.unet(x)
        x = self.linear(x)
        x = self.final(x) * self.final_scale
        return x

class SparseConvFCN(nn.Module):

    def __init__(self, input_dim, output_dim, spatial, reps=1, n_planes=[i * 64 for i in [1, 2, 3, 4]]):
        nn.Module.__init__(self)
        self.dim = 3 # 3D
        self.reps = reps # conv block repetition factor
        self.n_planes = n_planes # number of features per level
        self.fcn = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4)).add(
            scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0], filter_size=3, bias=False)).add(
            scn.FullyConvolutionalNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2])).add(
            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(sum(self.n_planes), output_dim)

    def forward(self, x):
        x = self.fcn(x)
        x = self.linear(x)
        return x



class GatedSparseConvUNet(nn.Module):

    def __init__(self, input_dim, output_dim, spatial):
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
        self.reps = 1 # conv block repetition factor
        self.n_planes = [i * 32 for i in [1, 2, 3, 4, 5]] # unet number of features per level
        self.unet = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4)).add(
            scn.Gated(scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0] * 2, filter_size=3, bias=False), 0.1)).add(
            scn.GatedUNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2], leakiness=0.1)).add(
            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(self.n_planes[0], output_dim)
    
    def forward(self, x):
        x = self.unet(x)
        x = self.linear(x)
        return x


class LzqGatedSparseConvUNet(nn.Module):

    def __init__(self, input_dim, output_dim, spatial):
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
        self.reps = 1 # conv block repetition factor
        self.n_planes = [i * 32 for i in [2, 4, 8, 16]] # unet number of features per level
        self.unet = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4)).add(
            scn.Gated(scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0] * 2, filter_size=3, bias=True), 0.1, True)).add(
            # scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0], filter_size=3, bias=False)).add(
            scn.GatedUNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2], leakiness=0.1, batchnorm=True)).add(
            # scn.UNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2])).add(
            # scn.BatchNormReLU(self.n_planes[0])).add(

            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(self.n_planes[0], output_dim)
    
    def forward(self, x):
        x = self.unet(x)
        x = self.linear(x)
        return x

class ReplicaSingleDataset(object):

    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.files = sorted(glob.glob(self.path + '/npz/*.npz'))
        self.num = len(self.files)
        self.i = 0
        return
    
    def get_data(self):
        npz = np.load(self.files[self.i])
        tar = np.load(self.files[self.i].replace('/npz/', '/target_ft/'))

        d = {
            'ft': tar['vertex_feature'],
            'idx': npz['vertex_idx'],
            'info': npz['vertex_info'],
        }

        d['info'][:, -2] *= d['info'][:, -1]
        d['info'] = torch.from_numpy(d['info'][:, :-1]).float().to(self.device)
        d['idx'] = torch.from_numpy(d['idx']).long().to(self.device)
        d['ft'] = torch.from_numpy(d['ft']).float().to(self.device)
        
        self.i += 250
        if self.i >= self.num:
            self.i -= self.num
            self.i += 1
            self.i %= self.num
            # self.i = 0  # LZQ: use only the first view of each scene
        
        return d


if __name__ == '__main__':

    a = ReplicaSingleDataset('../ReplicaSingleAll10cmGlobal/all', 'cuda:0')

