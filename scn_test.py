import h5py
import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from Replica.make_dataset import unproject
from PIL import Image

class SparseConvNetUNet(nn.Module):

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
        self.sparse_conv_net = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4))#.add(
        #     scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0], filter_size=3, bias=False)).add(
        #     scn.UNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2])).add(
        #     scn.BatchNormReLU(self.n_planes[0])).add(
        #     scn.OutputLayer(self.dim))
        # self.linear = nn.Linear(self.n_planes[0], output_dim)
    
    def forward(self, x):
        x = self.sparse_conv_net(x)
        # x = self.linear(x)
        return x

if __name__ == '__main__':

    int_mat = np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]])
    ext_mat = np.eye(4)

    h5_filename = '/home/lzq/lzy/mono_dep_novel_view/dataset/val_replica_pose.h5'
    h5_file = h5py.File(h5_filename, 'r')

    scn_unet = SparseConvNetUNet(4, 32, torch.ones(3).long() * 8)

    locations = torch.ones(10, 3).long() * 1024
    features = torch.ones(10, 4).float()
    z = scn_unet((
        locations,
        features,
    ))
    print(z)
    quit()


    for i, (rgb, dep) in enumerate(zip(h5_file['rgb'][()], h5_file['dep'][()])): # pose , pose2rotmat(h5_file['pose'][()]))):

        pts = unproject(dep[..., 0] / 255.0 * 10.0, int_mat, ext_mat)
        pts = pts[dep[..., 0]>0].reshape((-1, 3))

        print(pts.min(axis=0))
        print(pts.max(axis=0))
        print((pts.max(axis=0)-pts.min(axis=0))/0.05)

        # z = scn_unet((
        #     torch.ones(10, 3).long() * 512,
        #     torch.ones(10, 4).float(),
        # ))
        quit()

