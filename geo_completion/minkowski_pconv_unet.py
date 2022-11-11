import torch
import torch.nn as nn
import MinkowskiEngine as ME
from minkowski_partial_conv import (
    MinkowskiPartialConvolution,
    MinkowskiPartialConvolutionTranspose,
)
from minkowski_utils import (
    me_clamp,
    replace_features,
)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0
        self.conv1 = MinkowskiPartialConvolution(
            in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiInstanceNorm(planes)
        self.conv2 = MinkowskiPartialConvolution(
            in_channels=planes, out_channels=planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiInstanceNorm(planes)
        self.relu = ME.MinkowskiLeakyReLU(0.1)
        self.downsample = downsample

    def forward(self, x_with_mask):
        x, mask = x_with_mask
        residual, residual_mask = x, mask

        out, out_mask = self.conv1((x, mask))
        out = self.norm1(out)
        out = self.relu(out)

        out, out_mask = self.conv2((out, out_mask))
        out = self.norm2(out)

        if self.downsample is not None:
            residual, residual_mask = self.downsample((x, mask))

        out += residual
        out = self.relu(out)

        out_mask += residual_mask
        out_mask = me_clamp(out_mask, 0, 1)

        return out, out_mask


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0
        self.conv1 = MinkowskiPartialConvolution(
            in_channels=inplanes, out_channels=planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiInstanceNorm(planes)

        self.conv2 = MinkowskiPartialConvolution(
            in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiInstanceNorm(planes)

        self.conv3 = MinkowskiPartialConvolution(
            in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiInstanceNorm(
            planes * self.expansion)

        self.relu = ME.MinkowskiLeakyReLU(0.1)
        self.downsample = downsample

    def forward(self, x_with_mask):
        x, mask = x_with_mask
        residual, residual_mask = x, mask

        out, out_mask = self.conv1((x, mask))
        out = self.norm1(out)
        out = self.relu(out)

        out, out_mask = self.conv2((out, out_mask))
        out = self.norm2(out)
        out = self.relu(out)

        out, out_mask = self.conv3((out, out_mask))
        out = self.norm3(out)

        if self.downsample is not None:
          residual, residual_mask = self.downsample((x, mask))

        out += residual
        out = self.relu(out)

        out_mask += residual_mask
        out_mask = me_clamp(out_mask, 0, 1)

        return out, out_mask













class MinkUNetBase(nn.Module):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1
    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        super().__init__()
        assert(self.BLOCK is not None)
        self.D = D
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()
        return

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = MinkowskiPartialConvolution(
            in_channels=in_channels, out_channels=self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiInstanceNorm(self.inplanes)

        self.conv1p1s2 = MinkowskiPartialConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiInstanceNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = MinkowskiPartialConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiInstanceNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = MinkowskiPartialConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = MinkowskiPartialConvolution(
            in_channels=self.inplanes, out_channels=self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = MinkowskiPartialConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiInstanceNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = MinkowskiPartialConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiInstanceNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = MinkowskiPartialConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiInstanceNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = MinkowskiPartialConvolutionTranspose(
            in_channels=self.inplanes, out_channels=self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiInstanceNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = MinkowskiPartialConvolution(
            in_channels=self.PLANES[7] * self.BLOCK.expansion,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiLeakyReLU(0.1)
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, MinkowskiPartialConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            
            if isinstance(m, MinkowskiPartialConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiInstanceNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MinkowskiPartialConvolution(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                    normalization='instance_norm',
                ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x_with_mask):
        x, mask = x_with_mask
        out, out_p1_mask = self.conv0p1s1((x, mask))
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out, out_mask = self.conv1p1s2((out_p1, out_p1_mask))
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2, out_b1p2_mask = self.block1((out, out_mask))

        out, out_mask = self.conv2p2s2((out_b1p2, out_b1p2_mask))
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4, out_b2p4_mask = self.block2((out, out_mask))

        out, out_mask = self.conv3p4s2((out_b2p4, out_b2p4_mask))
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8, out_b3p8_mask = self.block3((out, out_mask))

        # tensor_stride=16
        out, out_mask = self.conv4p8s2((out_b3p8, out_b3p8_mask))
        out = self.bn4(out)
        out = self.relu(out)
        out, out_mask = self.block4((out, out_mask))

        # tensor_stride=8
        out, out_mask = self.convtr4p16s2((out, out_mask))
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out_mask += out_b3p8_mask
        out_mask = me_clamp(out_mask, 0, 1)
        out, out_mask = self.block5((out, out_mask))

        # tensor_stride=4
        out, out_mask = self.convtr5p8s2((out, out_mask))
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out_mask += out_b2p4_mask
        out_mask = me_clamp(out_mask, 0, 1)
        out, out_mask = self.block6((out, out_mask))

        # tensor_stride=2
        out, out_mask = self.convtr6p4s2((out, out_mask))
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out_mask += out_b1p2_mask
        out_mask = me_clamp(out_mask, 0, 1)
        out, out_mask = self.block7((out, out_mask))

        # tensor_stride=1
        out, out_mask = self.convtr7p2s2((out, out_mask))
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out_mask += out_p1_mask
        out_mask = me_clamp(out_mask, 0, 1)
        out, out_mask = self.block8((out, out_mask))

        return self.final((out, out_mask))


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)













if __name__ == '__main__':

    import numpy as np
    npz1 = np.load('/home/lzq/lzy/NSVF/ReplicaGenFtTriplets/easy/npz/00_000_067_254.npz')
    npz2 = np.load('/home/lzq/lzy/NSVF/ReplicaGenFtTriplets/easy/npz/01_000_188_235.npz')
    # to_fill = npz1['vertex_input'][:,-1] < 1e-3
    # to_keep = npz1['vertex_input'][:,-1] > 1e-3
    # print(np.abs(npz1['vertex_input'][to_fill]).max())
    # print(np.abs(npz1['vertex_output'][to_keep] - npz1['vertex_input'][to_keep, :-1]).max())

    def get_data(npz):
        coords = torch.from_numpy(npz['vertex_idx']).int()#.cuda()
        feats = torch.from_numpy(npz['vertex_input']).float()#.cuda()
        return coords, feats # last channel is mask keep

    coords1, feats1 = get_data(npz1)
    coords2, feats2 = get_data(npz2)
    
    coords, feats = ME.utils.sparse_collate([coords1, coords2], [feats1, feats2])

    x = ME.SparseTensor(coordinates=coords, features=feats[:,:32])
    x_mask = replace_features(x, feats[:,32:]) # mask_keep
    
    net = MinkUNet34C(32, 32, D=3)

    y, y_mask = net((x, x_mask))


    
    def vis(data, data_mask, filename):
        i = 0
        a,b=ME.cat(data, data_mask).decomposed_coordinates_and_features
        for coords, feats in zip(a,b):
            mask = feats[:,-1] > 0.5
            with open(f'{filename}_{i}.txt', 'w') as f:
                for x, y, z in coords[mask].numpy():
                    f.write(f'{x};{y};{z};127;127;127\n')
            i += 1
    

