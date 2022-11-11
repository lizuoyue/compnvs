# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
from urllib.request import urlretrieve
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class BasicBlockInstNorm(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = ME.MinkowskiInstanceNorm(args[1])
        self.norm2 = ME.MinkowskiInstanceNorm(args[1])

class BottleneckInstNorm(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = ME.MinkowskiInstanceNorm(args[1])
        self.norm2 = ME.MinkowskiInstanceNorm(args[1])



class BasicBlockInstanceNorm(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = ME.MinkowskiInstanceNorm(args[1])
        self.norm2 = ME.MinkowskiInstanceNorm(args[1])
        self.norm3 = ME.MinkowskiInstanceNorm(args[1] * self.expansion)

class BottleneckInstanceNorm(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = ME.MinkowskiInstanceNorm(args[1])
        self.norm2 = ME.MinkowskiInstanceNorm(args[1])
        self.norm3 = ME.MinkowskiInstanceNorm(args[1] * self.expansion)


# if not os.path.isfile("1.ply"):
#     print('Downloading an example pointcloud...')
#     urlretrieve("https://bit.ly/3c2iLhg", "1.ply")


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
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

    def forward(self, x: ME.SparseTensor):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


class ResFieldNetBase(ResNetBase):
    def network_initialization(self, in_channels, out_channels, D):
        field_ch = 32
        field_ch2 = 64
        self.field_network = nn.Sequential(
            ME.MinkowskiSinusoidal(in_channels, field_ch),
            ME.MinkowskiBatchNorm(field_ch),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch, field_ch),
            ME.MinkowskiBatchNorm(field_ch),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiToSparseTensor(),
        )
        self.field_network2 = nn.Sequential(
            ME.MinkowskiSinusoidal(field_ch + in_channels, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiToSparseTensor(),
        )

        ResNetBase.network_initialization(self, field_ch2, out_channels, D)

    def forward(self, x: ME.TensorField):
        otensor = self.field_network(x)
        otensor2 = self.field_network2(otensor.cat_slice(x))
        return ResNetBase.forward(self, otensor2)


class ResFieldNet14(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResFieldNet18(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResFieldNet34(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResFieldNet50(ResFieldNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResFieldNet101(ResFieldNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)






















class MinkResNetBackboneBase(ResNetBase):
    BLOCK = None
    LAYERS = (2, 3, 4, 6)
    PLANES = (32, 64, 128, 256)
    INIT_DIM = 32

    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block0 = None

        self.conv1 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn1 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn2 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn3 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn4 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)
        self.relu = ME.MinkowskiReLU(inplace=True)
        return

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        if self.block0 is not None:
            residual = out
            out = self.block0(out)
            out += residual
            out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.block1(out)
        # self.vis(out, 'vis/vis_down1.txt')

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.block2(out)
        # self.vis(out, 'vis/vis_down2.txt')

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.block3(out)
        # self.vis(out, 'vis/vis_down3.txt')

        if self.conv4 is not None:
            out = self.conv4(out)
            out = self.bn4(out)
            out = self.relu(out)
            out = self.block4(out)
        # self.vis(out, 'vis/vis_down4.txt')

        return self.final(out)
    
    def vis(self, x, filename, bias=0):
        with open(filename, 'w') as f:
            for x,y,z in (x.C[:,1:].detach().cpu().numpy() + bias):
                f.write(f'{x};{y};{z}\n')


class MinkResNetBackbone(MinkResNetBackboneBase):
    BLOCK = BottleneckInstNorm




class MinkResNetEncoder(MinkResNetBackboneBase):
    BLOCK = BottleneckInstanceNorm
    LAYERS = (2, 3, 3)
    PLANES = (32, 64, 128)
    INIT_DIM = 32

    def init_block(self, channels, D):
        layers = [
            ME.MinkowskiConvolution(channels, channels, kernel_size=3, dilation=2, dimension=D),
            ME.MinkowskiInstanceNorm(channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(channels, channels, kernel_size=3, dilation=4, dimension=D),
            ME.MinkowskiInstanceNorm(channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(channels, channels, kernel_size=3, dilation=8, dimension=D),
            ME.MinkowskiInstanceNorm(channels),
        ]
        return nn.Sequential(*layers)


    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dilation=1, dimension=D)
        self.bn0 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block0 = self.init_block(self.inplanes, D)

        self.conv1 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn1 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn2 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn3 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4 = None
        # self.conv4 = ME.MinkowskiConvolution(
        #     self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        # self.bn4 = ME.MinkowskiInstanceNorm(self.inplanes)
        # self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
        #                                self.LAYERS[3])
        
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)
        self.relu = ME.MinkowskiReLU(inplace=True)
        return

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                assert(False)
            
            if isinstance(m, ME.MinkowskiInstanceNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiInstanceNorm(planes * block.expansion),
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

class MinkResNetEncoder10cm(MinkResNetEncoder):

    LAYERS = (2, 3, 3, 3)
    PLANES = (32, 64, 128, 128)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        super().network_initialization(in_channels, out_channels, D)
        self.conv4 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D, expand_coordinates=True)
        self.bn4 = ME.MinkowskiInstanceNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        
        return

class MinkResNetEncoderStrong(MinkResNetEncoder10cm):
    LAYERS = (2, 3, 4, 6)
    PLANES = (32, 64, 128, 128)


if __name__ == "__main__":



    net = MinkResNetEncoder10cm(3, 32, D=2)
    print(net)
    quit()






    vertex_idx = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
        [1,2],
        [1,3],
        [2,1],
        [2,2],
        [2,3],
        [3,2],
        [3,3],
    ])
    vertex_info = vertex_idx

    sinput = ME.SparseTensor(
        features=torch.from_numpy(vertex_info[:,:1]*0+1).float(),
        coordinates=ME.utils.batched_coordinates([vertex_idx+1], dtype=torch.float32),
    )

    conv = ME.MinkowskiConvolution(1, 1, kernel_size=3, stride=2, dimension=2, expand_coordinates=True)
    down1 = conv(sinput)
    print(down1)
    down2 = conv(down1)
    print(down2)
    quit()

    

    quit()














    # loss and network
    voxel_size = 0.02
    N_labels = 10

    criterion = nn.CrossEntropyLoss()
    net = ResNet14(in_channels=3, out_channels=N_labels, D=3)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    coords, colors, pcd = load_file("1.ply")
    coords = torch.from_numpy(coords)
    # Get new data
    coordinates = ME.utils.batched_coordinates(
        [coords / voxel_size, coords / 2 / voxel_size, coords / 4 / voxel_size],
        dtype=torch.float32,
    )
    features = torch.rand((len(coordinates), 3), device=device)
    for i in range(10):
        optimizer.zero_grad()

        input = ME.SparseTensor(features, coordinates, device=device)
        dummy_label = torch.randint(0, N_labels, (3,), device=device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, dummy_label)
        print("Iteration: ", i, ", Loss: ", loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), "test.pth")
    net.load_state_dict(torch.load("test.pth"))
