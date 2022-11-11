import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from minkowski_gated_conv import (
    MinkowskiGatedConvolution,
    MinkowskiGatedConvolutionTranspose,
    MinkowskiConvolutionBlock,
)
from minkowski_attention import (
    MinkowskiContextualAttention,
)
from minkowski_partial_conv import (
    MinkowskiPartialConvolution,
)
from minkowski_utils import replace_features

class MinkowskiDeepFillv2(ME.MinkowskiNetwork):

    def __init__(self, in_channels, out_channels, n_feats, D=3):
        super(MinkowskiDeepFillv2, self).__init__(D)

        # in, out, size, stride, dilation, bias

        # first stage
        self.conv1 = MinkowskiGatedConvolution(in_channels, n_feats, 5, 1, 1, True, dimension=D)
        self.conv2_downsample = MinkowskiGatedConvolution(n_feats, 2*n_feats, 3, 2, 1, True, dimension=D)
        self.conv3 = MinkowskiGatedConvolution(2*n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.conv4_downsample = MinkowskiGatedConvolution(2*n_feats, 4*n_feats, 3, 2, 1, True, dimension=D)
        self.conv5 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.conv6 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)

        self.conv7_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 2, True, dimension=D)
        self.conv8_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 4, True, dimension=D)
        self.conv9_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 8, True, dimension=D)
        self.conv10_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 16, True, dimension=D)

        self.conv11 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.conv12 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.conv13_upsample = MinkowskiGatedConvolutionTranspose(4*n_feats, 2*n_feats, 2, 2, 1, True, dimension=D)
        self.conv14 = MinkowskiGatedConvolution(2*n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.conv15_upsample = MinkowskiGatedConvolutionTranspose(2*n_feats, n_feats, 2, 2, 1, True, dimension=D)
        self.conv16 = MinkowskiGatedConvolution(n_feats, n_feats//2, 3, 1, 1, True, dimension=D)
        self.conv17 = MinkowskiGatedConvolution(n_feats//2, out_channels, 3, 1, 1, True, activation='none', dimension=D)

        self.first_stage = nn.Sequential(
            self.conv1,
            self.conv2_downsample,
            self.conv3,
            self.conv4_downsample,
            self.conv5,
            self.conv6,
            self.conv7_atrous,
            self.conv8_atrous,
            self.conv9_atrous,
            self.conv10_atrous,
            self.conv11,
            self.conv12,
            self.conv13_upsample,
            self.conv14,
            self.conv15_upsample,
            self.conv16,
            self.conv17,
        )

        # conv branch
        self.xconv1 = MinkowskiGatedConvolution(out_channels, n_feats, 5, 1, 1, True, dimension=D)
        self.xconv2_downsample = MinkowskiGatedConvolution(n_feats, n_feats, 3, 2, 1, True, dimension=D)
        self.xconv3 = MinkowskiGatedConvolution(n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.xconv4_downsample = MinkowskiGatedConvolution(2*n_feats, 2*n_feats, 3, 2, 1, True, dimension=D)
        self.xconv5 = MinkowskiGatedConvolution(2*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.xconv6 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)

        self.xconv7_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 2, True, dimension=D)
        self.xconv8_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 4, True, dimension=D)
        self.xconv9_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 8, True, dimension=D)
        self.xconv10_atrous = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 16, True, dimension=D)

        self.conv_branch = nn.Sequential(
            self.xconv1,
            self.xconv2_downsample,
            self.xconv3,
            self.xconv4_downsample,
            self.xconv5,
            self.xconv6,
            self.xconv7_atrous,
            self.xconv8_atrous,
            self.xconv9_atrous,
            self.xconv10_atrous,
        )

        # attention branch
        self.pmconv1 = MinkowskiGatedConvolution(out_channels, n_feats, 5, 1, 1, True, dimension=D)
        self.pmconv2_downsample = MinkowskiGatedConvolution(n_feats, n_feats, 3, 2, 1, True, dimension=D)
        self.pmconv3 = MinkowskiGatedConvolution(n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.pmconv4_downsample = MinkowskiGatedConvolution(2*n_feats, 4*n_feats, 3, 2, 1, True, dimension=D)
        self.pmconv5 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.pmconv6 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)

        self.pmconv9 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.pmconv10 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)

        self.attn_branch1 = nn.Sequential(
            self.pmconv1,
            self.pmconv2_downsample,
            self.pmconv3,
            self.pmconv4_downsample,
            self.pmconv5,
            self.pmconv6,
        )
        
        self.attn_layer = MinkowskiContextualAttention(4*n_feats, 4*n_feats, 4*n_feats, 'hard')
        
        self.attn_branch2 = nn.Sequential(
            self.pmconv9,
            self.pmconv10,
        )

        # final conv
        self.allconv11 = MinkowskiGatedConvolution(8*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.allconv12 = MinkowskiGatedConvolution(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.allconv13_upsample = MinkowskiGatedConvolutionTranspose(4*n_feats, 2*n_feats, 2, 2, 1, True, dimension=D)
        self.allconv14 = MinkowskiGatedConvolution(2*n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.allconv15_upsample = MinkowskiGatedConvolutionTranspose(2*n_feats, n_feats, 2, 2, 1, True, dimension=D)
        self.allconv16 = MinkowskiGatedConvolution(n_feats, n_feats//2, 3, 1, 1, True, dimension=D)
        self.allconv17 = MinkowskiGatedConvolution(n_feats//2, out_channels, 3, 1, 1, True, activation='none', dimension=D)

        self.final_conv = nn.Sequential(
            self.allconv11,
            self.allconv12,
            self.allconv13_upsample,
            self.allconv14,
            self.allconv15_upsample,
            self.allconv16,
            self.allconv17,
        )

        self.mask_downsample = nn.Sequential(
            ME.MinkowskiMaxPooling(3, 2, 1, dimension=D),
            ME.MinkowskiMaxPooling(3, 2, 1, dimension=D),
        )

        return

    def forward(self, x):

        mask_fill = replace_features(x, (x.F[:,-1:] > 0.5).float())
        mask_keep = replace_features(x, (x.F[:,-1:] <= 0.5).float())

        # Masks used for attention
        mask_fill_coarse = self.mask_downsample(mask_fill)
        mask_keep_coarse = replace_features(mask_fill_coarse, 1 - mask_fill_coarse.F)

        y_coarse = self.first_stage(x)
        y_coarse = replace_features(y_coarse, torch.tanh(y_coarse.F))
        assert(y_coarse.coordinate_map_key == mask_fill.coordinate_map_key)
        y_coarse_replace = replace_features(y_coarse, y_coarse.F * mask_fill.F + x.F[:,:-1] * mask_keep.F)

        x_attn = self.attn_branch1(y_coarse_replace)
        x_attn = self.attn_layer(x_attn, mask_keep_coarse)
        x_attn = self.attn_branch2(x_attn)

        x_conv = self.conv_branch(y_coarse)

        y = self.final_conv(ME.cat(x_conv, x_attn))
        y = replace_features(y, torch.tanh(y.F))
        assert(y.coordinate_map_key == mask_fill.coordinate_map_key)
        y_replace = replace_features(y, y.F * mask_fill.F + x.F[:,:-1] * mask_keep.F)

        return y_coarse_replace, y

class MinkowskiDeepFillv2Discriminator(ME.MinkowskiNetwork):

    def __init__(self, in_channels, n_feats, D=3):
        super(MinkowskiDeepFillv2Discriminator, self).__init__(D)

        # in, out, size, stride, dilation, bias

        self.dis_conv1 = MinkowskiConvolutionBlock(in_channels, n_feats, 5, 1, 1, True, dimension=D)
        self.dis_conv2_downsample = MinkowskiConvolutionBlock(n_feats, 2*n_feats, 5, 2, 1, True, dimension=D)
        self.dis_conv3 = MinkowskiConvolutionBlock(2*n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv4_downsample = MinkowskiConvolutionBlock(2*n_feats, 4*n_feats, 5, 2, 1, True, dimension=D)
        self.dis_conv5 = MinkowskiConvolutionBlock(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv6_downsample = MinkowskiConvolutionBlock(4*n_feats, 4*n_feats, 5, 2, 1, True, dimension=D)
        self.dis_conv7 = MinkowskiConvolutionBlock(4*n_feats, n_feats, 1, 1, 1, True, dimension=D)
        self.dis_conv8 = MinkowskiConvolutionBlock(n_feats, 1, 1, 1, 1, True, dimension=D, activation='none', normalization='none')

        self.dis = nn.Sequential(
            self.dis_conv1,
            self.dis_conv2_downsample,
            self.dis_conv3,
            self.dis_conv4_downsample,
            self.dis_conv5,
            self.dis_conv6_downsample,
            self.dis_conv7,
            self.dis_conv8,
        )

        return

    def forward(self, x):
        y = self.dis(x)
        return y


class MinkowskiDiscriminator(ME.MinkowskiNetwork):

    def __init__(self, in_channels, n_feats, D=3):
        super(MinkowskiDiscriminator, self).__init__(D)

        # in, out, size, stride, dilation, bias

        self.dis_conv1 = MinkowskiConvolutionBlock(in_channels, n_feats, 5, 1, 1, True, dimension=D)
        self.dis_conv2_downsample = MinkowskiConvolutionBlock(n_feats, 2*n_feats, 3, 2, 1, True, dimension=D)
        self.dis_conv3_downsample = MinkowskiConvolutionBlock(2*n_feats, 4*n_feats, 3, 2, 1, True, dimension=D)
        self.dis_conv4_downsample = MinkowskiConvolutionBlock(4*n_feats, 4*n_feats, 3, 2, 1, True, dimension=D)
        self.dis_conv5 = MinkowskiConvolutionBlock(4*n_feats, n_feats, 1, 1, 1, True, dimension=D)
        self.dis_conv6 = MinkowskiConvolutionBlock(n_feats, 1, 1, 1, 1, True, dimension=D, activation='sigmoid', normalization='none')

        self.dis = nn.Sequential(
            self.dis_conv1,
            self.dis_conv2_downsample,
            self.dis_conv3_downsample,
            self.dis_conv4_downsample,
            self.dis_conv5,
            self.dis_conv6,
        )

        return

    def forward(self, x):
        y = self.dis(x)
        return y

class MinkowskiDilatedDiscriminator(ME.MinkowskiNetwork):

    def __init__(self, in_channels, n_feats, D=3):
        super(MinkowskiDilatedDiscriminator, self).__init__(D)

        # in, out, size, stride, dilation, bias

        self.dis_conv1 = MinkowskiConvolutionBlock(in_channels, n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv2 = MinkowskiConvolutionBlock(n_feats, n_feats, 3, 1, 2, True, dimension=D)
        self.dis_conv3 = MinkowskiConvolutionBlock(n_feats, n_feats, 3, 1, 4, True, dimension=D)
        self.dis_conv4_downsample = MinkowskiConvolutionBlock(n_feats, 2*n_feats, 3, 2, 1, True, dimension=D)
        self.dis_conv5 = MinkowskiConvolutionBlock(2*n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv6 = MinkowskiConvolutionBlock(2*n_feats, 2*n_feats, 3, 1, 2, True, dimension=D)
        self.dis_conv7 = MinkowskiConvolutionBlock(2*n_feats, 2*n_feats, 3, 1, 4, True, dimension=D)
        self.dis_conv8_downsample = MinkowskiConvolutionBlock(2*n_feats, 4*n_feats, 3, 2, 1, True, dimension=D)
        self.dis_conv9 = MinkowskiConvolutionBlock(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv10 = MinkowskiConvolutionBlock(4*n_feats, 4*n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv11 = MinkowskiConvolutionBlock(4*n_feats, n_feats, 1, 1, 1, True, dimension=D)
        self.dis_conv12 = MinkowskiConvolutionBlock(n_feats, 1, 1, 1, 1, True, dimension=D, activation='none', normalization='none')
        self.pool = ME.MinkowskiMaxPooling(3, 2, 1, dimension=D)
        self.dis = nn.Sequential(
            self.dis_conv1,
            self.dis_conv2,
            self.dis_conv3,
            self.dis_conv4_downsample,
            self.dis_conv5,
            self.dis_conv6,
            self.dis_conv7,
            self.dis_conv8_downsample,
            self.dis_conv9,
            self.dis_conv10,
            self.dis_conv11,
            self.dis_conv12,
        )

        return

    def forward(self, x):
        y = self.dis(x)
        mask = self.pool(replace_features(x, x.F[:,-1:]))
        mask = self.pool(mask)
        return ME.cat(y, mask)









class MinkowskiDilatedDiscriminatorOneDown(ME.MinkowskiNetwork):

    def __init__(self, in_channels, n_feats, D=3):
        super(MinkowskiDilatedDiscriminatorOneDown, self).__init__(D)
        # in, out, size, stride, dilation, bias
        self.dis_conv1 = MinkowskiConvolutionBlock(in_channels, n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv2 = MinkowskiConvolutionBlock(n_feats, n_feats, 3, 1, 2, True, dimension=D)
        self.dis_conv3 = MinkowskiConvolutionBlock(n_feats, n_feats, 3, 1, 4, True, dimension=D)
        self.dis_conv4 = MinkowskiConvolutionBlock(n_feats, n_feats, 3, 1, 8, True, dimension=D)
        self.dis_conv5_downsample = MinkowskiConvolutionBlock(n_feats, 2*n_feats, 3, 2, 1, True, dimension=D)
        self.dis_conv6 = MinkowskiConvolutionBlock(2*n_feats, 2*n_feats, 3, 1, 1, True, dimension=D)
        self.dis_conv7 = MinkowskiConvolutionBlock(2*n_feats, 2*n_feats, 3, 1, 2, True, dimension=D)
        self.dis_conv8 = MinkowskiConvolutionBlock(2*n_feats, 2*n_feats, 3, 1, 4, True, dimension=D)
        self.dis_conv9 = MinkowskiConvolutionBlock(2*n_feats, n_feats, 3, 1, 8, True, dimension=D)
        self.dis_conv10 = MinkowskiConvolutionBlock(n_feats, 1, 1, 1, 1, True, dimension=D, activation='none', normalization='none')

        self.pool = ME.MinkowskiMaxPooling(3, 2, 1, dimension=D)
        self.dis = nn.Sequential(
            self.dis_conv1,
            self.dis_conv2,
            self.dis_conv3,
            self.dis_conv4,
            self.dis_conv5_downsample,
            self.dis_conv6,
            self.dis_conv7,
            self.dis_conv8,
            self.dis_conv9,
            self.dis_conv10,
        )
        return

    def forward(self, x):
        y = self.dis(x)
        mask = self.pool(replace_features(x, x.F[:,-1:]))
        return ME.cat(y, mask)










class MinkowskiUNetPartialConvBase(nn.Module):
    BLOCK = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.weight_initialization()

    

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

    

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)

   

    











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

    pconv = MinkowskiPartialConvolution(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
        dimension=3,
        norm_to='geometry', # kernel_size
    )

    def vis(data, data_mask, filename):
        i = 0
        a,b=ME.cat(data, data_mask).decomposed_coordinates_and_features
        for coords, feats in zip(a,b):
            mask = feats[:,-1] > 0.5
            with open(f'{filename}_{i}.txt', 'w') as f:
                for x, y, z in coords[mask].numpy():
                    f.write(f'{x};{y};{z};127;127;127\n')
            i += 1

    y, y_mask = pconv(x, x_mask)
    z, z_mask = pconv(y, y_mask)

    vis(x, x_mask, 'pconv/pconv_x')
    vis(y, y_mask, 'pconv/pconv_y')
    vis(z, z_mask, 'pconv/pconv_z')
    