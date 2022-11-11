import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from minkowski_utils import (
    init_normalization,
    init_activation,
    me_add,
    me_mul,
    me_div,
    me_ones_like,
    me_clamp,
    replace_features,
    me_maximum,
)

class MinkowskiPartialConvolution(ME.MinkowskiConvolution):
    def __init__(self, *args, **kwargs):

        self.multi_channel = kwargs.pop('multi_channel', False) # Default: False
        self.return_mask = kwargs.pop('return_mask', True) # Default: True
        self.norm_to = kwargs.pop('norm_to', 'geometry') # Default: 'geometry'
        assert(self.norm_to in ['kernel_size', 'geometry', 'none'])
        
        self.normalization = kwargs.pop('normalization', 'none') # Default: 'none'
        self.activation = kwargs.pop('activation', 'none') # Default: 'none'

        super(MinkowskiPartialConvolution, self).__init__(*args, **kwargs)
        # self.kernel_generator.kernel_size
        # self.kernel_generator.kernel_stride
        # self.kernel_generator.kernel_dilation
        # self.kernel: spatial, in_channels, out_channels
        self.normalization = init_normalization(self.normalization, self.kernel.shape[-1])
        self.activation = init_activation(self.activation)

        kwargs['bias'] = False
        if self.multi_channel:
            kwargs['out_channels'] = 1
            self.mask_conv = ME.MinkowskiConvolution(*args, **kwargs)
        else:
            kwargs['in_channels'] = 1
            kwargs['out_channels'] = 1
            self.mask_conv = ME.MinkowskiConvolution(*args, **kwargs)
        
        assert(len(list(self.mask_conv.parameters())) == 1) # no bias
        for param in self.mask_conv.parameters():
            param.data.fill_(1.0)
            param.requires_grad = False

        self.slide_window_size = self.kernel.shape[0] * kwargs['in_channels']
        self.update_mask = None
        self.mask_ratio = None

        return

    def forward(self, x_with_mask):
        # both x and mask have to be sparse tensor
        # with the same coordinate_manager and coordinate_map_key
        # print('\n\n\nx:', x)
        # print('\n\n\nmask:', mask)
        x, mask = x_with_mask
        with torch.no_grad():
            self.update_mask = self.mask_conv(mask)
            if self.norm_to == 'kernel_size':
                self.mask_ratio = me_div(self.slide_window_size, me_maximum(self.update_mask, 1e-8))
            elif self.norm_to == 'geometry':
                self.mask_ratio = self.mask_conv(me_ones_like(mask))
                self.mask_ratio /= me_maximum(self.update_mask, 1e-8)
            else:
                assert(self.norm_to == 'none')
                self.mask_ratio = me_ones_like(self.update_mask)

            self.update_mask = me_clamp(self.update_mask, 0, 1)
            self.mask_ratio *= self.update_mask
        
        # print('\n\n\nmask_ratio', self.mask_ratio)
        # print('\n\n\nupdate_mask', self.update_mask)

        masked_x = me_mul(x, mask.features)
        output = super(MinkowskiPartialConvolution, self).forward(masked_x)

        if self.bias is not None:
            output = me_add(output, -self.bias)
            output *= self.mask_ratio
            output = me_add(output, self.bias)
            output *= self.update_mask
        else:
            output *= self.mask_ratio
        
        # print('\n\n\noutput:', output)

        if self.normalization is not None:
            output = self.normalization(output)
        
        if self.activation is not None:
            output = self.activation(output)

        if self.return_mask:
            return output, self.update_mask
        else:
            assert(False)
            return None


class MinkowskiPartialConvolutionTranspose(ME.MinkowskiConvolutionTranspose):
    def __init__(self, *args, **kwargs):

        self.multi_channel = kwargs.pop('multi_channel', False) # Default: False
        self.return_mask = kwargs.pop('return_mask', True) # Default: True
        self.norm_to = kwargs.pop('norm_to', 'geometry') # Default: 'geometry'
        assert(self.norm_to in ['kernel_size', 'geometry', 'none'])
        
        self.normalization = kwargs.pop('normalization', 'none') # Default: 'instance_norm'
        self.activation = kwargs.pop('activation', 'none') # Default: 'leaky_relu'

        super(MinkowskiPartialConvolutionTranspose, self).__init__(*args, **kwargs)
        # self.kernel_generator.kernel_size
        # self.kernel_generator.kernel_stride
        # self.kernel_generator.kernel_dilation
        # self.kernel: spatial, in_channels, out_channels
        self.normalization = init_normalization(self.normalization, self.kernel.shape[-1])
        self.activation = init_activation(self.activation)

        kwargs['bias'] = False
        if self.multi_channel:
            kwargs['out_channels'] = 1
            self.mask_conv = ME.MinkowskiConvolutionTranspose(*args, **kwargs)
        else:
            kwargs['in_channels'] = 1
            kwargs['out_channels'] = 1
            self.mask_conv = ME.MinkowskiConvolutionTranspose(*args, **kwargs)
        
        assert(len(list(self.mask_conv.parameters())) == 1) # no bias
        for param in self.mask_conv.parameters():
            param.data.fill_(1.0)
            param.requires_grad = False

        self.slide_window_size = self.kernel.shape[0] * kwargs['in_channels']
        self.update_mask = None
        self.mask_ratio = None

        return

    def forward(self, x_with_mask):
        # both x and mask have to be sparse tensor
        # with the same coordinate_manager and coordinate_map_key
        # print('\n\n\nx:', x)
        # print('\n\n\nmask:', mask)
        x, mask = x_with_mask
        with torch.no_grad():
            self.update_mask = self.mask_conv(mask)
            if self.norm_to == 'kernel_size':
                self.mask_ratio = me_div(self.slide_window_size, me_maximum(self.update_mask, 1e-8))
            elif self.norm_to == 'geometry':
                self.mask_ratio = self.mask_conv(me_ones_like(mask))
                self.mask_ratio /= me_maximum(self.update_mask, 1e-8)
            else:
                assert(self.norm_to == 'none')
                self.mask_ratio = me_ones_like(self.update_mask)

            self.update_mask = me_clamp(self.update_mask, 0, 1)
            self.mask_ratio *= self.update_mask
        
        # print('\n\n\nmask_ratio', self.mask_ratio)
        # print('\n\n\nupdate_mask', self.update_mask)

        masked_x = me_mul(x, mask.features)
        output = super(MinkowskiPartialConvolutionTranspose, self).forward(masked_x)

        if self.bias is not None:
            output = me_add(output, -self.bias)
            output *= self.mask_ratio
            output = me_add(output, self.bias)
            output *= self.update_mask
        else:
            output *= self.mask_ratio
        
        # print('\n\n\noutput:', output)

        if self.normalization is not None:
            output = self.normalization(output)
        
        if self.activation is not None:
            output = self.activation(output)

        if self.return_mask:
            return output, self.update_mask
        else:
            assert(False)
            return None


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
    x_mask_multi = replace_features(x, feats[:,32:].expand(-1, 32)) # mask_keep

    pconv_down = MinkowskiPartialConvolution(
        in_channels=32,
        out_channels=64,
        kernel_size=2,
        stride=2,
        dilation=1,
        bias=True,
        dimension=3,
        norm_to='kernel_size', # geometry
        multi_channel=False,
    )

    pconv_up = MinkowskiPartialConvolutionTranspose(
        in_channels=64,
        out_channels=32,
        kernel_size=2,
        stride=2,
        dilation=1,
        bias=True,
        dimension=3,
        norm_to='kernel_size', # geometry
        multi_channel=False,
    )

    pconv = MinkowskiPartialConvolution(
        in_channels=64,
        out_channels=3,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
        dimension=3,
        norm_to='kernel_size', # geometry
        multi_channel=True,
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
    



    x_down, x_down_mask = pconv_down(x, x_mask)
    vis(x_down, x_down_mask, 'pconv/pconv_x_down')
    y, y_mask = pconv_up(x_down, x_down_mask)
    vis(y, y_mask, 'pconv/pconv_y')
    y_mask_multi = replace_features(y_mask, y_mask.F.expand(-1, 32)) # mask_keep

    xy = ME.cat(x, y)
    xy_mask = ME.cat(x_mask_multi, y_mask_multi)

    z, z_mask = pconv(xy, xy_mask)
    vis(z, z_mask, 'pconv/pconv_z')


    # quit()
    # y1, y_mask1 = pconv_multi(x, x_mask_multi)
    # # print(torch.abs((y1 - y).F).mean())
    # print(torch.abs((y_mask1 - y_mask).F).mean())
    # for a,b in zip(pconv.mask_ratio.F.detach().cpu().numpy(), pconv_multi.mask_ratio.F.detach().cpu().numpy()):
    #     print(a,b)
    #     input()

    # quit()
    

    
    
    
    