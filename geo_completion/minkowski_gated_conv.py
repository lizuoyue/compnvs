import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from minkowski_utils import (
    init_normalization,
    init_activation,
)

class MinkowskiGatedConvolution(ME.MinkowskiConvolution):
    def __init__(self, *args, **kwargs):

        self.normalization = kwargs.pop('normalization', 'instance_norm') # Default: 'instance_norm'
        self.activation = kwargs.pop('activation', 'leaky_relu') # Default: 'leaky_relu'

        super(MinkowskiGatedConvolution, self).__init__(*args, **kwargs)
        self.normalization = init_normalization(self.normalization, self.kernel.shape[-1])
        self.activation = init_activation(self.activation)

        self.mask_conv = ME.MinkowskiConvolution(*args, **kwargs)
        self.mask_activation = ME.MinkowskiSigmoid()
        return

    def forward(self, x):
        output = super(MinkowskiGatedConvolution, self).forward(x)

        if self.normalization is not None:
            output = self.normalization(output)
        
        if self.activation is not None:
            output = self.activation(output)

        mask = self.mask_conv(x)
        mask = self.mask_activation(mask)

        output *= mask

        return output



class MinkowskiGatedConvolutionTranspose(ME.MinkowskiConvolutionTranspose):
    def __init__(self, *args, **kwargs):

        self.normalization = kwargs.pop('normalization', 'instance_norm') # Default: 'instance_norm'
        self.activation = kwargs.pop('activation', 'leaky_relu') # Default: 'leaky_relu'

        super(MinkowskiGatedConvolutionTranspose, self).__init__(*args, **kwargs)
        self.normalization = init_normalization(self.normalization, self.kernel.shape[-1])
        self.activation = init_activation(self.activation)

        self.mask_conv = ME.MinkowskiConvolutionTranspose(*args, **kwargs)
        self.mask_activation = ME.MinkowskiSigmoid()
        return

    def forward(self, x):
        output = super(MinkowskiGatedConvolutionTranspose, self).forward(x)

        if self.normalization is not None:
            output = self.normalization(output)
        
        if self.activation is not None:
            output = self.activation(output)

        mask = self.mask_conv(x)
        mask = self.mask_activation(mask)

        output *= mask

        return output


class MinkowskiConvolutionBlock(ME.MinkowskiConvolution):
    def __init__(self, *args, **kwargs):

        self.normalization = kwargs.pop('normalization', 'instance_norm') # Default: 'instance_norm'
        self.activation = kwargs.pop('activation', 'leaky_relu') # Default: 'leaky_relu'

        super(MinkowskiConvolutionBlock, self).__init__(*args, **kwargs)

        self.normalization = init_normalization(self.normalization, self.kernel.shape[-1])
        self.activation = init_activation(self.activation)

        return

    def forward(self, x):
        output = super(MinkowskiConvolutionBlock, self).forward(x)

        if self.normalization is not None:
            output = self.normalization(output)
        
        if self.activation is not None:
            output = self.activation(output)

        return output
