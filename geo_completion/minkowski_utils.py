import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

def replace_features(x, features):
    return ME.SparseTensor(
        features=features,
        coordinate_manager=x.coordinate_manager,
        coordinate_map_key=x.coordinate_map_key,
    )

def me_add(x, const):
    return replace_features(x, x.features + const)

def me_mul(x, const):
    return replace_features(x, x.features * const)

def me_div(const, x):
    return replace_features(x, const / x.features)

def me_zeros_like(x):
    return replace_features(x, torch.zeros_like(x.features))

def me_ones_like(x):
    return replace_features(x, torch.ones_like(x.features))

def me_clamp(x, a, b):
    return replace_features(x, torch.clamp(x.features, a, b))

def me_maximum(x, val):
    return replace_features(x, torch.maximum(x.features, x.features * 0 + val))

def me_feature_slice(x, a, b):
    return replace_features(x, x.features[:, a:b])

def init_normalization(normalization, out_channels):
    if normalization == 'batch_norm':
        normalization = ME.MinkowskiBatchNorm(out_channels)
    elif normalization == 'instance_norm':
        normalization = ME.MinkowskiInstanceNorm(out_channels)
    elif normalization == 'none':
        normalization = None
    else:
        assert(False)
    return normalization

def init_activation(activation):
    if activation == 'relu':
        activation = ME.MinkowskiReLU()
    elif activation == 'leaky_relu':
        activation = ME.MinkowskiLeakyReLU(0.1)
    elif activation == 'tanh':
        activation = ME.MinkowskiTanh()
    elif activation == 'sigmoid':
        activation = ME.MinkowskiSigmoid()
    elif activation == 'none':
        activation = None
    else:
        assert(False)
    return activation

if __name__ == '__main__':

    # generate 1000 random coordinate entries range from 0 to 1000
    coord = torch.empty(1000, 4, dtype=torch.int).random_(10000)
    # generate 1000 feature values with dim 8
    feats = torch.rand(1000, 8)
    # generate a random permutation
    perm = torch.randperm(1000)
    # generate sparse tensor
    x1 = ME.SparseTensor(coordinates=coord, features=feats[:,:4])
    assert(torch.abs(x1.C - x1.coordinates).max().numpy() == 0)
    assert(torch.abs(x1.F - x1.features).max().numpy() == 0)
    assert(torch.abs(x1.C - coord).max().numpy() == 0)

    # print(dir(x1.coordinate_manager))
    # print(x1.coordinate_map_key)

    x2 = ME.SparseTensor(features=feats[:,4:], coordinate_manager=x1.coordinate_manager, coordinate_map_key=x1.coordinate_map_key)
    assert(torch.abs(x2.C - coord).max().numpy() == 0)
    
    x = ME.cat(x1, x2)
    assert(torch.abs(x.F - feats).max().numpy() == 0)

    # y = ME.SparseTensor(coordinates=coord[perm], features=feats[perm], coordinate_manager=x1.coordinate_manager)
    # assert(torch.abs(y.C - coord[perm]).max().numpy() == 0)
    # z = x - y

    # print(z.F.min())
    # print(z.F.max())

    
    # # generate random coordinate to do indexing
    # b_C = torch.empty(2, 2, dtype=torch.long).random_(3)
    # # I want to select features according to b_C
    # b_F = a_S[b_C]  # <<<< this is what I want