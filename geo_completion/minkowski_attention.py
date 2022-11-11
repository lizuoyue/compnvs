import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np

class MinkowskiContextualAttention(nn.Module):

    def __init__(self, in_channels, kq_channels, out_channels, mode='hard'):
        super().__init__()
        self.in_channels = in_channels
        self.kq_channels = kq_channels # key and query
        self.out_channels = out_channels # value

        self.query_layer = nn.Linear(in_channels, kq_channels)
        self.key_layer = nn.Linear(in_channels, kq_channels)
        self.value_layer = nn.Linear(in_channels, out_channels)
    
        self.ratio = np.sqrt(kq_channels)
        self.softmax = nn.Softmax(dim=-1)

        self.mode = mode
        assert(self.mode in ['hard', 'soft'])
        
        return

    def forward_hard(self, x):

        mask_keep = x.F[:,-1] > 0.5
        mask_fill = x.F[:,-1] <= 0.5
        batch_size = x.C[:,0].max().item() + 1

        features = x.F * 1.0 # is with mask
        for i in range(batch_size):
            mask_batch = (x.C[:,0] == i)
            mask_batch_fill = mask_batch & mask_fill
            mask_batch_keep = mask_batch & mask_keep

            fill_query = self.query_layer(x.F[mask_batch_fill, :-1])
            keep_key = self.key_layer(x.F[mask_batch_keep, :-1])
            keep_value = self.value_layer(x.F[mask_batch_keep, :-1])
            prob_dis = self.softmax(torch.mm(fill_query, keep_key.transpose(0, 1)) / self.ratio)

            features[mask_batch_fill, :-1] = torch.mm(prob_dis, keep_value)
        
        return ME.SparseTensor(
            features=features[:, :-1], # is without mask
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )

    def forward(self, x, mask_keep):
        assert(x.F.shape[-1] == self.in_channels)
        x = ME.cat(x, mask_keep) # 0: fill, 1: keep
        if self.mode == 'hard':
            return self.forward_hard(x)
        elif self.mode == 'soft':
            return self.forward_soft(x)
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

    print(coords.shape)
    print(feats.shape)
    quit()

    x = ME.SparseTensor(coordinates=coords, features=feats[:,:-1])
    mask = ME.SparseTensor(
        features=feats[:,-1:],
        coordinate_manager=x.coordinate_manager,
        coordinate_map_key=x.coordinate_map_key,
    ) # mask_keep

    net = MinkowskiContextualAttention(32, 32, 32)#.cuda()

    import time
    tic = time.time()
    y = net(x, mask)
    toc = time.time()

    x -= y

    print(toc - tic)
    for a,b in zip(x.F, mask.F):
        print(a, '    ', b)
        if b.item() == 0:
            input()
