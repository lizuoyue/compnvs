import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, glob, tqdm
import numpy as np
from net import SparseConvUNet

class PosEnc(object):

    def __init__(self, d_in, d_out, seed=1993):
        np.random.seed(seed)
        self.a = np.ones(d_out) 
        self.b = np.random.normal(size=(d_out, d_in))
        return
    
    def transform(self, x):
        return np.concatenate([
            self.a * np.sin((2.*np.pi*x).dot(self.b.T)),
            self.a * np.cos((2.*np.pi*x).dot(self.b.T))
        ], axis=-1)

if __name__ == '__main__':

    posenc = PosEnc(3, 32)

    npz = np.load('../ReplicaGenFtTriplets/easy/npz/00_000_067_254.npz')
    vertex_idx = torch.from_numpy(npz['vertex_idx'].astype(np.int32)).cuda()
    vertex_input = torch.from_numpy(npz['vertex_input'].astype(np.float32)).cuda()
    vertex_output = torch.from_numpy(npz['vertex_output'].astype(np.float32)).cuda()
    mask_stay = torch.from_numpy(np.round(npz['vertex_input'][..., -1]).astype(np.bool)).cuda()
    mask_fill = torch.from_numpy(np.round(1 - npz['vertex_input'][..., -1]).astype(np.bool)).cuda()

    # noise_input = torch.normal(vertex_input * 0, vertex_input * 0 + 0.01)[..., :8].cuda()
    noise_input = torch.from_numpy(posenc.transform(npz['vertex_idx']).astype(np.float32)).cuda()

    scn_unet = SparseConvUNet(64, 32, torch.Tensor([144, 64, 160]).long(), reps=3, n_planes=[i * 64 for i in [1, 2, 3, 4, 5]]).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(scn_unet.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.31622776601)

    location = vertex_idx.long().contiguous()

    for it in range(10000):

        optimizer.zero_grad()
        pred = scn_unet((location, noise_input, 1))
        loss_stay = criterion(pred[mask_stay], vertex_output[mask_stay])
        loss_fill = criterion(pred[mask_fill], vertex_output[mask_fill])

        print(f'Iter {it}', loss_stay.item(), loss_fill.item())
        sys.stdout.flush()

        loss_stay.backward()
        optimizer.step()
        scheduler.step()

        if it % 100 == 0:
            np.savez_compressed(f'{it:04d}_posenc_l2.npz', dip=pred.detach().cpu().numpy())
