import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from net import LzqGatedSparseConvUNet
from net import ReplicaSingleDataset

if __name__ == '__main__':

    device = 'cuda:0'
    scn_unet = LzqGatedSparseConvUNet(4, 32, torch.ones(3).long() * 1024).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(scn_unet.parameters(), lr=0.001)
    data_loader = ReplicaSingleDataset('../ReplicaSingleAll10cmGlobal/all', device)

    it = 0
    while True:

        data = data_loader.get_data()
        
        optimizer.zero_grad()
        pred = scn_unet((data['idx'], data['info'], 1))
        loss = criterion(pred, data['ft'])
        loss.backward()
        optimizer.step()

        print(it, loss.item())
        sys.stdout.flush()

        if it % 1000 == 0:
            torch.save(scn_unet.state_dict(), f'gated_unet_{it}.pt')
        
        it += 1

