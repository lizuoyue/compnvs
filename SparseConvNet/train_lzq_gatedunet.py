import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import argparse
from net import LzqGatedSparseConvUNet
from net import ReplicaSingleDataset

parser = argparse.ArgumentParser()
parser.add_argument(
        "--loss",
        default='cos',
        type=str,
        choices=['cos', 'l1', 'both']
    )
args = parser.parse_args()

if __name__ == '__main__':

    device = 'cuda:0'
    scn_unet = LzqGatedSparseConvUNet(4, 32, torch.ones(3).long() * 1024).to(device)
    optimizer = torch.optim.Adam(scn_unet.parameters(), lr=0.01)
    data_loader = ReplicaSingleDataset('../ReplicaSingleAll10cmGlobal/all', device)

    l1_sim = nn.L1Loss()
    cos_sim = nn.CosineSimilarity(dim=1)

    it = 0
    sum_loss = 0
    l1_dist = 0
    cos_dist = 0
    while True:

        data = data_loader.get_data()
        
        optimizer.zero_grad()
        pred = scn_unet((data['idx'], data['info'], 1))
        l1_loss = l1_sim(pred, data['ft'])
        cos_loss = torch.mean(1.0 - cos_sim(pred, data['ft']))
        if args.loss == 'cos':
            loss = cos_loss
        elif args.loss == 'l1':
            loss = l1_loss
        elif args.loss == 'both':
            loss = l1_loss + cos_loss
        # print(f'pred: {pred.shape}, data: {data['ft'].shape}')
        loss.backward()
        optimizer.step()

        sum_loss += loss
        l1_dist += l1_loss
        cos_dist += cos_loss
        if (it + 1) % 45 == 0:
            avg_loss = sum_loss / 45
            l1_dist = l1_dist / 45
            cos_dist = cos_dist / 45
            print(f'it: {it}, circle: {it//45}, loss: {avg_loss.item():.4f}, l1: {l1_dist.item():.4f}, cos: {cos_dist.item():.4f}')
            sum_loss = 0
        sys.stdout.flush()

        if (it + 1) % (45 * 250) == 0:
            torch.save(scn_unet.state_dict(), f'ckpt_unet_lzq/gatedunet_{args.loss}_{it}.pt')
        
        it += 1

