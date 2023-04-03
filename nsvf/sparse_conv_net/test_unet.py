import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/lzq/lzy/NSVF')
import numpy as np
# from net import SparseConvUNet
from fairnr.modules.encoder import SparseConvUNet
from fairnr.modules.encoder import see_io
from net import ReplicaSingleDataset
from net import pca_to_rgb

if __name__ == '__main__':

    device = 'cuda:0'
    scn_unet = SparseConvUNet(4, 32, torch.ones(3).long() * 1024).to(device)

    h = scn_unet.unet[1].register_forward_hook(see_io)

    scn_unet.load_state_dict(torch.load('ckpt_unet/unet_466000.pt'))
    # scn_unet.eval()
    data_loader = ReplicaSingleDataset('../ReplicaSingleAll10cmGlobal/all', device)

    while True:

        # with torch.no_grad():
        if True:

            data_loader.i = 0
            data = data_loader.get_data()
            print(data['idx'].shape, data['idx'].dtype, data['idx'])
            print('unique', torch.unique(data['idx'], dim=0).shape)
            print(torch.unique(data['idx'], dim=0))
            print(data['info'].shape, data['info'].dtype, data['info'])
            print(1)

            print(data['idx'].is_contiguous())

            print('------------')
            quit()
            pred = scn_unet((data['idx'], data['info'], 1))

            # np.save(f'unet_pred_{i}.npy', pred.cpu().float().numpy())

            # torch.save(pred.cpu(), f'pred_{i}.pt')
            # continue

            print('------------')

            print(pred.shape, pred)
            print(data['ft'].shape, data['ft'])

            print(torch.abs(pred-data['ft']).mean())

            # print('Prediction done!')
            quit()

            # pred_vis = pca_to_rgb(pred.cpu().float().numpy())

            print('PCA done!')
            continue

            with open(f'pc_emb_pred_{i}.txt', 'w') as f:
                pts = data['idx'].cpu().numpy() * 0.1
                for pt, c in zip(pts, pred_vis):
                    x, y, z = pt
                    r, g, b = c.astype(np.int)
                    f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))



