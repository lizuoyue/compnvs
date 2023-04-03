import numpy as np
from net import pca_to_rgb
from net import ReplicaSingleDataset

if __name__ == '__main__':

    data_loader = ReplicaSingleDataset('../ReplicaSingleAll10cmGlobal/all', 'cpu')

    while True:

        i = data_loader.i
        data = data_loader.get_data()

        # pred = np.load(f'unet_pred_{i}.npy')
        pred_vis = np.load(f'pca_{i}.npy')#pca_to_rgb(pred)
        vmin = pred_vis.min(axis=0)
        vmax = pred_vis.max(axis=0)
        pred_vis = (pred_vis - vmin) / (vmax - vmin)

        print('PCA done!')

        with open(f'pc_emb_pred_{i}.txt', 'w') as f:
            pts = data['idx'].cpu().numpy() * 0.1
            for pt, c in zip(pts, pred_vis):
                x, y, z = pt
                r, g, b = (c * 255).astype(np.int)
                f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))



