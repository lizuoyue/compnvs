import tqdm, os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

class PseudoColorConverter:
    def __init__(self, cmap_name, start_val=-0.15, stop_val=0.15):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        return

    def convert(self, val):
        return np.round(self.scalarMap.to_rgba(val)[..., :3] * 255).astype(np.uint8)

if __name__ == '__main__':

    npz = np.load(f'ReplicaGen/multi_scene_nsvf.npz', allow_pickle=True)
    npz_fix = np.load(f'ReplicaGen/multi_scene_nsvf_fix_field.npz', allow_pickle=True)
    obj = PseudoColorConverter('jet')
    to_see = np.zeros((48, 32, 3), np.uint8)
    val_sid = [13, 14, 19, 20, 21, 42]

    train_pts, val_pts = [], []
    train_fix_pts = []
    for sid in tqdm.tqdm(list(range(48))):

        fts = npz[f'vertex_feature_{sid:02d}']
        fts_fix = npz_fix[f'vertex_feature_{sid:02d}']
        mu = fts.mean(axis=0)
        to_see[sid] = obj.convert(mu)

        if sid in val_sid:
            val_pts.append(fts[::100])
        else:
            train_pts.append(fts[::100])
            train_fix_pts.append(fts_fix[::100])
    
    train_pts = np.concatenate(train_pts)
    train_fix_pts = np.concatenate(train_fix_pts)
    val_pts = np.concatenate(val_pts)

    cov_train = np.cov(train_pts.T)
    cov_train_fix = np.cov(train_fix_pts.T)
    cov_val = np.cov(val_pts.T)
    plt.imshow(np.hstack([cov_train, cov_train_fix, cov_val]))
    plt.colorbar()
    plt.savefig('cov.png')
    plt.clf()

    for i in range(16):
        plt.plot(train_pts[:,i*2], train_pts[:,i*2+1], '*')
        plt.plot(train_fix_pts[:,i*2], train_fix_pts[:,i*2+1], '*', c='tab:green')
        plt.plot(val_pts[:,i*2], val_pts[:,i*2+1], '*', c='tab:orange')
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.savefig(f'ft_{i:02d}.png')
        plt.clf()
    
    Image.fromarray(to_see).resize((320, 480)).save('scene.png')

    ims = [np.array(Image.open(f'ft_{i:02d}.png').convert('RGB')) for i in range(16)]
    h, w, _ = ims[0].shape
    ims = np.stack(ims).reshape((4, 4, h, w, 3)).transpose([0, 2, 1, 3, 4])
    Image.fromarray(ims.reshape((4*h, 4*w, 3))).save('ft.png')

    os.system('rm ft_*.png')



