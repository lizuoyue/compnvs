import tqdm, os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

class PseudoColorConverter:
    def __init__(self, cmap_name, vmin=-1, vmax=1):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        self.scalarMap = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        return

    def convert(self, val):
        return np.round(self.scalarMap.to_rgba(val)[..., :3] * 255).astype(np.uint8)

if __name__ == '__main__':

    npz = np.load(f'ReplicaGen/multi_scene_nsvf.npz', allow_pickle=True)
    # npz = np.load(f'ReplicaGen/multi_scene_nsvf_fix_field.npz', allow_pickle=True)
    obj = PseudoColorConverter('jet')
    to_see = np.zeros((48, 32, 3), np.uint8)
    val_sid = [13, 14, 19, 20, 21, 42]

    for sid in tqdm.tqdm(list(range(48))):

        fts = npz[f'vertex_feature_{sid:02d}']
        c2v = npz[f'center_to_vertex_{sid:02d}']
        cpts = npz[f'center_point_{sid:02d}']

        test = c2v[456]
        test = fts[test]

        aaa = obj.convert(test)
        plt.imshow(aaa, vmin=-1, vmax=1)
        # plt.colorbar()
        plt.savefig('bbb.png')
        print(aaa.shape)
        quit()



        
    train_pts = np.concatenate(train_pts)
    train_fix_pts = np.concatenate(train_fix_pts)

    norm = np.sqrt(np.sum(train_pts ** 2, axis=-1))
    norm_fix = np.sqrt(np.sum(train_fix_pts ** 2, axis=-1))

    ratio = norm_fix / norm
    theta = np.sum(train_pts * train_fix_pts, axis=-1) / norm / norm_fix
    theta = np.rad2deg(np.arccos(np.clip(theta, -1, 1)))

    plt.hist(ratio, bins=100)
    plt.savefig('ratio_hist.png')
    plt.clf()

    plt.hist(theta, bins=100)
    plt.savefig('theta_hist.png')
    plt.clf()

    plt.plot(ratio, theta, '+', ms=3, alpha=0.05)
    plt.savefig('ratio_theta.png')
    plt.clf()

    print(train_pts[12345])
    print(train_fix_pts[12345])
    quit()

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
    


