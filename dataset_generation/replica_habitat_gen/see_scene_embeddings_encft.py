import tqdm, os, glob
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

    obj = PseudoColorConverter('jet')
    val_sid = [13, 14, 19, 20, 21, 42]

    train_pts, val_pts = [], []
    for sid in tqdm.tqdm(list(range(48))):
        files = glob.glob(f'ReplicaGenEncFtTriplets/easy/npz_triple/{sid:02d}_*.npz')
        for file in files[123:]:
            npz = np.load(file, allow_pickle=True)
            fts = npz['values']

            cov_mat = np.cov(fts.T)
            plt.imshow(cov_mat[:8,:8])
            plt.colorbar()
            plt.savefig('temp_cov.png')
            plt.clf()
            plt.imshow(cov_mat[8:,8:])
            plt.colorbar()
            plt.savefig('temp_cov1.png')
            plt.clf()

            for i in range(16):
                plt.plot(fts[:,i*2], fts[:,i*2+1], '*')
                # plt.ylim(-1, 1)
                # plt.xlim(-1, 1)
                plt.savefig(f'ft_{i:02d}.png')
                plt.clf()
            
            ims = [np.array(Image.open(f'ft_{i:02d}.png').convert('RGB')) for i in range(16)]
            h, w, _ = ims[0].shape
            ims = np.stack(ims).reshape((4, 4, h, w, 3)).transpose([0, 2, 1, 3, 4])
            Image.fromarray(ims.reshape((4*h, 4*w, 3))).save('temp_ft.png')

            os.system('rm ft_*.png')
            quit()



