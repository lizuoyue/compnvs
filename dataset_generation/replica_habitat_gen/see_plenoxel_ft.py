import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    ckpt = np.load('../lego_real_night_hitvbg_dec/ckpt.npz', allow_pickle=True)
    # for key in ckpt.keys():
    #     print(key, ckpt[key].shape, ckpt[key].dtype)
    
    density = ckpt['density_data']
    coeff = ckpt['sh_data']

    rg = np.arange(0, 512).astype(np.int32)
    grid_pts = np.stack(np.meshgrid(rg, rg, rg), axis=-1)[::2, ::2, ::2]

    sub = ckpt['links'][::2, ::2, ::2]

    # sub = sub[sub >= 0]
    grid_density = density[sub.flatten()].reshape(sub.shape)

    # plt.hist(grid_density, bins=1000, range=(-1, 9))
    # plt.ylim(0, 20000)
    # plt.savefig('plenoxel_density_hist.png')

    # quit()

    grid_coeff = coeff[sub.flatten()].reshape(sub.shape + (27,)) * 0.28209479177 * 2

    mask = (sub >= 0) & (5 < grid_density)# & (grid_density < 1)
    pc_pts = grid_pts[mask]
    pc_coeff = np.round((grid_coeff[mask] + 1) / 2 * 255).astype(np.int32)
    pc_density = np.round((grid_density[mask] + 1) / 2 * 255).astype(np.int32)

    for i in range(9):
        with open(f'lego_{i}.txt', 'w') as f:
            for (x,y,z), (r,g,b) in zip(pc_pts, pc_coeff[:,[i,9+i,18+i]]):
                f.write('%d;%d;%d;%d;%d;%d\n' % (x, y, z, r, g, b))

# radius (3,) float32
# center (3,) float32
# links (512, 512, 512) int32
# density_data (20041885, 1) float32
# sh_data (20041885, 27) float16
# background_links (2048, 1024) int32
# background_data (2081586, 64, 4) float32
# basis_type () int64

