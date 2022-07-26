import numpy as np
from PIL import Image
import glob

if __name__ == '__main__':

    # folder = 'mink_nsvf_base'
    # folder = 'geo_scn_nsvf_base'
    # result_gif/mink_nsvf_base_rgba_init_disc
    folder = 'ReplicaGenFtTriplets/easy/geo_scn_nsvf_base'

    for sid in [36]:#34
        frames = 15
        folder = f'{folder}/output{sid:02d}/color'
        files = sorted(glob.glob(folder + '/*.png'))

        for i in range(len(files) // frames):
            images = [Image.open(files[i * frames + j]) for j in range(frames)]
            images[0].save(folder + f'/_{i:04d}.gif', save_all=True, append_images=images[1:], duration=200, loop=0)
