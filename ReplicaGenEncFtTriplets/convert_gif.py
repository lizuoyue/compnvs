import numpy as np
from PIL import Image
import os, glob

if __name__ == '__main__':

    # folder = 'mink_nsvf_base'
    # folder = 'geo_scn_nsvf_base'
    # folder = 'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base'
    folder = 'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base'

    for sid in [13, 14, 19, 20, 21, 42]:
        fid = 0
        tar = f'{folder}/output{sid:02d}/gif'
        os.system(f'mkdir {tar}')
        for file in sorted(glob.glob(f'ReplicaGenEncFtTriplets/easy/npz/{sid:02d}_*.npz')):
            basename = os.path.basename(file).replace('.npz', '')

            ims = [Image.open(f'{folder}/output{sid:02d}/color/{(fid+i):05d}.png') for i in range(15)]
            fid += 15
            ims[0].save(f'{tar}/{basename}.gif', save_all=True, append_images=ims[1:], duration=200, loop=0)

    # for sid in [0,1,35,36]:
    #     frames = 15
    #     full_folder = f'{folder}/output{sid:02d}/color'
    #     files = sorted(glob.glob(full_folder + '/*.png'))
    #     print(len(files))

    #     for i in range(len(files) // frames):
    #         images = [Image.open(files[i * frames + j]) for j in range(frames)]
            
