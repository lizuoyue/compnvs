import numpy as np
from PIL import Image
import os, glob

if __name__ == '__main__':

    selects = {
        0: [6, 10, 12],
        1: [6, 12, 18],
        35: [1, 26, 29],
        36: [2, 12, 29],
    }

    empty = np.ones((256, 16, 3), np.uint8) * 255

    for sid in selects:
        files = sorted(glob.glob(f'easy/npz_regdim32emb32out/{sid:02d}_*.npz'))
        for trip_id in selects[sid]:
            _, k, i, j = os.path.basename(files[trip_id][:-4]).split('_')
            img_i = np.array(Image.open(f'../ReplicaGen/scene{sid:02d}/rgb/{i}.png'))[..., :3]
            img_j = np.array(Image.open(f'../ReplicaGen/scene{sid:02d}/rgb/{j}.png'))[..., :3]
            img_k = np.array(Image.open(f'../ReplicaGen/scene{sid:02d}/rgb/{k}.png'))[..., :3]
            imgs = []
            for l in range(15):
                img = Image.open(f'easy/geo_scn_nsvf_base_gif/output{sid:02d}/color/{(trip_id*15+l):04d}.png')
                imgs.append(np.array(img)[..., :3])
            
            line1 = [empty, img_i]
            for l in range(7):
                line1.append(empty)
                line1.append(imgs[l])
            line1 += [empty, img_k, empty]
            line1 = np.hstack(line1)

            line2 = [empty, img_j]
            for l in range(8):
                line2.append(empty)
                line2.append(imgs[14-l])
            line2 += [empty]
            line2 = np.hstack(line2)

            blank = np.ones((16, line2.shape[1], 3), np.uint8) * 255

            final = np.vstack([blank, line1, blank, line2, blank])

            Image.fromarray(final).save(f'{sid:02d}_{trip_id:03d}.png')
            
            
            

                


