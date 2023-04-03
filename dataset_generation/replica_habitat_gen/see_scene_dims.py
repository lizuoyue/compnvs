import tqdm, os
import numpy as np
from PIL import Image

if __name__ == '__main__':

    # npz = np.load(f'ReplicaGen/multi_scene_nsvf.npz', allow_pickle=True)
    # npz = np.load(f'ReplicaGen/reg_multi_scene_nsvf_dim32.npz', allow_pickle=True)
    npz = np.load(f'ReplicaGen/multi_scene_nsvf_rgba_field_rgba_init.npz', allow_pickle=True)
    # npz_fix = np.load(f'ReplicaGen/multi_scene_nsvf_fix_field.npz', allow_pickle=True)

    train_pts, val_pts = [], []
    train_fix_pts = []
    for sid in tqdm.tqdm(list(range(48))):

        center = npz[f'center_point_{sid:02d}']
        fts = np.tanh(npz[f'vertex_feature_{sid:02d}'])

        # fts_fix = npz_fix[f'vertex_feature_{sid:02d}']
        pts = npz[f'vertex_point_x10_{sid:02d}']

        to_vis = fts#np.concatenate([fts, fts[:,:1]], axis=-1)
        to_vis = (to_vis + 1.0) / 2.0 * 255.0

        # to_vis = np.round(np.clip(to_vis, 0.0, 255.0)).astype(np.uint8).reshape((-1, 4, 3))
        # to_vis = to_vis[:,1:,:].transpose([0,2,1])

        to_vis = np.round(np.clip(to_vis, 0.0, 255.0)).astype(np.uint8).reshape((-1, 4, 4))

        # with open(f'pts/scene{sid:02d}.txt', 'w') as f:
        #     for x,y,z in center:
        #         f.write('%.2lf;%.2lf;%.2lf\n' % (x,y,z))

        for i in range(4):
            with open(f'pts_rgba_init/scene{sid:02d}_{i:02d}.txt', 'w') as f:
                for (x,y,z), (r,g,b) in zip(pts, to_vis[:,i,:3]):
                    f.write('%d;%d;%d;%d;%d;%d\n' % (x,y,z,r,g,b))

        # for i in range(11):
        #     with open(f'pts/scene{sid:02d}_{i:02d}.txt', 'w') as f:
        #         for (x,y,z), (r,g,b) in zip(pts, to_vis[:,i*3:i*3+3]):
        #             f.write('%d;%d;%d;%d;%d;%d\n' % (x,y,z,r,g,b))
        quit()
