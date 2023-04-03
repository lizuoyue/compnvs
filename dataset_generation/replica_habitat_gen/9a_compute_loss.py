import numpy as np
import tqdm, random
from PIL import Image
import os, glob
from utils import unproject, filter_points, project, project_depth

if __name__ == '__main__':

    int_mat = np.array([
        [128.0,   0.0, 128.0],
        [  0.0, 128.0, 128.0],
        [  0.0,   0.0,   1.0],
    ])
    bboxes = np.loadtxt('ReplicaGen/bboxes.txt')
    # prefix = 'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base'
    # prefix = 'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base_wodisc'
    prefix = 'ReplicaGenFtTriplets/easy/result_gif/geo_scn_nsvf_base'
    # prefix = 'ReplicaGenFtTriplets/easy/result_gif/geo_scn_nsvf_base_wodisc'
    # prefix = 'ReplicaGenFtTriplets/easy/result_gif/mink_nsvf_base'
    # prefix = 'ReplicaGenFtTriplets/easy/result_gif/mink_nsvf_base_rgba_init_disc'

    res = []
    for sid in [0,1,35,36]:
        bbox = bboxes[sid]
        files = sorted(glob.glob(f'ReplicaGenFtTriplets/easy/npz_regdim32emb32out/{sid:02d}_*.npz'))
        for trip_id in range(30):
            basename = os.path.basename(files[trip_id][:-4])
            _, k, i, j = basename.split('_')

            def read(i):
                rgb = np.array(Image.open(f'ReplicaGen/scene{sid:02d}/rgb/{i}.png'))[..., :3]
                dep = np.load(f'ReplicaGen/scene{sid:02d}/depth/{i}.npz')['depth']
                pose = np.loadtxt(f'ReplicaGen/scene{sid:02d}/pose/{i}.txt')
                pts = unproject(dep, int_mat, pose)
                pts, mask = filter_points(pts, bbox, init_mask=dep>1e-3, return_mask=True)
                info = np.array(rgb)[..., :3][mask]#.astype(np.float32)
                return pts, info, rgb, pose
            
            pts_i, info_i, _, _ = read(i)
            pts_j, info_j, _, _ = read(j)
            _, _, gt_rgb, pose = read(k)

            pts = np.concatenate([pts_i, pts_j], axis=0)
            info = np.concatenate([info_i, info_j], axis=0)


            out = np.zeros((256, 256, 3), np.uint8)
            mask = np.zeros((256, 256), np.bool)

            valid_mask = project(pts, int_mat, pose)
            uv, d = project_depth(pts, int_mat, pose)
            uv = np.floor(uv).astype(np.int32)
            d = d[valid_mask]
            order = np.argsort(d)[::-1]
            d = d[order]
            uv = uv[valid_mask][order]
            vinfo = info[valid_mask][order]

            out[uv[:,1], uv[:,0]] = vinfo
            mask[uv[:,1], uv[:,0]] = True

            pred_rgb = np.array(Image.open(f'{prefix}/output{sid:02d}/color/{(trip_id*15+7):04d}.png'))

            # Image.fromarray(np.hstack([gt_rgb, pred_rgb])).save('tmp.png')
            # input()

            a = np.abs(pred_rgb*1.0 - gt_rgb*1.0).mean()/255.0
            b = np.abs(pred_rgb[mask]*1.0 - gt_rgb[mask]*1.0).mean()/255.0
            c = np.abs(pred_rgb[~mask]*1.0 - gt_rgb[~mask]*1.0).mean()/255.0
            res.append(np.array([a,b,c]))
            print(np.stack(res).mean(axis=0))



