import numpy as np
import os, sys, json, tqdm
from PIL import Image
from utils import pose2rotmat, unproject, filter_points, simple_voxelize_points, voxelize_points

if __name__ == '__main__':

    n_scene = 48
    n_frame = 300

    bboxes = np.loadtxt('ReplicaGen/bboxes.txt')

    num_a, num_b = int(sys.argv[1]), int(sys.argv[2])
    print(num_a, num_b)

    for sid in tqdm.tqdm(list(range(num_a, num_b))):

        bbox = bboxes[sid]
        
        scene_name = f'ReplicaGenFt/all'
        os.system(f'mkdir {scene_name}/crop_npz')

        scene_pts = []
        scene_info = []
        for fid in tqdm.tqdm(list(range(sid * n_frame, (sid + 1) * n_frame))):
            rgb = Image.open(f'{scene_name}/rgb/{sid:02d}_{(fid % n_frame):03d}.png')
            dep = np.load(f'{scene_name}/depth/{sid:02d}_{(fid % n_frame):03d}.npz')['depth']

            int_mat = np.array([
                [128.0,   0.0, 128.0],
                [  0.0, 128.0, 128.0],
                [  0.0,   0.0,   1.0],
            ])
            ext_mat = np.loadtxt(f'{scene_name}/pose/{sid:02d}_{(fid % n_frame):03d}.txt')

            pts = unproject(dep, int_mat, ext_mat)

            init_mask = (dep>1e-3)
            init_mask[:32] = False
            init_mask[-32:] = False
            init_mask[:,:32] = False
            init_mask[:,-32:] = False

            pts, mask = filter_points(pts, bbox, init_mask=init_mask, return_mask=True)
            info = np.array(rgb)[..., :3][mask]

            np.savez_compressed(
                f'ReplicaGenFt/all/crop_npz/{sid:02d}_{(fid % n_frame):03d}.npz',
                pts=pts.astype(np.float32),
                rgb=info.astype(np.int32),
            )
