import numpy as np
import os, sys, json, tqdm
from PIL import Image
from utils import pose2rotmat, unproject, filter_points, simple_voxelize_points, voxelize_points

if __name__ == '__main__':

    n_scene = 48
    n_frame = 300
    # voxel_size = 0.1

    with open('ReplicaGenRawData/annotations/panoptic_train.json') as f:
        d = json.load(f)

    bboxes = np.loadtxt('ReplicaGenRawData/bboxes.txt')

    num_a, num_b = int(sys.argv[1]), int(sys.argv[2])
    print(num_a, num_b)

    for sid in tqdm.tqdm(list(range(num_a, num_b))):

        li = d['images'][sid * n_frame: (sid + 1) * n_frame]
        pose = np.array([item['pose'] for item in li])
        rotmat = pose2rotmat(pose)
        bbox = bboxes[sid]
        
        # scene_name = f'ReplicaGen/scene{sid:02d}'
        scene_name = f'ReplicaGenFt/all'
        os.system(f'mkdir {scene_name}')
        os.system(f'mkdir {scene_name}/rgb')
        os.system(f'mkdir {scene_name}/depth')
        os.system(f'mkdir {scene_name}/pose')
        # os.system(f'mkdir {scene_name}/init_voxel')
        # os.system(f'mkdir {scene_name}/vertex_info')
        os.system(f'mkdir {scene_name}/npz')

        with open(f'{scene_name}/intrinsics.txt', 'w') as f:
            f.write('128.0 0.0 128.0 0.0\n')
            f.write('0.0 128.0 128.0 0.0\n')
            f.write('0.0 0.0 1.0 0.0\n')
            f.write('0.0 0.0 0.0 1.0\n')

        scene_pts = []
        scene_info = []
        for fid in tqdm.tqdm(list(range(sid * n_frame, (sid + 1) * n_frame))):
            rgb = Image.open(f'ReplicaGenRawData/images/train/{fid:05d}.png')
            # rgb.resize((256, 256), resample=Image.LANCZOS).save(f'{scene_name}/rgb/{(fid % n_frame):03d}.png')
            rgb.resize((256, 256), resample=Image.LANCZOS).save(f'{scene_name}/rgb/{sid:02d}_{(fid % n_frame):03d}.png')
            dep = np.load(f'ReplicaGenRawData/depth/train/{fid:05d}.npz')['depth']
            # os.system(f'cp ReplicaGenRawData/depth/train/{fid:05d}.npz {scene_name}/depth/{(fid % n_frame):03d}.npz')
            os.system(f'cp ReplicaGenRawData/depth/train/{fid:05d}.npz {scene_name}/depth/{sid:02d}_{(fid % n_frame):03d}.npz')

            int_mat = np.array([
                [128.0,   0.0, 128.0],
                [  0.0, 128.0, 128.0],
                [  0.0,   0.0,   1.0],
            ])
            ext_mat = rotmat[fid % n_frame]

            pts = unproject(dep, int_mat, ext_mat)
            pts, mask = filter_points(pts, bbox, init_mask=dep>1e-3, return_mask=True)
            info = np.array(rgb)[..., :3][mask]#.astype(np.float32)
            # pts = simple_voxelize_points(pts, voxel_size, ref=bbox[:3])
            # scene_pts.append(pts.astype(np.float32))
            # scene_info.append(info.astype(np.float32))

            np.savez_compressed(
                f'ReplicaGenFt/all/npz/{sid:02d}_{(fid % n_frame):03d}.npz',
                pts=pts.astype(np.float32),
                rgb=info.astype(np.int32),
            )

            with open(f'{scene_name}/pose/{sid:02d}_{(fid % n_frame):03d}.txt', 'w') as f:
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(ext_mat[0])))
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(ext_mat[1])))
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(ext_mat[2])))
                f.write('0.0 0.0 0.0 1.0\n')
            
            # with open(f'{scene_name}/npz/{sid:02d}_{(fid % n_frame):03d}.txt', 'w') as f:
            #     for (x, y, z), (r, g, b) in zip(pts, info):
            #         f.write('%.4lf;%.4lf;%.4lf;%d;%d;%d\n' % (x, y, z, r, g, b))
        
        # scene_pts = np.concatenate(scene_pts, axis=0)
        # scene_info = np.concatenate(scene_info, axis=0)
        # scene_pts = np.round(scene_pts * 1000).astype(np.int32)
        # scene_pts = np.unique(scene_pts, axis=0)
        # scene_pts = (scene_pts / 1000).astype(np.float32)

        # _, _, vertex_idx, vertex_info = voxelize_points(scene_pts, scene_info, voxel_size, ref=scene_pts.min(axis=0))
        # vertex_info = vertex_info.astype(np.int32)
        # vertex_idx = vertex_idx * 2 - 1

        # with open(f'{scene_name}/init_voxel/center_points_5cm.txt', 'w') as f:
        #     for pt in scene_pts:
        #         f.write('%.3lf %.3lf %.3lf\n' % tuple(list(pt)))

        # with open(f'{scene_name}/vertex_info/vertex_info.txt', 'w') as f:
        #     for (x, y, z), (r, g, b, n) in zip(vertex_idx, vertex_info):
        #         f.write('%d %d %d %d %d %d %d\n' % (x, y, z, r, g, b, n))
        
        # with open(f'{scene_name}/vertex_info/vertex_info_vis.txt', 'w') as f:
        #     for (x, y, z), (r, g, b, n) in zip(vertex_idx, vertex_info):
        #         if n > 0:
        #             rr = np.round(r / n).astype(np.int32)
        #             gg = np.round(g / n).astype(np.int32)
        #             bb = np.round(b / n).astype(np.int32)
        #             f.write('%d %d %d %d %d %d\n' % (x, y, z, rr, gg, bb))
