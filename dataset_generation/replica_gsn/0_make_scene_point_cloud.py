import numpy as np
import os, json, tqdm
from PIL import Image

def unproject(dep, int_mat, ext_mat):
    # int_mat: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # ext_mat: local to world
    assert(len(dep.shape) == 2)
    h, w = dep.shape
    x, y = np.meshgrid(np.arange(w).astype(np.float32)+0.5, np.arange(h).astype(np.float32)+0.5)
    z = np.ones((h, w), np.float32)
    pts = np.stack([x, y, z], axis=-1)
    pts = pts.dot(np.linalg.inv(int_mat[:3, :3]).T)
    pts = pts * np.stack([dep] * 3, axis=-1) # local
    pts = np.concatenate([pts, np.ones((h, w, 1), np.float32)], axis=-1)
    pts = pts.dot(ext_mat.T)[..., :3]
    return pts

def voxelize_points(pts, info, voxel_size, ref=np.array([0, 0, 0])):
    # pts: shape of N, 3
    # info: shape of N, C
    # return voxel centers
    half_voxel = voxel_size / 2.0
    offset = np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], np.int)

    center_idx = np.unique(np.floor((pts - ref) / voxel_size).astype(np.int), axis=0) * 2 + 1 # always odd

    vertex_idx = np.repeat(center_idx, 8, axis=0) + np.tile(offset, (center_idx.shape[0], 1)) # all voxel vertices index, always even
    vertex_idx, center_to_vertex = np.unique(vertex_idx, axis=0, return_inverse=True)
    center_to_vertex = center_to_vertex.reshape((-1, 8))

    center_pts = center_idx / 2.0 * voxel_size + ref # center coordinates
    vertex_pts = vertex_idx / 2.0 * voxel_size + ref # vertex coordinates

    num_c = info.shape[1]
    vertex_info = np.zeros((vertex_idx.shape[0], num_c + 1))

    vertex_idx_to_1d_idx = {}
    for d_idx, line in enumerate(vertex_idx):
        x, y, z = line
        vertex_idx_to_1d_idx[(x, y, z)] = d_idx

    pt_to_voxel = np.round((pts - ref) / voxel_size).astype(np.int) * 2 # always even
    for (x, y, z), pt_info in zip(pt_to_voxel, info):
        idx = vertex_idx_to_1d_idx[(x, y, z)]
        vertex_info[idx, :num_c] += pt_info
        vertex_info[idx, num_c] += 1

    return center_pts, center_to_vertex, (vertex_idx / 2.0).astype(np.int), vertex_info

if __name__ == '__main__':

    phase, n_seq = 'train', 101
    # phase, n_seq = 'test', 10
    voxel_size = 0.1

    for sid in tqdm.tqdm(list(range(n_seq))):

        with open(f'{phase}/{sid:02d}/cameras.json') as f:
            cameras = json.load(f)
        
        scene_name = f'ReplicaGSN/scene{sid:03d}'
        os.system(f'mkdir {scene_name}')
        os.system(f'mkdir {scene_name}/rgb')
        os.system(f'mkdir {scene_name}/pose')
        os.system(f'mkdir {scene_name}/init_voxel')

        # with open(f'{scene_name}/intrinsics.txt', 'w') as f:
        #     f.write('128.0 0.0 -128.0 0.0\n')
        #     f.write('0.0 -128.0 -128.0 0.0\n')
        #     f.write('0.0 0.0 -1.0 0.0\n')
        #     f.write('0.0 0.0 0.0 1.0\n')
        with open(f'{scene_name}/intrinsics.txt', 'w') as f:
            f.write('128.0 0.0 128.0 0.0\n')
            f.write('0.0 128.0 128.0 0.0\n')
            f.write('0.0 0.0 1.0 0.0\n')
            f.write('0.0 0.0 0.0 1.0\n')

        all_pts = []
        for fid in range(100):
            # rgb_pil = Image.open(f'{phase}/{sid:02d}/{fid:03d}_rgb.png')
            # rgb = np.array(rgb_pil)
            # print(rgb.shape, rgb.dtype, rgb[..., -1].min(), rgb[..., -1].max())
            # dep = np.array(Image.open(f'{phase}/{sid:02d}/{fid:03d}_depth.tiff')) * 1000
            # print(dep.shape, dep.dtype, dep.min(), dep.max())
            # valid = dep > 1e-3
            # print(fid, valid.mean())

            # int_mat = np.array([
            #     [256.0,   0.0, 256.0],
            #     [  0.0, 256.0, 256.0],
            #     [  0.0,   0.0,   1.0],
            # ])
            # int_mat = np.array(cameras[fid]['K'])[:3, :3]
            ext_mat = np.linalg.inv(np.array(cameras[fid]['Rt']))
            ext_mat = ext_mat.dot(np.diag([1.0, -1.0, -1.0, 1.0]))

            # pts = unproject(dep, int_mat, ext_mat)
            # pts = np.concatenate([pts, rgb[..., :3].astype(np.float)], axis=-1)
            # all_pts.append(pts[valid])

            # rgb_pil.resize((256, 256), resample=Image.LANCZOS).save(f'{scene_name}/rgb/{fid:02d}.png')
            with open(f'{scene_name}/pose/{fid:02d}.txt', 'w') as f:
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(ext_mat[0])))
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(ext_mat[1])))
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(ext_mat[2])))
                f.write('0.0 0.0 0.0 1.0\n')
        
        continue

        all_pts = np.concatenate(all_pts, axis=0)
        ref = np.floor(all_pts[:, :3].min(axis=0) / voxel_size) * voxel_size

        center_coord, center_to_vertex, vertex_idx, vertex_info = voxelize_points(all_pts[:, :3], all_pts[:, 3:], voxel_size, ref)
        vertex_coord = vertex_idx * voxel_size + ref

        with open(f'{scene_name}/init_voxel/center_points_{voxel_size}.txt', 'w') as f:
            for pt in center_coord:
                f.write('%.2lf %.2lf %.2lf\n' % tuple(list(pt)))
        
        with open(f'{scene_name}/init_voxel/vertex_points_rgb_{voxel_size}.txt', 'w') as f:
            for (x, y, z), info in zip(vertex_coord, vertex_info):
                if info[-1] > 0.1:
                    r, g, b, n = np.round(info / info[-1]).astype(np.int)
                else:
                    r, g, b, n = 0, 0, 0, 0
                f.write('%.1lf %.1lf %.1lf %d %d %d %d\n' % (x, y, z, r, g, b, n))

