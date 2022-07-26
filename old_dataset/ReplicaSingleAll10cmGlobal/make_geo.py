from make_dataset_proj import project
import numpy as np
import tqdm

if __name__ == '__main__':

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
    voxel_size = 0.1

    max_spatial = [128, 64, 128]
    x, y, z = np.meshgrid(np.arange(max_spatial[0]) + 0.5, np.arange(max_spatial[1]) + 0.5, np.arange(max_spatial[2]) + 0.5, indexing='ij')
    all_pts = np.stack([x, y, z], axis=-1).reshape((-1, 3))
    for i in tqdm.tqdm(list(range(11250))):
        pose = np.loadtxt(f'all/pose/0_{i:05d}.txt')
        d = np.load(f'all/npz/0_{i:05d}.npz')
        pts = all_pts * voxel_size + d['reference']

        is_occupied_raw = (d['voxel_pts'] - d['reference']) / voxel_size - 0.5
        is_occupied = np.round(is_occupied_raw).astype(np.int32)
        assert(np.abs(is_occupied_raw - is_occupied).mean() < 1e-4)

        invalid = project(
            pts,
            np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]),
            pose,
            (128-32, 128+32, 0, 256),
        )
        is_unobserved_raw = all_pts[invalid] - 0.5
        is_unobserved = np.round(is_unobserved_raw).astype(np.int32)
        assert(np.abs(is_unobserved_raw - is_unobserved).mean() < 1e-4)

        # np.savez_compressed(f'all/geo/0_{i:05d}.npz', is_occupied=is_occupied, is_unobserved=is_unobserved, shape=max_spatial)
        # continue

        all_invalid_pts = all_pts[invalid]
        all_invalid_pts_vertex = np.repeat(all_invalid_pts, 8, axis=0) + np.tile(offset, (all_invalid_pts.shape[0], 1)) * 0.5
        all_invalid_pts_vertex = np.unique(all_invalid_pts_vertex, axis=0)

        s = set()
        for px, py, pz in all_invalid_pts_vertex:
            s.add((int(px), int(py), int(pz)))

        for px, py, pz in d['vertex_idx'][d['vertex_info'][:, -1] == 0]:
            if (int(px), int(py), int(pz)) not in s:
                print(i, (int(px), int(py), int(pz)))


