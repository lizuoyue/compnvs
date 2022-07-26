import glob, tqdm
import numpy as np

if __name__ == '__main__':

    max_spatial = [128, 64, 128]
    e = np.array([
        max_spatial[1] * max_spatial[2],
        max_spatial[2],
        1
    ], np.int32)
    files = tqdm.tqdm(sorted(glob.glob('all/geo/*.npz')))
    for file in files:
        d = np.load(file)

        occ_grid = np.zeros((max_spatial[0] * max_spatial[1] * max_spatial[2]), np.float32)
        uob_grid = np.zeros((max_spatial[0] * max_spatial[1] * max_spatial[2]), np.float32)

        occ_indices_1d = d['is_occupied'].dot(e)
        uob_indices_1d = d['is_unobserved'].dot(e)

        occ_grid[occ_indices_1d] = 1
        uob_grid[uob_indices_1d] = 1

        ch1 = (1 - uob_grid) * occ_grid + uob_grid * 0.5
        ch2 = uob_grid
        geo_x = np.stack([ch1, ch2], axis=-1).reshape((max_spatial[0], max_spatial[1], max_spatial[2], 2))
        geo_y = occ_grid.reshape((max_spatial[0], max_spatial[1], max_spatial[2], 1))
        mask = geo_x[..., 1] > 0.5


        vertex = np.load(file.replace('/geo/', '/npz/')) # vertex_idx vertex_info
        vertex_info = vertex['vertex_info']
        assert(vertex_info.shape[1] == 5)
        vertex_info[:, -2] *= vertex_info[:, -1]
        vertex_info = vertex_info[:, :-1]
        vertex_idx_1d = vertex['vertex_idx'].dot(e)
        info_grid = np.zeros((max_spatial[0] * max_spatial[1] * max_spatial[2], 4), np.float32)
        info_grid[vertex_idx_1d] = vertex_info

        np.savez_compressed(file.replace('/geo/', '/geo_occ/'),
            geo_x=geo_x,
            geo_y=geo_y,
            info_grid=info_grid,
            mask=mask,
        )
