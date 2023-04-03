import numpy as np
import tqdm, random, json, os, sys
from PIL import Image
from utils import project, unproject, OccupancyGrid, pose2rotmat, simple_voxelize_points, PseudoColorConverter, filter_points
import lib.python.nearest_neighbors as nearest_neighbors
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# Validation
# frl_apartment_0: 19 20 21
# apartment_1: 13 14
# office_2: 42

# 0: free space
# 1: occupied
# 2: in-view unobserved
# 3: out-view unobserved

if __name__ == '__main__':

    a, b = int(sys.argv[1]), int(sys.argv[2])
    cmap = PseudoColorConverter('viridis', 0.0, 0.3)

    n_scene, n_frame = 48, 300
    spatial_size = [144, 64, 160] # center
    voxel_size = 0.1
    bboxes = np.loadtxt(f'ReplicaGen/bboxes.txt')
    int_mat = np.array([
        [128.0,   0.0, 128.0],
        [  0.0, 128.0, 128.0],
        [  0.0,   0.0,   1.0],
    ])

    for sid in tqdm.tqdm(list(range(a, b))):
        bbox = bboxes[sid]
        reference = bbox[:3] # boundary ref
        
        for fid in tqdm.tqdm(list(range(n_frame))):
            grid = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300 + fid):05d}.npz')['grid']
            # occ = np.stack(np.nonzero(grid == 1), axis=-1).astype(np.int32)
            # pts = np.stack(np.nonzero(grid < 100), axis=-1).astype(np.int32)
            pts = np.stack(np.nonzero(grid < 100), axis=-1).astype(np.float32)
            pts = (pts + 0.5) * voxel_size + reference

            dep = np.load(f'ReplicaGen/scene{sid:02d}/depth/{fid:03d}.npz')['depth']
            ext_mat = np.loadtxt(f'ReplicaGen/scene{sid:02d}/pose/{fid:03d}.txt')
            occ = unproject(dep, int_mat, ext_mat)
            occ = filter_points(occ, bbox, init_mask=dep>1e-3, return_mask=False)

            nn_idx = nearest_neighbors.knn_batch(occ[np.newaxis], pts[np.newaxis], 1, omp=True)
            diff = pts - occ[nn_idx.flatten()] # voxel unit / dm
            # assert(diff.dtype == np.int32)

            sq_dist = (diff ** 2).sum(axis=-1).reshape(spatial_size)
            dist = np.sqrt(sq_dist.astype(np.float32)) # m
            # sq_dist[sq_dist >= 255] = 255 # truncated th around 1.6m or 16dm
            # sq_dist = sq_dist.astype(np.uint8)
            # np.savez_compressed(f'ReplicaGenGeometry/distance_field/{(sid * 300 + fid):05d}.npz', distance_field=dist)


            occu = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300 + fid):05d}.npz', allow_pickle=True)['grid']
            print(dist[occu == 1].min(), dist[occu == 1].max())

            if False:
                dist = dist.flatten()
                with open('vis_df.txt', 'w') as f:
                    for pt, (r, g, b) in zip(pts[dist <= 0.3], cmap.convert(dist[dist <= 0.3])):
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
                quit()
