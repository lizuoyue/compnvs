import numpy as np
import sys, tqdm
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import project_depth
from utils import project_depth_dist
from utils import unproject
from utils import simple_voxelize_points
from utils import unproject_free_space
from utils import OccupancyGrid

# 0: free space
# 1: occupied
# 2: in-view unobserved
# 3: out-view unobserved

if __name__ == '__main__':

    a, b = int(sys.argv[1]), int(sys.argv[2])
    spatial_size = [144, 64, 160]
    voxel_size = 0.1
    bboxes = np.loadtxt(f'ReplicaGen/bboxes.txt')

    for sid in tqdm.tqdm(list(range(a, b))):
        bbox = bboxes[sid]
        reference = bbox[:3] # boundary ref
        center_pts = np.loadtxt(f'ReplicaGen/scene{sid:02d}/init_voxel/center_points.txt')
        int_mat = np.loadtxt(f'ReplicaGen/scene{sid:02d}/intrinsics.txt')[:3, :3]
        npz = np.load(f'ReplicaGenRelation/scene{sid:02d}.npz')
        views_voxel = npz['mapping']
        views_pixel2voxel = npz['frame_voxel_idx']

        center_grid_idx_raw = (center_pts - reference) / voxel_size - 0.5
        center_grid_idx = np.round(center_grid_idx_raw).astype(np.int32)
        assert(np.abs(center_grid_idx - center_grid_idx_raw).mean() < 1e-6)
        center_grid_1d_idx = center_grid_idx.dot(np.array([64 * 160, 160, 1], np.int32))
        
        pts_min = center_pts.min(axis=0)
        pts_max = center_pts.max(axis=0)

        for fid in tqdm.tqdm(list(range(300))):

            pixel2voxel = views_pixel2voxel[fid] # (256, 256)
            # contains -1: no mapping

            pose = np.loadtxt(f'ReplicaGen/scene{sid:02d}/pose/{fid:03d}.txt')
            dep = np.load(f'ReplicaGen/scene{sid:02d}/depth/{fid:03d}.npz')['depth']

            _, dep_proj = project_depth(center_pts, int_mat, pose)
            dep_voxelized = dep_proj[pixel2voxel.flatten()].reshape(pixel2voxel.shape)
            dep_voxelized[pixel2voxel == -1] = 0

            diff = np.abs(dep_voxelized - dep)
            # print(' dep avg. err.:', diff.mean(), diff.min(), diff.max())
            # continue
            assert(diff.mean() < 0.1)

            occ_grid = OccupancyGrid(spatial_size, voxel_size, reference + voxel_size / 2, init_value=3)
            grid_pts = occ_grid.get_all_grid_coord()

            should_be_occupied = grid_pts[center_grid_1d_idx[views_voxel[fid] > 0]]

            bound_mask = (pts_min <= grid_pts) & (grid_pts <= pts_max)
            bound_mask = bound_mask[:,0] & bound_mask[:,1] & bound_mask[:,2]
            grid_pts = grid_pts[bound_mask]

            grid_pix_idx, grid_dep_proj = project_depth(grid_pts, int_mat, pose)
            grid_pix_idx = np.floor(grid_pix_idx).astype(np.int32)
            in_view_mask  = grid_pix_idx[:, 0] >= 0
            in_view_mask &= grid_pix_idx[:, 0] <= 255
            in_view_mask &= grid_pix_idx[:, 1] >= 0
            in_view_mask &= grid_pix_idx[:, 1] <= 255
            in_view_mask &= grid_dep_proj > 0.3 # z_min 30cm
            # in_view_mask &= grid_dep_proj < 10.0 # z_max 10m ??
            # in_view_mask &= dist > 0.2

            in_view_grid_pts = grid_pts[in_view_mask]
            grid_pix_idx = grid_pix_idx[in_view_mask]
            grid_dep_proj = grid_dep_proj[in_view_mask]
            grid_dep_ref = dep_voxelized[grid_pix_idx[:, 1], grid_pix_idx[:, 0]]

            free_space_mask = grid_dep_proj < (grid_dep_ref - voxel_size / 2)
            in_view_unobserved_mask = grid_dep_proj > (grid_dep_ref + voxel_size / 2)
            occupied_mask = ~(free_space_mask | in_view_unobserved_mask)

            if True:
                with open(f'pts/4_occ_pts_{sid:02d}_{fid:03d}_in_fs.txt', 'w') as f:
                    for pt in in_view_grid_pts[free_space_mask]:
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                with open(f'pts/4_occ_pts_{sid:02d}_{fid:03d}_in_uob.txt', 'w') as f:
                    for pt in in_view_grid_pts[in_view_unobserved_mask]:
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
                with open(f'pts/4_occ_pts_{sid:02d}_{fid:03d}_in_occ.txt', 'w') as f:
                    for pt in in_view_grid_pts[occupied_mask]:
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                with open(f'pts/4_occ_pts_{sid:02d}_{fid:03d}_occ.txt', 'w') as f:
                    for pt in should_be_occupied:
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple
                input('press')
                continue

            occ_grid.set_occupancy_by_coord(in_view_grid_pts[free_space_mask], 0)
            occ_grid.set_occupancy_by_coord(in_view_grid_pts[in_view_unobserved_mask], 2)
            occ_grid.set_occupancy_by_coord(in_view_grid_pts[occupied_mask], 1)
            occ_grid.set_occupancy_by_coord(should_be_occupied, 1)

            np.savez_compressed(f'ReplicaGenGeometry/occupancy_new/{(sid * 300 + fid):05d}.npz', grid=occ_grid.get_grid())

