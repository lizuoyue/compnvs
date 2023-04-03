import numpy as np
import tqdm
from utils import unproject, project

# for each scene we have the following mapping
#          | voxel  1 | voxel  2 | voxel .. | voxel  N
# view   1 |
# view   2 |    number of pixel
# view ... |
# view 300 |

if __name__ == '__main__':

    voxel_size = 0.1
    n_scene, n_frame = 48, 300
    coef = np.array([64*160, 160, 1], np.int32)
    bboxes = np.loadtxt(f'ReplicaRawData/bboxes.txt')

    for sid in tqdm.tqdm(list(range(n_scene))):
        bbox = bboxes[sid] # boundary ref
        int_mat = np.loadtxt(f'ReplicaGen/scene{sid:02d}/intrinsics.txt')[:3, :3]
        center_pts = np.loadtxt(f'ReplicaGen/scene{sid:02d}/init_voxel/center_points.txt')

        center_glb_3d_idx_raw = (center_pts - bbox[:3]) / voxel_size - 0.5
        center_glb_3d_idx = np.round(center_glb_3d_idx_raw).astype(np.int32)
        assert(np.abs(center_glb_3d_idx - center_glb_3d_idx_raw).mean() < 1e-6)
        center_glb_1d_idx = center_glb_3d_idx.dot(coef)

        n_voxel = center_pts.shape[0]
        glb2loc_mapping = -np.ones((144 * 64 * 160 + 1), np.int32)
        glb2loc_mapping[center_glb_1d_idx] = np.arange(n_voxel) # global 1d index to scene 1d index
        
        valid_map = np.zeros((n_frame, n_voxel), np.int32)
        frame_voxel_idx = -np.ones((n_frame, 256, 256), np.int32)
        for fid in tqdm.tqdm(list(range(n_frame))):
            ext_mat = np.loadtxt(f'ReplicaGen/scene{sid:02d}/pose/{fid:03d}.txt')
            dep = np.load(f'ReplicaRawData/depth/train/{(sid * 300 + fid):05d}.npz')['depth']
            frame_coord = unproject(dep, int_mat, ext_mat)
            frame_3d_idx = np.floor((frame_coord - bbox[:3]) / voxel_size).astype(np.int32)
            valid  = dep > 1e-3
            valid &= frame_3d_idx[..., 0] >= 0
            valid &= frame_3d_idx[..., 0] < 144
            valid &= frame_3d_idx[..., 1] >= 0
            valid &= frame_3d_idx[..., 1] < 64
            valid &= frame_3d_idx[..., 2] >= 0
            valid &= frame_3d_idx[..., 2] < 160
            frame_1d_idx = frame_3d_idx.dot(coef)
            frame_1d_idx[~valid] = -1 # global 1d index
            frame_1d_idx_loc = glb2loc_mapping[frame_1d_idx.flatten()].reshape(frame_1d_idx.shape)

            frame_voxel_idx[fid] = frame_1d_idx_loc
            # import matplotlib; matplotlib.use('agg')
            # import matplotlib.pyplot as plt
            # plt.imshow(frame_1d_idx_loc)
            # plt.savefig(f'{fid:03d}.png')
            # input('press')

            which_voxel, proj_num = np.unique(frame_1d_idx_loc, return_counts=True)
            proj_num = proj_num[which_voxel >= 0]
            which_voxel = which_voxel[which_voxel >= 0]
            valid_map[fid, which_voxel] = proj_num

        relation = np.zeros((n_frame, n_frame), np.float32)
        for i in range(n_frame):
            voxel_mask = valid_map[i] > 0
            for j in range(n_frame):
                relation[i, j] = valid_map[j][voxel_mask].sum() / valid_map[j].sum()

        # import matplotlib; matplotlib.use('agg')
        # import matplotlib.pyplot as plt
        # plt.imshow(relation)
        # plt.savefig(f'{sid:02d}.png')
        # input('press')
    
        np.savez_compressed(
            f'ReplicaGenRelation/scene{sid:02d}.npz',
            mapping=valid_map,
            relation=relation,
            frame_voxel_idx=frame_voxel_idx,
        )
