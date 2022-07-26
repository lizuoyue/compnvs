import torch
import numpy as np

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

    # ckpt_train = torch.load('/home/lzq/lzy/NSVF/ReplicaGen/multi_scene_nsvf_basev1/checkpoint60.pt')
    # ckpt_train = torch.load('/home/lzq/lzy/NSVF/ReplicaGen/reg_multi_scene_nsvf_basev1/checkpoint60.pt')
    ckpt_train = torch.load(f'ReplicaGen/multi_scene_nsvf_basev1_rgba_init_train/checkpoint60.pt')
    # ckpt_train = torch.load(f'ReplicaGen/multi_scene_nsvf_rgba_field_basev2_zero_init/checkpoint{i}.pt')
    ckpt_val = torch.load('ReplicaGen/multi_scene_nsvf_basev1_rgba_init_val/checkpoint16.pt')

    val_sid = [13,14,19,20,21,42]
    train_sid = [sid for sid in range(48) if sid not in val_sid]

    d = {}
    for sid_li, ckpt in zip([train_sid, val_sid], [ckpt_train, ckpt_val]):
        if ckpt is None:
            continue
        for ckpt_id, sid in enumerate(sid_li):
            print(ckpt_id, sid)
            d[f'center_point_{sid:02d}'] = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.points'].numpy().astype(np.float32)
            d[f'center_keep_{sid:02d}'] = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.keep'].numpy().astype(np.bool)
            d[f'center_to_vertex_{sid:02d}'] = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.feats'].numpy().astype(np.int32)
            d[f'vertex_feature_{sid:02d}'] = ckpt['model'][f'encoder.all_voxels.{ckpt_id}.values.weight'].numpy().astype(np.float32)

            center_point = d[f'center_point_{sid:02d}']
            vertex_point = np.repeat(center_point, 8, axis=0) + np.tile(offset * 0.05, (center_point.shape[0], 1))
            vertex_point_x10 = np.round(vertex_point * 10).astype(np.int32)
            assert(np.abs(vertex_point - vertex_point_x10 / 10).mean() < 5e-6)
            vertex_point = np.zeros((d[f'center_to_vertex_{sid:02d}'].max() + 1, 3), np.int32)
            vertex_point_x10 = np.concatenate([
                d[f'center_to_vertex_{sid:02d}'].flatten()[:, np.newaxis],
                vertex_point_x10,
            ], axis=-1)
            vertex_point_x10 = np.unique(vertex_point_x10, axis=0)
            assert(vertex_point.shape[0] == vertex_point_x10.shape[0])
            vertex_point[vertex_point_x10[:,0]] = vertex_point_x10[:,1:]
            d[f'vertex_point_x10_{sid:02d}'] = vertex_point
    
    np.savez_compressed(f'ReplicaGen/multi_scene_nsvf_rgba_init.npz', **d)
