import glob
import numpy as np

bboxes = np.loadtxt('ReplicaGen/bboxes.txt')
ckpt = np.load('ReplicaGen/reg_multi_scene_nsvf_dim32.npz',allow_pickle=True) # index not shifted

val_sid = [13,14,19,20,21,42]
train_sid = [sid for sid in range(48) if sid not in val_sid]

for sid in train_sid:
    print(sid)
    loc = ckpt[f'vertex_point_x10_{sid:02d}'] - (bboxes[sid, :3] * 10).astype(np.int32)
    ft = ckpt[f'vertex_feature_{sid:02d}']
    assert(ft.shape[0] == loc.shape[0])
    d = {}
    for (x, y, z), f in zip(loc, ft):
        d[(x, y, z)] = f

    for file in sorted(glob.glob(f'ReplicaGenFtTriplets/easy/npz_regdim32emb32out/{sid:02d}*.npz')):
        # index shifted
        print(file)
        npz = np.load(file, allow_pickle=True)
        assert(npz['vertex_idx'].shape[0] == npz['vertex_output'].shape[0])
        for (x, y, z), f in zip(npz['vertex_idx'], npz['vertex_output']):
            assert(np.abs(d[(x, y, z)] - f).max() < 1e-4)




