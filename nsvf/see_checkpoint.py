import torch
import numpy as np
from sklearn.decomposition import PCA

def visualize_pc(d, filename):

    pca = PCA(n_components=3)

    pts = d['points'].numpy()
    num_pt = pts.shape[0]
    ft = d['values.weight'].numpy()
    pts_ft = ft[d['feats'].numpy().reshape((-1))].reshape((num_pt, 8, ft.shape[-1]))
    pts_ft = pts_ft.mean(axis=1)

    vis = pca.fit_transform(pts_ft)
    vmin = vis.min(axis=0)
    vmax = vis.max(axis=0)
    vis = (vis - vmin) / (vmax - vmin)
    
    # For debugging, write to point cloud txt file
    with open(f'{filename}', 'w') as f:
        for pt, c in zip(pts, vis):
            x, y, z = pt
            r, g, b = (c * 255).astype(np.int)
            f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
    
    return


def export_multi_scene(ckpt_path, save_path):

    # li = ['points', 'keys', 'feats', 'num_keys', 'keep', 'voxel_size', 'step_size', 'max_hits', 'values.weight']
    li = ['points', 'feats', 'values.weight']

    ckpt = torch.load(ckpt_path)
    for voxel_id in range(45):
        d = {}
        for key in li:
            d[key] = ckpt['model'][f'encoder.all_voxels.{voxel_id}.{key}']
            # print(voxel_id, key, d[key].shape)
        # print(ckpt['model'][f'encoder.all_voxels.{voxel_id}.voxel_size'])
        torch.save(d, save_path + f'/voxel_{voxel_id:02d}.pt')
        # visualize_pc(d, save_path + f'/center_pc_{voxel_id:02d}.txt')
        # d['points'].numpy()
        print(voxel_id, 'Done!')
        # input()
    
    return




if __name__ == '__main__':

    # export_multi_scene('ReplicaMultiScene/multi_scene_nsvf_basev1/checkpoint_47_234000.pt', 'ReplicaMultiScene/voxel_10cm_checkpoint_47_234000')
    # export_multi_scene('ReplicaMultiScene/multi_scene_nsvf_basev1_10cm_refine/checkpoint_3_14000.pt', 'ReplicaMultiScene/voxel_10cm_refine_3_14000')
    # quit()

    field = torch.load(f'ReplicaGen/multi_scene_nsvf_basev1/checkpoint60.pt')
    # current = torch.load(f'ReplicaGenTriplets/all/geo_scn_nsvf_basev1/save/checkpoint_1_1.pt')
    for key in field['model'].keys():
        if key.startswith('field'):
            # current['model'][key] = current['model'][key] * 0 + field['model'][key]
            print('F', key, field['model'][key].shape)   
        else:
            # print('#', key)
            pass
    
    # torch.save(current, f'ReplicaGenTriplets/all/geo_scn_nsvf_basev1/checkpoint_last.pt')
    # for i in range(45):
    #     print(i)
    #     print(ckpt_1['model'][f'encoder.all_voxels.{i}.feats'].shape)
    #     print(ckpt_2['model'][f'encoder.all_voxels.{i}.feats'].shape)
    #     print(ckpt_2['model'][f'encoder.all_voxels.{i}.keep'].dtype)
    #     print()
    # quit()


    # ckpt = torch.load('ReplicaMultiScene/multi_scene_nsvf_basev1_5cm/checkpoint_last.pt')
    # torch.save(ckpt['model'], 'multi_voxel_and_one_field.pt')
    # quit()
    # ckpt = torch.load('Replica/office3_init_voxel/nsvf_basev1/checkpoint5.pt')
    # ckpt = torch.load('ReplicaMultiScene/multi_scene_nsvf_basev1_5cm/checkpoint_last.pt')
    # ckpt = torch.load('ReplicaSingleAll/all/scn_nsvf_basev1/checkpoint_best.pt')
    # ckpt = torch.load('ReplicaMultiScene/multi_scene_nsvf_basev1_10cm/checkpoint_47_234000.pt')
    # ['args', 'model', 'optimizer_history', 'extra_state', 'criterion', 'last_optimizer_state']
    # print(ckpt['args'])
    # print(ckpt)
    # quit()
    # for key in ckpt['model'].keys():
    #     print(key, ckpt['model'][key].shape)
    # quit()

    # print(ckpt['model']['encoder.num_keys'])
    # print(ckpt['model']['encoder.voxel_size'])
    # print(ckpt['model']['encoder.step_size'])
    # print(ckpt['model']['encoder.max_hits'])

    # encoder.points torch.Size([177112, 3])         # from 3099 to 24656 to 177112     # real number coord of voxel center
    # encoder.keys torch.Size([6139, 3])             # stay same                        # may be useless?
    # encoder.feats torch.Size([177112, 8])          # from 3099 to 24656 to 177112     # 8 corner index
    # encoder.num_keys torch.Size([]) ### 227725     # from 6139 to 36606 to 227725
    # encoder.keep torch.Size([177112])              # from 3099 to 24656 to 177112     # whether keep or not, mean 0.9117
    # encoder.voxel_size torch.Size([]) ### 0.0500   # from 0.2000 to 0.1000 to 0.0500
    # encoder.step_size torch.Size([]) ### 0.0063    # from 0.0250 to 0.0125 to 0.0063
    # encoder.max_hits torch.Size([]) ### 135.       # from 60. to 90. to 135.
    # encoder.values.weight torch.Size([227725, 32]) # from 6139 to 36606 to 227725


    # encoder.points torch.Size([177112, 3])         # from 3099 to 24656 to 177112     # real number coord of voxel center
    # encoder.feats torch.Size([177112, 8])          # from 3099 to 24656 to 177112     # 8 corner index
    # encoder.values.weight torch.Size([227725, 32]) # from 6139 to 36606 to 227725

    # quit()

    # print(ckpt['model']['encoder.points'].min(dim=0)[0])
    # print(ckpt['model']['encoder.points'].max(dim=0)[0])
    # print(ckpt['model']['encoder.points'][0])
    # print(ckpt['model']['encoder.points'][1])

    # print(ckpt['model']['encoder.feats'].min())
    # print(ckpt['model']['encoder.feats'].max())

    # print(ckpt['model']['encoder.keys'])

    # print(ckpt['model']['encoder.keep'].min())
    # print(ckpt['model']['encoder.keep'].max())
    # print(ckpt['model']['encoder.keep'].float().mean())




