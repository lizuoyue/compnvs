import torch
import numpy as np

if __name__ == '__main__':

    val_sid = [13,14,19,20,21,42]
    train_sid = [sid for sid in range(48) if sid not in val_sid]

    field = torch.load(f'../reg_multi_scene_nsvf_basev1_dim32/checkpoint60.pt')
    encoders = np.load(f'../reg_multi_scene_nsvf_dim32.npz')
    current = torch.load(f'../multi_scene_nsvf_basev1/checkpoint60.pt')

    for key in current['model'].keys():
        if key.startswith('field'):
            current['model'][key] = current['model'][key] * 0 + field['model'][key]
            print('F', key)
        elif key.startswith('encoder'):
            print('E', key)
        else:
            assert(False)
    
    for ckpt_id, sid in enumerate(train_sid):
        print(ckpt_id, sid)
        current['model'][f'encoder.all_voxels.{ckpt_id}.keep'] *= 0
        current['model'][f'encoder.all_voxels.{ckpt_id}.keep'] += 1
        current['model'][f'encoder.all_voxels.{ckpt_id}.keep'] = current['model'][f'encoder.all_voxels.{ckpt_id}.keep'].bool()
        current['model'][f'encoder.all_voxels.{ckpt_id}.values.weight'] = current['model'][f'encoder.all_voxels.{ckpt_id}.values.weight'] * 0 + \
            torch.from_numpy(encoders[f'vertex_feature_{sid:02d}']).float()
    
    torch.save(current, f'checkpoint_last.pt')
