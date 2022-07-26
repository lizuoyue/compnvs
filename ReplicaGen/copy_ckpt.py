import torch

if __name__ == '__main__':

    field = torch.load(f'ReplicaGen/reg_multi_scene_nsvf_basev1_dim32_train/checkpoint60.pt')
    current = torch.load(f'ReplicaGen/reg_multi_scene_nsvf_basev1_dim32_val/checkpoint_1_1.pt')
    for key in current['model'].keys():
        if key.startswith('field'):
            current['model'][key] = current['model'][key] * 0 + field['model'][key]
            print('F', key)
        elif key.startswith('encoder.scn_model'):
            current['model'][key] = current['model'][key] * 0 + field['model'][key]
            print('E', key)   
        else:
            print('#', key)
    
    torch.save(current, f'ReplicaGen/reg_multi_scene_nsvf_basev1_dim32_val/checkpoint_last.pt')
    torch.save(current, f'ReplicaGen/reg_multi_scene_nsvf_basev1_dim32_val/checkpoint_1_1.pt')
