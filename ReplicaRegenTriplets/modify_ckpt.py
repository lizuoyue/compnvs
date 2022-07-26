import torch

if __name__ == '__main__':

    # ckpt_10val_5_1000.pt
    # ckpt_10val_10_1000.pt
    
    field = torch.load(f'ReplicaGen/multi_scene_nsvf_basev1/checkpoint60.pt')
    encoder = torch.load(f'SparseConvNet/ckpt_unet_regen/ckpt_27_0.pt')['model_state_dict']
    # geo = torch.load(f'/home/lzq/lzy/spsg/torch/ckpt/ckpt_14000.pt')['model_state_dict']
    current = torch.load(f'ReplicaRegenTriplets/all/geo_scn_nsvf_basev1/checkpoint_last.pt')
    for key in current['model'].keys():
        if key.startswith('field'):
            current['model'][key] = field['model'][key]
            print('F', key)
        elif key.startswith('encoder.scn_model'):
            current['model'][key] = current['model'][key] * 0 + encoder[key.replace('encoder.scn_model.', '')].to(current['model'][key].device)
            print('E', key)
        # elif key.startswith('encoder.geo_model'):
        #     current['model'][key] = current['model'][key] * 0 + geo[key.replace('encoder.geo_model.', '')].to(current['model'][key].device)
        #     print('G', key)
        else:
            print('#', key)
        
    torch.save(current, f'ReplicaRegenTriplets/all/geo_scn_nsvf_basev1/checkpoint_last.pt')