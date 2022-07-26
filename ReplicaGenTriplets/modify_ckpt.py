import torch

if __name__ == '__main__':
    
    field = torch.load(f'ReplicaGen/old_ckpt/checkpoint60.pt')
    encoder = torch.load(f'SparseConvNet/ckpt_unet_gen_ftio/ckpt_6_2000.pt')['model_state_dict']
    # geo = torch.load(f'/home/lzq/lzy/spsg/torch/ckpt/ckpt_14000.pt')['model_state_dict']
    current = torch.load(f'ReplicaGenTriplets/all/geo_scn_nsvf_basev1_ftio/save/checkpoint_last.pt')
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
        
    torch.save(current, f'ReplicaGenTriplets/all/geo_scn_nsvf_basev1_ftio/checkpoint_last.pt')