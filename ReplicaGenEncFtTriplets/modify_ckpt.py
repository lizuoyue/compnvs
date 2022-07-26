import torch

if __name__ == '__main__':

    # source = torch.load(f'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_basev1_inst/checkpoint_last.pt')
    # target = torch.load(f'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_basedisc_v1_inst/checkpoint_1_1_saved.pt')

    source = torch.load(f'ReplicaGenFt/all/minkpc_nsvf_basev1/checkpoint_last.pt')
    target = torch.load(f'ReplicaGenEncFtTriplets/easy/mink_nsvf_basev1_basic_pred/checkpoint_1_1.pt')
    
    for key in target['model'].keys():
        if key.startswith('field'):
            target['model'][key] = target['model'][key] * 0 + source['model'][key]
            print('F', key)
        # elif key.startswith('encoder.mink_net'):#scn_model
        #     target['model'][key] = target['model'][key] * 0 + source['model'][key]
        #     print('E', key)
        else:
            print('#', key)
    
    # torch.save(target, f'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_basedisc_v1_inst/checkpoint_last.pt')
    torch.save(target, f'ReplicaGenEncFtTriplets/easy/mink_nsvf_basev1_basic_pred/checkpoint_last.pt')
    quit()



    field = torch.load(f'ReplicaGenFt/all/minkpc_nsvf_rgbasep_field_basev1/checkpoint60.pt')
    encoder = torch.load(f'/home/lzq/lzy/minkowski_3d_completion/ckpt_minkowski_unet50inst_encft/ckpt_2_0.pt')['model_state_dict']['gen']
    current = torch.load(f'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_basev1_inst/checkpoint_1_1_saved.pt')
    for key in current['model'].keys():
        if key.startswith('field'):
            current['model'][key] = current['model'][key] * 0 + field['model'][key]
            print('F', key)
        # elif key.startswith('encoder.scn_model'):
        #     current['model'][key] = current['model'][key] * 0 + encoder[key.replace('encoder.scn_model.', '')].to(current['model'][key].device)
        #     print('E', key)
        elif key.startswith('encoder.mink_net'):
            current['model'][key] = current['model'][key] * 0 + encoder[key.replace('encoder.mink_net.', '')].to(current['model'][key].device)
            print('E', key)
        else:
            print('#', key)
    
    torch.save(current, f'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_basev1_inst/checkpoint_last.pt')
