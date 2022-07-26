import torch

if __name__ == '__main__':

    li = [torch.load(f'all/geo_scn_nsvf_basev1_fixfield_no_ft_loss/checkpoint{i}.pt') for i in range(1, 27)]
    for key in li[0]['model'].keys():
        if key.startswith('encoder.geo_model'):
            ref = li[0]['model'][key]
            print(key)
            for i in range(1, 26):
                tar = li[i]['model'][key]
                print('\t', torch.mean(torch.abs(tar.float() - ref.float())).item())
            input()

    quit()

    # field = torch.load(f'../ReplicaMultiScene/multi_scene_nsvf_basev1_10cm_refine/save/checkpoint_16_78000.pt')
    # encoder = torch.load(f'../SparseConvNet/ckpt_unet_lzq/gatedunet_both_146249.pt') # ckpt_unet/unet_466000.pt
    # current = torch.load(f'all/gated_scn_nsvf_basev1_fixfield_no_ft_loss/save/checkpoint_last.pt')
    # for key in current['model'].keys():
    #     if key.startswith('field'):
    #         current['model'][key] = field['model'][key]
    #         print('F', key)
    #     elif key.startswith('encoder.scn_model'):
    #         current['model'][key] = current['model'][key] * 0 + encoder[key.replace('encoder.scn_model.', '')].to(current['model'][key].device)
    #         print('E', key)      
    #     else:
    #         print('#', key)
        
    # torch.save(current, f'all/gated_scn_nsvf_basev1_fixfield_no_ft_loss/checkpoint_last.pt')

    
    
    
    
    
    field = torch.load(f'../ReplicaMultiScene/multi_scene_nsvf_basev1_10cm_refine/save/checkpoint_16_78000.pt')
    encoder = torch.load(f'../SparseConvNet/ckpt_unet/unet_466000.pt')
    geo = torch.load(f'/home/lzq/lzy/spsg/torch/ckpt/ckpt_14000.pt')['model_state_dict']
    current = torch.load(f'all/geo_scn_nsvf_basev1_fixfield_no_ft_loss/save/checkpoint_last.pt')
    for key in current['model'].keys():
        if key.startswith('field'):
            current['model'][key] = field['model'][key]
            print('F', key)
        elif key.startswith('encoder.scn_model'):
            current['model'][key] = current['model'][key] * 0 + encoder[key.replace('encoder.scn_model.', '')].to(current['model'][key].device)
            print('E', key)
        elif key.startswith('encoder.geo_model'):
            current['model'][key] = current['model'][key] * 0 + geo[key.replace('encoder.geo_model.', '')].to(current['model'][key].device)
            print('G', key)
        else:
            print('#', key)
        
    torch.save(current, f'all/geo_scn_nsvf_basev1_fixfield_no_ft_loss/checkpoint_last.pt')