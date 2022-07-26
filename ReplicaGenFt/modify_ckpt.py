import torch

if __name__ == '__main__':

    # field = torch.load(f'ReplicaGen/multi_scene_nsvf_basev1/checkpoint60.pt')
    # field = torch.load(f'ReplicaGen/reg_multi_scene_nsvf_basev1_dim32/checkpoint60.pt')
    # field = torch.load(f'ReplicaGen/reg_multi_scene_nsvf_basev1/checkpoint60.pt')
    field = torch.load(f'ReplicaGen/multi_scene_nsvf_basev1_rgba_init/checkpoint60.pt')
    # encoder = torch.load(f'SparseConvNet/ckpt_unet_gen_ftio/ckpt_3_6000.pt')['model_state_dict']
    # encoder = torch.load(f'SparseConvNet/ckpt_lzq_unet_gen_ftio/ckpt_32_0.pt')['model_state_dict']
    # encoder = torch.load(f'SparseConvNet/ckpt_lzq_unet_gen_ftio_disc/ckpt_9_6000.pt')['model_state_dict']
    # encoder = torch.load(f'SparseConvNet/ckpt_lzq_unet_gen_ftio_discregdim4emb32in/ckpt_8_5000.pt')['model_state_dict']
    # encoder = torch.load('/home/lzq/lzy/minkowski_3d_completion/ckpt_minkowski_unet34c_regdim32emb32out/ckpt_2_2000.pt')['model_state_dict']['gen']
    # encoder = torch.load('/home/lzq/lzy/minkowski_3d_completion/ckpt_minkowski_deepfillv2_regdim32emb32out/ckpt_16_0.pt')['model_state_dict']['gen']
    encoder = torch.load('/home/lzq/lzy/minkowski_3d_completion/ckpt_minkowski_unet34c_rgbainit/ckpt_1_0.pt')['model_state_dict']['gen']
    # current = torch.load(f'ReplicaGenFtTriplets/easy/geo_scn_nsvf_basev1_noftloss/checkpoint_last.pt')
    # current = torch.load(f'ReplicaGenFtTriplets/easy/geo_scn_nsvf_basev1_regft/checkpoint_last_saved.pt')
    current = torch.load(f'ReplicaGenFtTriplets/easy/mink_nsvf_basergba_init_v1/checkpoint_last_saved.pt')
    
    for key in current['model'].keys():
        if key.startswith('field'):
            current['model'][key] = current['model'][key] * 0 + field['model'][key]
            print('F', key)
        # elif key.startswith('encoder.mink_net'):
        #     current['model'][key] = current['model'][key] * 0 + field['model'][key.replace('reg_scn_model', 'scn_model')]
        #     print('Reg E', key)
        elif key.startswith('encoder.mink_net'):
            current['model'][key] = current['model'][key] * 0 + encoder[key.replace('encoder.mink_net.', '')].to(current['model'][key].device)
            print('E', key)
        else:
            print('#', key)
    
    torch.save(current, f'ReplicaGenFtTriplets/easy/mink_nsvf_basergba_init_v1/checkpoint_last.pt')
    # torch.save(current, f'ReplicaGenFtTriplets/easy/geo_scn_nsvf_basev1/checkpoint_1_1.pt')
