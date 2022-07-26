import torch

if __name__ == '__main__':

    field = torch.load(f'../ReplicaMultiScene/multi_scene_nsvf_basev1_10cm_refine/checkpoint_45_224000.pt')
    encoder = torch.load(f'../SparseConvNet/ckpt_unet/unet_466000.pt')
    current = torch.load(f'all/scn_nsvf_basev1/save/checkpoint_last.pt')
    for key in current['model'].keys():
        if key.startswith('field'):
            current['model'][key] = current['model'][key] * 0 + field['model'][key]
            print('F', key)
        elif key.startswith('encoder.scn_model'):
            current['model'][key] = current['model'][key] * 0 + encoder[key.replace('encoder.scn_model.', '')].to(current['model'][key].device)
            print('E', key)      
        else:
            print('#', key)
        
    torch.save(current, f'all/scn_nsvf_basev1/checkpoint_last.pt')
