import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, glob, tqdm, os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from minkowski_network import MinkowskiDilatedDiscriminatorOneDown#MinkowskiDilatedDiscriminator
import MinkowskiEngine as ME
from examplesMink.minkunet import MinkUNet50
from minkowski_utils import replace_features
from minkowski_loss_fn import minkowski_masked_l1_loss_rgba
from minkowski_loss_fn import minkowski_binary_loss_with_logits

def set_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad
    return

def collate_fn(data):
    for idx, d in enumerate(data):
        if idx == 0:
            continue
        d['vertex_idx'][:, 0] += idx
        for key in data[0]:
            data[0][key] = np.concatenate([data[0][key], d[key]], axis=0)
    return {k: data[0][k] for k in data[0]}

class ReplicaGenDataset(Dataset):

    def __init__(self, npz_folder, sid_list=set()):
        self.npz_folder = npz_folder
        self.files = []
        def get_sid(file):
            return int(os.path.basename(file).split('_')[0])
        for file in sorted(glob.glob(f'{npz_folder}/*.npz')):
            if get_sid(file) in sid_list:
                self.files.append(file)
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file, allow_pickle=True)
        data = dict(data)   # ['center_pts', 'center_to_vertex', 'vertex_idx', 'vertex_input', 'vertex_output', 'reference']
        data.pop('center_pts')
        data.pop('center_to_vertex')
        data.pop('reference')
        data['vertex_idx'] = np.concatenate([data['vertex_idx'][:, :1] * 0, data['vertex_idx']], axis=-1).astype(np.int32)
        data['vertex_input'][:, -1] *= -1
        data['vertex_input'][:, -1] += 1 # to mask fill
        data['idx'] = np.array([idx]).astype(np.int32)
        return data




def val_model(
    epoch, it, data,
    gen, gen_loss_fn,
    dis, dis_loss_fn):

    # Data
    vertex_idx = torch.from_numpy(data['vertex_idx']).int().cuda()
    vertex_input = torch.from_numpy(data['vertex_input']).float().cuda()
    vertex_output = torch.from_numpy(data['vertex_output']).float().cuda()

    real_x = ME.SparseTensor(coordinates=vertex_idx, features=vertex_input)
    real_y = replace_features(real_x, vertex_output)
    mask_fill = replace_features(real_x, vertex_input[:, -1:])
    mask_keep = replace_features(real_x, 1 - vertex_input[:, -1:])
    mask_ones = replace_features(real_x, vertex_input[:, -1:] * 0.0 + 1.0)

    # Forward
    with torch.no_grad():
        fake_y = gen(real_x) + replace_features(real_x, vertex_input[:, :-1])

        pred_fake = dis(ME.cat(fake_y, mask_fill))
        loss_dis_fake = dis_loss_fn(pred_fake, False)

        pred_real = dis(ME.cat(real_y, mask_fill))
        loss_dis_real = dis_loss_fn(pred_real, True)

        loss_dis = (loss_dis_fake + loss_dis_real) * 0.5

        pred_fake = dis(ME.cat(fake_y, mask_fill))
        loss_gen_gan = dis_loss_fn(pred_fake, True)

        loss_gen_err_alpha, loss_gen_err_rgb = gen_loss_fn(fake_y, real_y, mask_ones)#mask_fill 

        loss_gen = loss_gen_gan + loss_gen_err_alpha + loss_gen_err_rgb

    print(f'Val Epoch {epoch} Iter {it}: {loss_dis.item():.4f} {loss_gen_gan.item():.4f} {loss_gen_err_alpha.item():.4f} {loss_gen_err_rgb.item():.4f}')  
    return

def train_model(
    gen, gen_loss_fn, gen_optm,
    dis, dis_loss_fn, dis_optm,
    data_loader, val_data_loader=None,
    ckpt_path=None, save_path='./', device='cpu'):

    epoch_saved, it_saved = 0, 0

    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        epoch_saved, it_saved = d['epoch_it']
        gen.load_state_dict(d['model_state_dict']['gen'])
        dis.load_state_dict(d['model_state_dict']['dis'])
        gen_optm.load_state_dict(d['optimizer_state_dict']['gen'])
        dis_optm.load_state_dict(d['optimizer_state_dict']['dis'])


    os.makedirs(save_path, exist_ok=True)
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)

    for epoch in range(epoch_saved, 100):
        if epoch > epoch_saved:
            it_saved = 0
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            if it < it_saved:
                continue

            # Data
            vertex_idx = torch.from_numpy(data['vertex_idx']).int().to(device)
            vertex_input = torch.from_numpy(data['vertex_input']).float().to(device)
            vertex_output = torch.from_numpy(data['vertex_output']).float().to(device)

            real_x = ME.SparseTensor(coordinates=vertex_idx, features=vertex_input)
            real_y = replace_features(real_x, vertex_output)

            mask_fill = replace_features(real_x, vertex_input[:, -1:])
            mask_keep = replace_features(real_x, 1 - vertex_input[:, -1:])
            mask_ones = replace_features(real_x, vertex_input[:, -1:] * 0.0 + 1.0)

            # Forward
            fake_y = gen(real_x) + replace_features(real_x, vertex_input[:, :-1]) # G(x)

            # Update discriminator
            set_requires_grad(dis, True)
            dis_optm.zero_grad() # set D's gradients to zero
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = dis(ME.cat(fake_y, mask_fill).detach())
            loss_dis_fake = dis_loss_fn(pred_fake, False)
            # Real
            pred_real = dis(ME.cat(real_y, mask_fill))

            loss_dis_real = dis_loss_fn(pred_real, True)
            # combine loss and calculate gradients
            loss_dis = (loss_dis_fake + loss_dis_real) * 0.5
            loss_dis.backward()
            dis_optm.step() # update D's weights

            # Update generator
            set_requires_grad(dis, False) # D requires no gradients when optimizing G
            gen_optm.zero_grad() # set G's gradients to zero
            # First, G(A) should fake the discriminator
            pred_fake = dis(ME.cat(fake_y, mask_fill))
            loss_gen_gan = dis_loss_fn(pred_fake, True)
            # Second, G(A) = B
            loss_gen_err_alpha, loss_gen_err_rgb = gen_loss_fn(fake_y, real_y, mask_ones)#mask_fill
            # combine loss and calculate gradients
            loss_gen = loss_gen_gan + loss_gen_err_alpha + loss_gen_err_rgb
            loss_gen.backward()
            gen_optm.step() # udpate G's weights

            print(f'Train Epoch {epoch} Iter {it}: {loss_dis.item():.4f} {loss_gen_gan.item():.4f} {loss_gen_err_alpha.item():.4f} {loss_gen_err_rgb.item():.4f}')            

            if it % 1000 == 0:
                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': {
                        'gen': gen.state_dict(),
                        'dis': dis.state_dict(),
                    },
                    'optimizer_state_dict': {
                        'gen': gen_optm.state_dict(),
                        'dis': dis_optm.state_dict(),
                    },
                }
                torch.save(d, f'{save_path}/ckpt_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                if val_data_loader is not None:
                    val_data = next(val_data_iter, None)
                    if val_data is None:
                        val_data_iter = iter(val_data_loader)
                        val_data = next(val_data_iter, None)
                    val_model(epoch, it, val_data, gen, gen_loss_fn, dis, dis_loss_fn)
            
            sys.stdout.flush()

    return


if __name__ == '__main__':

    sid_by_phase = {'Training': set(), 'Validation': set()}
    with open('/work/lzq/matterport3d/arkit_triplets/selected_scan_ids.txt', 'r') as f:
        sel_scan_ids = [line.strip().split() for line in f.readlines()]
        for scan_id, phase in sel_scan_ids:
            filename = f'/work/lzq/matterport3d/arkit_triplets/triplets/{scan_id}.npz'
            if os.path.exists(filename):
                sid_by_phase[phase].add(int(scan_id))

    train_data = ReplicaGenDataset('/work/lzq/matterport3d/ARKitScenes/ARKitScenes/ARKitScenesTripletsNSVF/all/npz', sid_by_phase['Training'])
    val_data = ReplicaGenDataset('/work/lzq/matterport3d/ARKitScenes/ARKitScenes/ARKitScenesTripletsNSVF/all/npz', sid_by_phase['Validation'])

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_data_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = 'cuda:0'
    emb_size, mid_size = 32, 128
    gen = MinkUNet50(emb_size + 1, emb_size).to(device)
    dis = MinkowskiDilatedDiscriminatorOneDown(emb_size + 1, mid_size).to(device)

    gen_criterion = minkowski_masked_l1_loss_rgba
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)

    dis_criterion = minkowski_binary_loss_with_logits
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-4)

    save_path = 'ckpt_minkowski_unet50_encft_arkit'
    train_model(
        gen, gen_criterion, gen_optimizer,
        dis, dis_criterion, dis_optimizer,
        data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        ckpt_path=None,
        save_path=save_path,
        device=device,
    )


    

