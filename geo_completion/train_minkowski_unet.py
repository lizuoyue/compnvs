import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, glob, tqdm, os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from minkowski_network import MinkowskiDiscriminator
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C
from minkowski_utils import replace_features
from minkowski_loss_fn import minkowski_masked_l1_loss
from minkowski_loss_fn import minkowski_binary_loss

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
            data[0][key] = np.concatenate([data[0][key], d[key]])
    return {k: torch.from_numpy(data[0][k]) for k in data[0]}

class ReplicaGenDataset(Dataset):

    def __init__(self, npz_folder, sid_list=[]):
        self.npz_folder = npz_folder
        self.files = []
        for sid in sid_list:
            self.files += sorted(glob.glob(npz_folder + f'/{sid:02d}_*.npz'))
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
    vertex_idx = data['vertex_idx'].int().to(device)
    vertex_input = data['vertex_input'].float().to(device)
    vertex_output = data['vertex_output'].float().to(device)

    real_x = ME.SparseTensor(coordinates=vertex_idx, features=vertex_input)
    real_y = replace_features(real_x, vertex_output)
    mask_fill = replace_features(real_x, vertex_input[:, -1:])
    mask_keep = replace_features(real_x, 1 - vertex_input[:, -1:])

    # Forward
    with torch.no_grad():
        fake_y = gen(real_x)
        fake_y = replace_features(fake_y, fake_y.F * mask_fill.F + real_y.F * mask_keep.F)

        pred_fake = dis(ME.cat(fake_y, mask_fill))
        loss_dis_fake = dis_loss_fn(pred_fake, False)

        pred_real = dis(ME.cat(real_y, mask_fill))
        loss_dis_real = dis_loss_fn(pred_real, True)

        loss_dis = (loss_dis_fake + loss_dis_real) * 0.5

        pred_fake = dis(ME.cat(fake_y, mask_fill))
        loss_gen_gan = dis_loss_fn(pred_fake, True)

        loss_gen_err = gen_loss_fn(fake_y, real_y, mask_fill) 

        loss_gen = loss_gen_gan + loss_gen_err

    print(f'Val Epoch {epoch} Iter {it}: {loss_dis.item():.4f} {loss_gen_gan.item():.4f} {loss_gen_err.item():.4f}')
    return

def train_model(
    gen, gen_loss_fn, gen_optm,
    dis, dis_loss_fn, dis_optm,
    data_loader, val_data_loader=None,
    ckpt_path=None, save_path='./', device='cpu'):

    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        gen.load_state_dict(d['model_state_dict']['gen'])
        dis.load_state_dict(d['model_state_dict']['dis'])
        gen_optm.load_state_dict(d['optimizer_state_dict']['gen'])
        dis_optm.load_state_dict(d['optimizer_state_dict']['dis'])

    os.makedirs(save_path, exist_ok=True)
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)

    for epoch in range(100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):
            
            # Data
            vertex_idx = data['vertex_idx'].int().to(device)
            vertex_input = data['vertex_input'].float().to(device)
            vertex_output = data['vertex_output'].float().to(device)

            real_x = ME.SparseTensor(coordinates=vertex_idx, features=vertex_input)
            real_y = replace_features(real_x, vertex_output)
            mask_fill = replace_features(real_x, vertex_input[:, -1:])
            mask_keep = replace_features(real_x, 1 - vertex_input[:, -1:])

            # Forward
            fake_y = gen(real_x) # G(x)
            fake_y = replace_features(fake_y, fake_y.F * mask_fill.F + real_y.F * mask_keep.F)

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
            loss_gen_err = gen_loss_fn(fake_y, real_y, mask_fill)
            # combine loss and calculate gradients
            loss_gen = loss_gen_gan + loss_gen_err
            loss_gen.backward()
            gen_optm.step() # udpate G's weights

            print(f'Train Epoch {epoch} Iter {it}: {loss_dis.item():.4f} {loss_gen_gan.item():.4f} {loss_gen_err.item():.4f}')            

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

    rm_sid = [13,14,19,20,21,42]
    all_sid = [i for i in range(48) if i not in rm_sid]
    val_sid = [0,1,34,35,36]
    train_sid = [i for i in all_sid if i not in val_sid]

    npz_folder = '../replica_habitat_gen/ReplicaGenFtTriplets/easy/npz_rgba_init'
    train_data = ReplicaGenDataset(npz_folder, train_sid)
    val_data = ReplicaGenDataset(npz_folder, val_sid)

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_data_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = 'cuda:0'
    emb_size, mid_size = 16, 64
    gen = MinkUNet34C(emb_size + 1, emb_size).to(device)
    dis = MinkowskiDiscriminator(emb_size + 1, mid_size).to(device)
    
    gen_criterion = minkowski_masked_l1_loss
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)

    dis_criterion = minkowski_binary_loss
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-4)

    save_path = 'ckpt_minkowski_unet34c_'
    save_path += npz_folder.split('/')[-1].split('_')[-1]
    train_model(
        gen, gen_criterion, gen_optimizer,
        dis, dis_criterion, dis_optimizer,
        data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        ckpt_path=None,
        save_path=save_path,
        device=device,
    )


    

