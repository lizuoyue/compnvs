import torch
import numpy as np
import os, sys, tqdm, glob, random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import GeoGeneratorTDF
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

def save_scene (save_dir, input_oc, input_df, gt_oc, gt_df, pd_oc, pd_df, mask_gen):
    os.makedirs(save_dir, exist_ok=True)
    cmap = PseudoColorConverter('viridis', 0.0, 5.0)
    # save input_oc
    if input_oc != None:
        with open(f'{save_dir}/pts_oc_in0.txt', 'w') as f:
            for pt in np.stack(np.nonzero(input_oc.detach().cpu().numpy()==0), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
        with open(f'{save_dir}/pts_oc_in1.txt', 'w') as f:
            for pt in np.stack(np.nonzero(input_oc.detach().cpu().numpy()==1), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
        with open(f'{save_dir}/pts_oc_in2.txt', 'w') as f:
            for pt in np.stack(np.nonzero(input_oc.detach().cpu().numpy()==2), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
    # save mask_gen
    if mask_gen != None:
        with open(f'{save_dir}/pts_mask_gen.txt', 'w') as f:
            for pt in np.stack(np.nonzero(mask_gen.detach().cpu().numpy()), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple
    # save gt_oc
    if gt_oc != None:
        with open(f'{save_dir}/pts_oc_gt0.txt', 'w') as f:
            for pt in np.stack(np.nonzero((gt_oc.detach().cpu().numpy() == 0) & mask_gen.detach().cpu().numpy()), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
        with open(f'{save_dir}/pts_oc_gt1.txt', 'w') as f:
            for pt in np.stack(np.nonzero((gt_oc.detach().cpu().numpy() == 1) & mask_gen.detach().cpu().numpy()), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
    # with open(f'{save_dir}/pts_out2.txt', 'w') as f:
    #     for pt in np.stack(np.nonzero((gt_oc.numpy() == 2) & mask_gen.numpy()), axis=-1):
    #         f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
    # save pd_oc
    if pd_oc != None:
        with open(f'{save_dir}/pts_oc_pd0.txt', 'w') as f:
            for pt in np.stack(np.nonzero((pd_oc.detach().cpu().numpy() == 0) & mask_gen.detach().cpu().numpy()), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
        with open(f'{save_dir}/pts_oc_pd1.txt', 'w') as f:
            for pt in np.stack(np.nonzero((pd_oc.detach().cpu().numpy() == 1) & mask_gen.detach().cpu().numpy()), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
    # save input_df
    if input_df != None:
        df_input = input_df.detach().cpu().numpy()
        pts = np.stack(np.nonzero(df_input < 5), axis=-1)
        rgb = cmap.convert(df_input[df_input < 5])
        with open(f'{save_dir}/pts_df_in.txt', 'w') as f:
            for pt, (r, g, b) in zip(pts, rgb):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
    # save gt_df
    if gt_df != None:
        df_output = gt_df.detach().cpu().numpy()
        vis_mask = (0.0 <= df_output) & (df_output < 5)
        pts = np.stack(np.nonzero(vis_mask), axis=-1)
        rgb = cmap.convert(df_output[vis_mask])
        with open(f'{save_dir}/pts_df_gt.txt', 'w') as f:
            for pt, (r, g, b) in zip(pts, rgb):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
    # save pd_df
    if pd_df != None:
        df_output = pd_df.detach().cpu().numpy()
        vis_mask = (0.0 <= df_output) & (df_output < 5)
        pts = np.stack(np.nonzero(vis_mask), axis=-1)
        rgb = cmap.convert(df_output[vis_mask])
        with open(f'{save_dir}/pts_df_pd.txt', 'w') as f:
            for pt, (r, g, b) in zip(pts, rgb):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
    quit()
    return 

class PseudoColorConverter(object):
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = matplotlib.pyplot.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        return

    def convert(self, val):
        return np.round(self.scalarMap.to_rgba(val)[..., :3] * 255).astype(np.uint8)

class ReplicaGenGeoDFDataset(Dataset):

    def __init__(self, npz_folder, max_spatial):
        self.files = sorted(glob.glob(npz_folder + '/*.npz'))
        self.max_spatial = np.array(max_spatial)
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, save_pc=False, save_dir=None):
        npz = np.load(self.files[idx], allow_pickle=True)
        # 0 free space
        # 1 occupied
        # 2 unobserved
        # 3 not relevant
        input_oc = torch.from_numpy(npz['input_oc']) # i and j
        input_df = torch.from_numpy(npz['input_df']) # i and j
        output_oc = torch.from_numpy(npz['output_oc']) # i and j and k
        output_df = torch.from_numpy(npz['output_df']) # i and j and k, out of ijk is negative number
        mask_roi = torch.from_numpy(npz['mask_roi']) # view i and j and k
        mask_gen = torch.from_numpy(npz['mask_gen']) # view k but not in i and j

        input_df[input_df >= 0.5] = 0.5
        input_df *= 10

        output_df[output_df < -0.01] = 0.5
        output_df[output_df >= 0.5] = 0.5
        output_df *= 10

        # input_oc not used
        output_oc[output_oc >= 2] = 0

        d = {}
        # d['input_oc'] = input_oc.unsqueeze(0).int()
        d['input_df'] = input_df.unsqueeze(0).float()
        d['output_oc'] = output_oc.unsqueeze(0).int()
        d['output_df'] = output_df.unsqueeze(0).float()
        d['mask_roi'] = mask_roi.unsqueeze(0).bool() # all the voxels in the views
        d['mask_gen'] = mask_gen.unsqueeze(0).bool() # unobserved 2/3 in ij but labelled 0/1/2 in k
        d['idx'] = idx

        if save_pc:
            save_scene(
                        save_dir, 
                        input_oc=None, input_df=input_df, 
                        gt_oc=output_oc, gt_df=output_df, 
                        pd_oc=None, pd_df=None,
                        mask_gen=mask_gen
                      )

        return d

def compute_iou_pr(pd, gt):
    assert(pd.shape == gt.shape)
    pd_bool = pd.cpu().numpy().astype(np.bool)
    gt_bool = gt.cpu().numpy().astype(np.bool)
    i = (pd_bool & gt_bool).sum()
    u = (pd_bool | gt_bool).sum()
    r = pd_bool[gt_bool].mean()
    p = gt_bool[pd_bool].mean()
    return i / u, p, r

def df_loss_fn(gt, pd, weights=None, masks=None):
    def log_transform(df):
        return torch.sign(df) * torch.log(torch.abs(df) + 1)
    log_gt = log_transform(gt)
    log_pd = log_transform(pd)
    diff = log_gt - log_pd
    loss = 0.0
    for weight, mask in zip(weights, masks):
        loss += (weight * torch.abs(diff[mask]).mean())
    return loss

def oc_loss_fn(gt, pd, weights=None, masks=None):
    def oc_sub_loss(gt, pd):
        num_0 = (gt == 0).sum()
        num_1 = (gt == 1).sum()
        # weight = 0.5 / num_0 * (gt == 0).float() + 0.5 / num_1 * (gt == 1).float()
        return torch.nn.functional.binary_cross_entropy_with_logits(pd, gt.float(), pos_weight=(num_0/num_1))
    loss = 0.0
    for weight, mask in zip(weights, masks):
        loss += oc_sub_loss(gt[mask], pd[mask]) * weight
    return loss

def oc_focal_loss_fn(gt, pd, weights=None, masks=None):
    def oc_sub_focal_loss(gt, pd, gamma=2):
        num_0 = (gt == 0).sum()
        num_1 = (gt == 1).sum()
        weight = 0.5 / num_0 * (gt == 0).float() + 0.5 / num_1 * (gt == 1).float()
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pd, gt.float(), reduction='none')
        prob = torch.exp(-bce_loss)
        focal_loss = weight * (1 - prob)**gamma * bce_loss
        return focal_loss.mean()
    loss = 0.0
    for weight, mask in zip(weights, masks):
        loss += oc_sub_focal_loss(gt[mask], pd[mask]) * weight
    return loss

def forward(model, data, save_pc=False, save_dir=None):
    # input_oc not used
    net_in = data['input_df'].cuda()
    gt_oc = data['output_oc'].cuda()
    gt_df = data['output_df'].cuda()
    mask_roi = data['mask_roi'].cuda()
    mask_gen = data['mask_gen'].cuda()
    idx = data['idx']

    masks = [torch.ones_like(mask_roi).bool(), mask_roi, mask_gen]
    weights = [0.1, 1.0, 5.0]

    pd_oc, pd_df = model(net_in)    # only takes df as input

    df_loss = df_loss_fn(gt_df, pd_df, weights, masks)
    oc_loss = oc_loss_fn(gt_oc, pd_oc, weights, masks)
    loss = df_loss + oc_loss

    # print(pd_oc.min(), pd_oc.max())
    pd_geo = (pd_oc > 0).int()
    # print(pd_geo.min(), pd_geo.max(), pd_geo.sum())
    iou, p, r = compute_iou_pr(pd_geo[mask_gen] == 1, gt_oc[mask_gen] == 1)

    if save_pc:
        for bidx in range(net_in.shape[0]):
            save_scene(
                        f'{save_dir}/{bidx}', 
                        input_oc=None, input_df=net_in[bidx, 0], 
                        gt_oc=gt_oc[bidx, 0], gt_df=gt_df[bidx, 0], 
                        pd_oc=pd_geo[bidx, 0], pd_df=pd_df[bidx, 0], 
                        mask_gen=mask_gen[bidx, 0]
                      )
    return loss, df_loss, oc_loss, (iou, p, r), pd_df, pd_oc



def train_model(model, data_loader, optimizer, ckpt_path=None, val_data_loader=None):

    epoch_load, it_load = 0, -1
    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])
        epoch_load, it_load = d['epoch_it']
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
    
    for epoch in range(epoch_load, 100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            if epoch <= epoch_load and it <= it_load:
                continue

            optimizer.zero_grad()
            loss, df_loss, oc_loss, (iou, p, r), pd_df, pd_oc = forward(model, data)

            print(f'Epoch {epoch} Iter {it} {loss.item():.6f} {df_loss.item():.6f} {oc_loss.item():.6f} {iou:.3f} {p:.3f} {r:.3f}', data['idx'].numpy())

            loss.backward()
            optimizer.step()

            if it % 1000 == 0:

                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(d, f'ckpt_replica_geo_tdf/ckpt_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                if val_data_loader is not None:
                    print(f'Val Epoch {epoch} Iter {it}', end=' ')
                    val_data = next(val_data_iter, None)
                    if val_data is None:
                        val_data_iter = iter(val_data_loader)
                        val_data = next(val_data_iter, None)
                    val_model(model, val_data)
            
            sys.stdout.flush()
    return

def val_model(model, data):
    with torch.no_grad():
        loss, df_loss, geo_loss, (iou, p, r), pred_geo_y, pred_df_out = forward(model, data)
        print(f'{loss.item():.6f} {df_loss.item():.6f} {geo_loss.item():.6f} {iou:.6f} {p:.6f} {r:.6f}')

def test_model(model, data_loader, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()
    
    with torch.no_grad():

        for it, data in enumerate(tqdm.tqdm(data_loader)):
            loss, df_loss, oc_loss, (iou, p, r), pd_df, pd_oc = forward(model, data, save_pc=False, save_dir='pts_tdf')
            print(f'Test Loss {loss.item():.6f} {df_loss.item():.6f} {oc_loss.item():.6f}')
            print(f'Test IoU {iou:.6f}')
            print(f'Test Pre {p:.6f}')
            print(f'Test Rec {r:.6f}')
            break


if __name__ == '__main__':

    train_data = ReplicaGenGeoDFDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/train_easy', [144,64,160])
    val_data = ReplicaGenGeoDFDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/val_easy', [144,64,160])

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)#, num_workers=24)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)#, num_workers=24)

    geo_generator = GeoGeneratorTDF(nf_in_geo=1, nf=64, max_data_size=[144,64,160]).cuda()
    optimizer = torch.optim.Adam(geo_generator.parameters(), lr=1e-4)

    train_model(geo_generator, train_dataloader, optimizer, None, val_dataloader)

    # test_model(geo_generator, val_dataloader, 'ckpt_replica_geo_tdf/ckpt_0_2000.pt')
# /home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/train_easy/00_180_068_239.npz False
# /home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/train_easy/01_180_068_239.npz False