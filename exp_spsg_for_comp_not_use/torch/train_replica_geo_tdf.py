import torch
import numpy as np
from glob import glob
import os, sys, tqdm, glob, random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import GeoGeneratorTDF
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

def save_scene (save_dir, mask_gen, mask_roi, 
                input_oc, gt_oc, pd_oc, 
                input_df=None, gt_df=None, pd_df=None,
                oc_save_index=False):
    os.makedirs(save_dir, exist_ok=True)
    cmap = PseudoColorConverter('viridis', 0.0, 5.0)
    # save input occupancy (view i and j in ROI)
    if input_oc != None:
        with open(f'{save_dir}/pts_oc_in.txt', 'w') as f:
            for pt in np.stack(np.nonzero(input_oc.detach().cpu().numpy()==1), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [176,224,230])) # Tableau blue
    # # save mask_gen (view k)
    # if mask_gen != None:
    #     with open(f'{save_dir}/pts_mask_gen.txt', 'w') as f:
    #         for pt in np.stack(np.nonzero(mask_gen.detach().cpu().numpy()), axis=-1):
    #             f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau purple
    # save ground-truth occupancy (on view k)
    if gt_oc != None:
        with open(f'{save_dir}/pts_oc_gt.txt', 'w') as f:
            for pt in np.stack(np.nonzero((gt_oc.detach().cpu().numpy() == 1) & mask_gen.detach().cpu().numpy()), axis=-1):
                f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple
    # save pred occupancy
    if pd_oc != None:
        if not oc_save_index:
            with open(f'{save_dir}/pts_oc_pd_all.txt', 'w') as f:
                for pt in np.stack(np.nonzero((pd_oc.detach().cpu().numpy() == 1)), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,255,224])) # Tableau light-yellow
            with open(f'{save_dir}/pts_oc_pd_genview.txt', 'w') as f:
                for pt in np.stack(np.nonzero((pd_oc.detach().cpu().numpy() == 1) & mask_gen.detach().cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
            with open(f'{save_dir}/pts_oc_pd_roi.txt', 'w') as f:
                for pt in np.stack(np.nonzero((pd_oc.detach().cpu().numpy() == 1) & mask_roi.detach().cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,255,0])) # Tableau yellow
        else:
            gen_np = np.stack(np.nonzero((pd_oc.detach().cpu().numpy() == 1) & mask_gen.detach().cpu().numpy()), axis=-1)
            roi_np = np.stack(np.nonzero((pd_oc.detach().cpu().numpy() == 1) & mask_roi.detach().cpu().numpy()), axis=-1)
            np.savez_compressed(f'{save_dir}/pts_ocindex_pd.npz', gen=gen_np, roi=roi_np)

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

    def __init__(self, npz_folder, max_spatial, phase):
        self.max_spatial = np.array(max_spatial)
        self.phase = phase
        if phase == 'train':
            self.files = sorted(glob.glob(npz_folder + '/*.npz'))
        elif phase == 'val':
            self.files = sorted(glob.glob(npz_folder + '/*.npz'))
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, save_pc=False, save_dir=None):
        npz = np.load(self.files[idx], allow_pickle=True)
        # 0 free space
        # 1 occupied
        # 2 unobserved
        # 3 not relevant
        # input_oc = torch.from_numpy(npz['input_oc']) # i and j
        input_df = 10 * torch.from_numpy(npz['input_df']) ** 2 # by voxel, i and j
        input_oc = input_df <= (np.sqrt(2) * 0.5) # 0.707
        # output_oc = torch.from_numpy(npz['output_oc']) # i and j and k
        output_df = 10 * torch.from_numpy(npz['output_df']) ** 2 # by voxel, i and j and k, out of ijk is negative number
        output_oc = output_df <= (np.sqrt(2) * 0.5) # 0.707
        mask_roi = torch.from_numpy(npz['mask_roi']) # view i and j and k
        mask_gen = torch.from_numpy(npz['mask_gen']) # view k but not in i and j

        input_df = torch.clamp(input_df, 0.0, 5.0)
        output_df = torch.clamp(output_df, 0.0, 5.0)

        d = {}
        d['input_oc'] = input_oc.unsqueeze(0).int()
        d['input_df'] = input_df.unsqueeze(0).float()
        d['output_oc'] = output_oc.unsqueeze(0).int()
        d['output_df'] = output_df.unsqueeze(0).float()
        d['mask_roi'] = mask_roi.unsqueeze(0).bool() # all the voxels in the views
        d['mask_gen'] = mask_gen.unsqueeze(0).bool() # unobserved 2/3 in ij but labelled 0/1/2 in k
        d['idx'] = idx
        d['basename'] = os.path.basename(self.files[idx]).split('.')[0]

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

class DFLoss (torch.nn.Module):
    def __init__ (self):
        return
        
    def log_transform(self, df):
        return torch.sign(df) * torch.log(torch.abs(df) + 1)

    def forward(self, gt, pd, weights=None, masks=None):
        log_gt = self.log_transform(gt)
        log_pd = self.log_transform(pd)
        diff = log_gt - log_pd
        loss = 0.0
        for weight, mask in zip(weights, masks):
            loss += (weight * torch.abs(diff[mask]).mean())
        return loss

class OCLoss (torch.nn.Module):
    def __init__ (self, loss_type, gamma=2):
        self.loss_type = loss_type
        self.gamma = gamma

    def oc_sub_loss(self, gt, pd):
        num_0 = (gt == 0).sum()
        num_1 = (gt == 1).sum()
        # weight = 0.5 / num_0 * (gt == 0).float() + 0.5 / num_1 * (gt == 1).float()
        return torch.nn.functional.binary_cross_entropy_with_logits(pd, gt.float(), pos_weight=(num_0/num_1))

    def oc_sub_focal_loss(self, gt, pd):
        num_0 = (gt == 0).sum()
        num_1 = (gt == 1).sum()
        alpha = (gt == 0).float() + (gt == 1).float() * (num_0 / num_1)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pd, gt.float(), reduction='none')
        prob = torch.exp(-bce_loss)
        focal_loss =  alpha * (1 - prob)**self.gamma * bce_loss
        return focal_loss.mean()
        
    def forward (self, gt, pd, weights=None, masks=None):
        loss = 0.0
        for weight, mask in zip(weights, masks):
            if self.loss_type == 'bce':
                loss += self.oc_sub_loss(gt[mask], pd[mask]) * weight
            elif self.loss_type == 'focal':
                loss += self.oc_sub_focal_loss(gt[mask], pd[mask]) * weight
        return loss

def forward(model, data, df_loss_fn, oc_loss_fn, oc_input=False, save_pc=False, save_dir=None):
    # input_oc not used
    in_oc = data['input_oc'].cuda()
    in_df = data['input_df'].cuda()
    gt_oc = data['output_oc'].cuda()
    gt_df = data['output_df'].cuda()
    mask_roi = data['mask_roi'].cuda()
    mask_gen = data['mask_gen'].cuda()
    idx = data['idx']
    basename = data['basename']

    masks = [mask_roi, mask_gen]
    weights = [1.0, 5.0]
    # weights = [0.5, 1.0]

    if oc_input:
        net_in = torch.cat([in_df, mask_roi, in_oc], dim=1)    # bs x 2 x 144 x 64 x 160
    else:
        net_in = torch.cat([in_df, mask_roi], dim=1)    # bs x 2 x 144 x 64 x 160
    oc_logits, pd_df = model(net_in)   # bs x 1 x 144 x 64 x 160

    df_loss = df_loss_fn.forward(gt_df, pd_df, weights, masks)
    oc_loss = oc_loss_fn.forward(gt_oc, oc_logits, weights, masks)
    loss = df_loss + oc_loss

    # pd_geo = (pd_oc > 0).int()
    # pd_df = pd_df ** 2
    # gt_df = gt_df ** 2
    # pd_geo_by_df = pd_df < 1.7
    # gt_df_by_oc = gt_df[gt_oc==1]
    # print(gt_df_by_oc.min(), gt_df_by_oc.mean(), gt_df_by_oc.max(), (gt_df_by_oc<=0.0707).float().mean())

    # print(gt_df.min(), gt_df.max())
    # print(pd_df.min(), pd_df.max())
    # print(pd_geo.min(), pd_geo.max())
    # print(pd_geo.sum().item(), pd_geo_by_df.sum().item(), (gt_df<1.0).sum().item())
    
    pd_oc = oc_logits > 0
    pd_oc_by_df = pd_df <= (np.sqrt(2) * 0.5) # 0.707
    iou1, p1, r1 = compute_iou_pr(pd_oc[mask_gen] == 1, gt_oc[mask_gen] == 1)
    iou2, p2, r2 = compute_iou_pr(pd_oc_by_df[mask_gen] == 1, gt_oc[mask_gen] == 1)
    # print('Pred_OC:', (iou1, p1, r1), 'Pred_OC_By_DF:', (iou2, p2, r2))

    # bs, _, h, w, d = oc_logits.shape
    # thres_num = int(0.2 * h * w * d)
    # oc_logits_viewed = oc_logits.reshape(bs, -1)
    # oc_logits_orders = torch.argsort(oc_logits_viewed, dim=1, descending=True)
    # thres = oc_logits_viewed[:, oc_logits_orders[]]

    if save_pc:
        os.makedirs(save_dir, exist_ok=True)
        for bidx in range(net_in.shape[0]):
            save_scene(
                        f'{save_dir}/{basename[bidx]}', 
                        mask_gen=mask_gen[bidx, 0], mask_roi=mask_roi[bidx, 0],
                        input_oc=None, gt_oc=None, pd_oc=pd_geo[bidx, 0], 
                        oc_save_index=True
                      )
            # save_scene(
            #             f'{save_dir}/{bidx}', 
            #             mask_gen=mask_gen[bidx, 0], mask_roi=mask_roi[bidx, 0],
            #             input_oc=None, gt_oc=gt_oc[bidx, 0], pd_oc=pd_geo[bidx, 0]
            #           )
    return loss, df_loss, oc_loss, (iou1, p1, r1), (iou2, p2, r2), pd_df, pd_oc

def train_model(model, data_loader, optimizer, df_loss_fn, oc_loss_fn, oc_input=False,
                val_data_loader=None, load_ckpt_from=None, save_label=''):

    epoch_load, it_load = 0, -1
    if load_ckpt_from is not None:
        d = torch.load(load_ckpt_from)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])
        print(f'Load checkpoint from {load_ckpt_from}.')
        epoch_load, it_load = d['epoch_it']
        
    ckpt_save_dir = f'ckpt_replica_geo_tdf'
    if save_label:
        ckpt_save_dir = ckpt_save_dir + '_' + save_label
    os.makedirs(ckpt_save_dir, exist_ok=True)

    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
    
    for epoch in range(epoch_load, 100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            if epoch <= epoch_load and it <= it_load:
                continue

            optimizer.zero_grad()
            loss, df_loss, oc_loss, (iou1, p1, r1), (iou2, p2, r2), pd_df, pd_oc = forward(model, data, df_loss_fn, oc_loss_fn,
                                                                                            oc_input=oc_input)
                                                                        # save_pc=True, save_dir=pt_save_dir)

            print(f'Epoch {epoch} Iter {it} {loss.item():.3f} {df_loss.item():.3f} {oc_loss.item():.3f} {iou1:.3f} {p1:.3f} {r1:.3f} {iou2:.3f} {p2:.3f} {r2:.3f}')

            loss.backward()
            optimizer.step()

            if it % 1000 == 0 and it != 0:
                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(d, f'{ckpt_save_dir}/ckpt_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                if val_data_loader is not None:
                    print(f'Val Epoch {epoch} Iter {it}', end=' ')
                    val_data = next(val_data_iter, None)
                    if val_data is None:
                        val_data_iter = iter(val_data_loader)
                        val_data = next(val_data_iter, None)
                    val_model(model, val_data, df_loss_fn, oc_loss_fn, oc_input)
            
            sys.stdout.flush()
    return

def val_model(model, data, df_loss_fn, oc_loss_fn, oc_input=False):
    with torch.no_grad():
        loss, df_loss, oc_loss, (iou1, p1, r1), (iou2, p2, r2), pred_geo_y, pred_df_out = forward(model, data, df_loss_fn, oc_loss_fn, oc_input)
        print(f'{loss.item():.3f} {df_loss.item():.3f} {oc_loss.item():.3f} {iou1:.3f} {p1:.3f} {r1:.3f} {iou2:.3f} {p2:.3f} {r2:.3f}')

def test_model(model, data_loader, load_ckpt_from, df_loss_fn, oc_loss_fn, oc_input=False, save_label=''):
    print(f'Load checkpoint from {load_ckpt_from}.')
    model.load_state_dict(torch.load(load_ckpt_from)['model_state_dict'])
    model.eval()
    
    pt_save_dir = f'pts_tdf'
    if save_label:
        pt_save_dir = pt_save_dir + '_' + save_label

    avg_iou = 0
    avg_pre = 0
    avg_rec = 0

    with torch.no_grad():
        for it, data in enumerate(tqdm.tqdm(data_loader)):
            loss, df_loss, oc_loss, (iou1, p1, r1), (iou2, p2, r2), pd_df, pd_oc = forward(model, data, df_loss_fn, oc_loss_fn, 
                                                                        oc_input=oc_input, save_pc=False, save_dir=pt_save_dir)
            print(f'Test Loss {loss.item():.6f} {df_loss.item():.6f} {oc_loss.item():.6f}')
            print(f'Test IoU (Pred_OC) {iou1:.4f}, Pre {p1:.4f}, Rec {r1:.4f}')
            print(f'Test IoU (Pred_OC_by_DF) {iou2:.4f}, Pre {p2:.4f}, Rec {r2:.4f}')
            avg_iou += iou1
            avg_pre += p1
            avg_rec += r1
    
    avg_iou /= len(data_loader)
    avg_pre /= len(data_loader)
    avg_rec /= len(data_loader)
    print(f'Avg test IoU {avg_iou:.4f}, Pre {avg_pre:.4f}, Rec {avg_rec:.4f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, choices=['bce', 'focal'])
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--oc_input", action="store_true", default=False)
    parser.add_argument("--focal_gamma", type=int, default=2)
    parser.add_argument("--start_from_ckpt", type=str, default=None)
    parser.add_argument("--extra_label", type=str, default=None)
    args = parser.parse_args()

    save_label = f'{args.loss}'
    if args.loss == 'focal' and args.focal_gamma != 2:
        save_label = save_label + f'_gamma{args.focal_gamma}'
    if args.lr != 1e-4:
        save_label = save_label + f'_lr{args.lr:.0e}'
    if args.oc_input:
        save_label = save_label + f'_ocinp'
    if args.extra_label != None:
        save_label = save_label + f'_{args.extra_label}'

    train_data = ReplicaGenGeoDFDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/train_easy', 
                                        [144,64,160], 'train')
    val_data = ReplicaGenGeoDFDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/val_easy', 
                                        [144,64,160], 'val')

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)#, num_workers=24)
    val_dataloader = DataLoader(val_data, batch_size=12, shuffle=False)#, num_workers=24)

    nf_in_geo = 2 if not args.oc_input else 3
    geo_generator = GeoGeneratorTDF(nf_in_geo=nf_in_geo, nf=64, max_data_size=[144,64,160]).cuda()
    # optimizer = torch.optim.Adam(geo_generator.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(geo_generator.parameters(), momentum=0.9, lr=args.lr)

    df_loss_fn = DFLoss()
    oc_loss_fn = OCLoss(loss_type=args.loss, gamma=args.focal_gamma)

    ckpt_save_dir = f'ckpt_replica_geo_tdf'
    if save_label:
        ckpt_save_dir = ckpt_save_dir + '_' + save_label
    if os.path.exists(ckpt_save_dir) and len(os.listdir(ckpt_save_dir)) != 0:
        if args.start_from_ckpt != None:
            ckpt_file = os.path.join(ckpt_save_dir, args.start_from_ckpt)
        else:
            ckpt_file = os.path.join(ckpt_save_dir, sorted(os.listdir(ckpt_save_dir))[-1])
    else:
        ckpt_file = None

    if not args.only_test:
        train_model(geo_generator, train_dataloader, optimizer, df_loss_fn, oc_loss_fn, oc_input=args.oc_input,
                    val_data_loader=val_dataloader, load_ckpt_from=ckpt_file, save_label=save_label)
    else:
        test_model(geo_generator, val_dataloader, load_ckpt_from=ckpt_file, oc_input=args.oc_input,
                   df_loss_fn=df_loss_fn, oc_loss_fn=oc_loss_fn, save_label=save_label+'_val_easy_pred')
        test_model(geo_generator, train_dataloader, load_ckpt_from=ckpt_file, oc_input=args.oc_input,
                   df_loss_fn=df_loss_fn, oc_loss_fn=oc_loss_fn, save_label=save_label+'_train_easy_pred')