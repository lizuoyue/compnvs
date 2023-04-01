import torch
import numpy as np
import os, tqdm, glob, random
from model import GeoGenerator

class GeoDataLoader(object):

    def __init__(self, npz_folder, batch_size, max_spatial=None):
        self.npz_files = sorted(glob.glob(npz_folder))
        self.train_files = []
        self.val_files = []
        for i, file in enumerate(self.npz_files):
            if i % 250 < 240:
                self.train_files.append(file)
            else:
                self.val_files.append(file)
        print(f'Total number of .npz files: {len(self.npz_files)}.')
        print(f'# Train: {len(self.train_files)}')
        print(f'# Val:   {len(self.val_files)}')
        self.batch_size = batch_size
        if max_spatial is not None:
            self.max_spatial = np.array(max_spatial)
        else:
            spatial = []
            for file in tqdm.tqdm(self.npz_files):
                d = np.load(file)
                spatial.append(d['vertex_idx'].max(axis=0, keepdims=True))
                # print(d['vertex_info'][:,-1].mean())
            self.max_spatial = np.concatenate(spatial_shapes).max(axis=0)
        
        self.e = np.array([
            self.max_spatial[0] * self.max_spatial[1] * self.max_spatial[2],
            self.max_spatial[1] * self.max_spatial[2],
            self.max_spatial[2],
            1
        ], np.int32)
        self.reset_train_files_order()
        self.val_idx = 0
        return
    
    def reset_train_files_order(self):
        self.train_idx = 0
        random.shuffle(self.train_files)
        return
    
    # def reset_val_files_order(self):
    #     self.val_idx = 0
    #     random.shuffle(self.val_files)
    #     return

    def get_data(self, files):

        self.current_fids = [os.path.basename(file).replace('.npz', '') for file in files]

        occ_indices, uob_indices = [], []
        occ_updates, uob_updates = [], []
        for b, file in enumerate(files):
            print(file)
            d = np.load(file.replace('/npz/', '/geo/'))
            occ_indices.append(np.concatenate([
                np.ones((d['is_occupied'].shape[0], 1)) * b,
                d['is_occupied'],
            ], axis=1))
            occ_updates.append(np.ones((d['is_occupied'].shape[0], 1)))
            uob_indices.append(np.concatenate([
                np.ones((d['is_unobserved'].shape[0], 1)) * b,
                d['is_unobserved'],
            ], axis=1))
            uob_updates.append(np.ones((d['is_unobserved'].shape[0], 1)))
        
        occ_indices = np.concatenate(occ_indices, axis=0).astype(np.int32)
        # occ_updates = np.concatenate(occ_updates, axis=0).astype(np.float32)
        uob_indices = np.concatenate(uob_indices, axis=0).astype(np.int32)
        # uob_updates = np.concatenate(uob_updates, axis=0).astype(np.float32)
        # print(occ_indices.shape, uob_indices.shape)

        occ_grid = np.zeros((self.batch_size * self.max_spatial[0] * self.max_spatial[1] * self.max_spatial[2]), np.float32)
        uob_grid = np.zeros((self.batch_size * self.max_spatial[0] * self.max_spatial[1] * self.max_spatial[2]), np.float32)

        occ_indices_1d = occ_indices.dot(self.e)
        uob_indices_1d = uob_indices.dot(self.e)

        occ_grid[occ_indices_1d] = 1
        uob_grid[uob_indices_1d] = 1

        # occ_grid = tf.scatter_nd(
        #     occ_indices,
        #     occ_updates,
        #     shape=np.array([self.batch_size] + self.max_spatial.tolist() + [1]),
        # )

        # uob_grid = tf.scatter_nd(
        #     uob_indices,
        #     uob_updates,
        #     shape=np.array([self.batch_size] + self.max_spatial.tolist() + [1]),
        # )

        # print(occ_grid.shape)
        # print(uob_grid.shape)
        # print(occ_grid.sum(), uob_grid.sum())
        # print((occ_grid * uob_grid).sum())
        # return
        # weight = (occ_grid * uob_grid).sum()

        ch1 = (1 - uob_grid) * occ_grid + uob_grid * 0.5
        ch2 = uob_grid
        train_x = np.stack([ch1, ch2], axis=-1).reshape((self.batch_size, self.max_spatial[0], self.max_spatial[1], self.max_spatial[2], 2))
        train_y = occ_grid.reshape((self.batch_size, self.max_spatial[0], self.max_spatial[1], self.max_spatial[2], 1))

        mask = train_x[..., 1] > 0.5
        info = {
            'mask': mask, # unvisible all
            'num_1': train_y.flatten()[mask.flatten()].sum(),
            'num_0': (1-train_y.flatten()[mask.flatten()]).sum(),
        }

        return train_x, train_y, info

    def get_train_data(self):
        files = self.train_files[self.train_idx: self.train_idx + self.batch_size]

        self.train_idx += self.batch_size
        if self.train_idx >= len(self.train_files):
            self.reset_train_files_order()
        
        return self.get_data(files)


    def get_val_data(self):

        if self.val_idx >= len(self.val_files):
            self.val_idx = 0
            return None
        
        files = self.val_files[self.val_idx: self.val_idx + self.batch_size]
        self.val_idx += self.batch_size

        return self.get_data(files)
    
    def has_val(self):
        return self.val_idx < len(self.val_files)


def compute_iou_pr(pd, gt):
    assert(pd.shape == gt.shape)
    pd_bool = pd.cpu().numpy().astype(np.bool)
    gt_bool = gt.cpu().numpy().astype(np.bool)
    i = (pd_bool & gt_bool).sum()
    u = (pd_bool | gt_bool).sum()
    r = pd_bool[gt_bool].mean()
    p = gt_bool[pd_bool].mean()
    return i / u, p, r



def train_model(model, data_loader, loss_fn, optimizer):
    for it in range(100000000):

        train_x, train_y, info = data_loader.get_train_data()

        optimizer.zero_grad()
        # print(info['num_0'], info['num_1'])
        # input('press')
        # continue

        # verify
        # for i in range(5):
        #     with open(f'pc/pc{i}_1.txt', 'w') as f:
        #         for pt in pts[sel1]:
        #             f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))
            
        #     with open(f'pc/pc{i}_2.txt', 'w') as f:
        #         for pt in pts[sel2]:
        #             f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))
            
        #     with open(f'pc/pc{i}_3.txt', 'w') as f:
        #         for pt in pts[sel3]:
        #             f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))
        
        # input()
        # continue
        n0, n1 = info['num_0'], info['num_1']

        train_x = torch.from_numpy(train_x).permute([0, 4, 1, 2, 3]).cuda()
        train_y = torch.from_numpy(train_y).permute([0, 4, 1, 2, 3]).cuda()[:, 0]
        mask = torch.from_numpy(info['mask']).cuda()

        pred_y = geo_generator(train_x)[:, 0]

        loss = loss_fn(pred_y[mask], train_y[mask])

        # iou = compute_iou(pred_y[mask], train_y[mask])

        weight = train_y[mask]/n1/2 + (1-train_y[mask])/n0/2

        loss = torch.sum(weight * loss)# / torch.sum(weight)
        # loss = torch.mean(loss)

        print(f'Iter {it}', loss.item())

        loss.backward()
        optimizer.step()

        if it % 1000 == 0:

            # d = {
            #     'epoch': 
            #     'model_state_dict': geo_generator.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }

            # torch.save(d, f'ckpt/ckpt_{it}.pt')
            pass


def test_model(model, data_loader, loss_fn, ckpt_path):

    ptx, pty, ptz = np.meshgrid(np.arange(128), np.arange(64), np.arange(128), indexing='ij')
    pts = np.stack([ptx, pty, ptz], axis=-1)#.reshape((-1, 3))
    pts = np.round(pts).astype(np.int32)

    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()

    with torch.no_grad():

        while data_loader.has_val():
            val_x, val_y, info = data_loader.get_val_data()

            sel1 = (val_x[..., 0] > 0.5) # visible occupied # + unvisible all
            sel2 = (val_x[..., 1] > 0.5) # unvisible all
            sel3 = val_y[..., 0].astype(np.bool) # visible occupied + unvisible occupied

            n0, n1 = info['num_0'], info['num_1']

            val_x = torch.from_numpy(val_x).permute([0, 4, 1, 2, 3]).cuda()
            val_y = torch.from_numpy(val_y).permute([0, 4, 1, 2, 3]).cuda()[:, 0]

            mask = torch.from_numpy(info['mask']).cuda()

            pred_y = geo_generator(val_x)[:, 0]

            loss = loss_fn(pred_y[mask], val_y[mask])

            pred_y_01 = torch.nn.Sigmoid()(pred_y) > 0.8

            iou, p, r = compute_iou_pr(pred_y_01[mask], val_y[mask])

            weight = val_y[mask]/n1/2 + (1-val_y[mask])/n0/2

            loss = torch.sum(weight * loss)# / torch.sum(weight)
            # loss = torch.mean(loss)

            print(f'Loss {loss.item():6f}')
            print(f'IoU {iou:6f}')
            print(f'Pre {p:6f}')
            print(f'Rec {r:6f}')

            # for i in range(5):
            #     with open(f'pc/pc{i}_1.txt', 'w') as f:
            #         for pt in pts[sel1[i]]:
            #             f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))
                
            #     with open(f'pc/pc{i}_2.txt', 'w') as f:
            #         for pt in pts[(sel1[i] & ~sel2[i]) | (sel2[i] & pred_y_01[i].cpu().numpy().astype(np.bool))]:
            #             f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))
                
            #     with open(f'pc/pc{i}_3.txt', 'w') as f:
            #         for pt in pts[sel3[i]]:
            #             f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))

            for i in range(5):
                name = data_loader.current_fids[i]
                with open(f'val_pred/{name}.txt', 'w') as f:
                    for pt in pts[(sel1[i] & ~sel2[i]) | (sel2[i] & pred_y_01[i].cpu().numpy().astype(np.bool))]:
                        f.write('%.1lf;%.1lf;%.1lf\n' % (pt[0]+0.5, pt[1]+0.5, pt[2]+0.5))














if __name__ == '__main__':

    data_loader = GeoDataLoader('/home/lzq/lzy/NSVF/ReplicaSingleAll10cmGlobal/all/npz/*.npz', 5, [128,64,128])

    geo_generator = GeoGenerator(nf_in_geo=2, nf=64, max_data_size=[128,64,128]).cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    optimizer = torch.optim.Adam(geo_generator.parameters(), lr=0.0001)

    test_model(geo_generator, data_loader, loss_fn, 'ckpt/ckpt_14000.pt')
    

