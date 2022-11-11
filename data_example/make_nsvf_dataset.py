import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import glob
import json


def simple_downsample(pts, rgb, voxel_size):
    d = {}
    pts_int = np.round(pts / voxel_size).astype(np.int32)
    for (x,y,z), c in zip(pts_int, rgb):
        if (x,y,z) in d:
            d[(x,y,z)].append(c)
        else:
            d[(x,y,z)] = [c]
    vp, vc = [], []
    for (x,y,z), c_li in d.items():
        vp.append(np.array([x,y,z]))
        vc.append(np.round(np.stack(c_li).astype(np.float32).mean(axis=0)).astype(np.uint8))
    return np.stack(vp)*voxel_size, np.stack(vc)


class ExampleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(f'{root_dir}/transforms_train_synth.json') as f:
            self.d_train = json.load(f)
        with open(f'{root_dir}/transforms_test_synth.json') as f:
            self.d_test = json.load(f)
        
        for key in ['fl_x', 'fl_y', 'cx', 'cy']:
            assert(self.d_train[key] == self.d_test[key])
        
        d = self.d_train
        u, v = np.meshgrid(np.arange(d["w"]) + 0.5, np.arange(d["h"]) + 0.5)
        self.uv1 = np.stack([u, v, np.ones((d["h"], d["w"]))])
        self.intrinsics = np.array([[d["fl_x"], 0.0, d["cx"]], [0.0, d["fl_y"], d["cy"]], [0.0, 0.0, 1.0]])
    
    def __len__(self):
        return len(self.d_train['frames'])

    def __getitem__(self, idx):

        frame = self.d_train['frames'][idx]
        img = np.array(Image.open(f'{self.root_dir}/{frame["file_path"]}'))

        dep_path = frame["file_path"].replace('images', 'depth').replace('.jpg', '.png')
        dep = np.array(Image.open(f'{self.root_dir}/{dep_path}')) / 1000.0

        mask_path = frame["file_path"].replace('images', 'masks').replace('.jpg', '.npy')
        mask = ~np.load(f'{self.root_dir}/{mask_path}')
        mask &= (0.1 < dep) & (dep < 6.0)

        c2w = np.array(frame["transform_matrix"])
        c2w = c2w.dot(np.diag([1,-1,-1,1.0]))

        coord_pix = np.reshape(self.uv1 * np.stack([dep] * 3), (3, -1))
        coord_cam = np.linalg.inv(self.intrinsics).dot(coord_pix)
        coord_wld = c2w[:3, :3].dot(coord_cam).T + c2w[:3, -1]

        xyzs = coord_wld[mask.flatten()]
        rgbs = img[mask]

        return {
            'image': img,
            'depth': dep,
            'mask': mask,
            'pose': c2w,
            'xyzs': xyzs,
            'rgbs': rgbs,
        }

    def create_dataset_folder(self, save_dir):
        self.save_dir = save_dir
        os.system(f'mkdir -p {self.save_dir}')
        os.system(f'mkdir -p {self.save_dir}/npz')
        os.system(f'mkdir -p {self.save_dir}/npz_fts')
        os.system(f'mkdir -p {self.save_dir}/npz_fts_predgeo')
        os.system(f'mkdir -p {self.save_dir}/rgb')
        os.system(f'mkdir -p {self.save_dir}/pose')

        with open(f'{self.save_dir}/intrinsics.txt', 'w') as f:
            f.write(f'{self.d_train["fl_x"]*2.0} 0.0 {self.d_train["cx"]*2.0} 0.0\n')
            f.write(f'0.0 {self.d_train["fl_y"]*2.0} {self.d_train["cy"]*2.0} 0.0\n')
            f.write('0.0 0.0 1.0 0.0\n')
            f.write('0.0 0.0 0.0 1.0\n')

        os.system(f'mkdir -p {self.save_dir}/pose_video')
        test_poses = [
            np.array(frame['transform_matrix']).dot(np.diag([1,-1,-1,1.0])) for frame in self.d_test['frames']
        ]
        test_poses = np.stack(test_poses)
        np.savez_compressed(f'{self.save_dir}/pose_video/00000.npz', poses=test_poses)

        for i in range(test_poses.shape[0]):
            np.savetxt(f'{self.save_dir}/pose/{i:05d}.txt', np.eye(4), fmt='%.1f')
            Image.fromarray(np.zeros((192, 256, 3), np.uint8)).save(f'{self.save_dir}/rgb/{i:05d}.jpg')
            if i > 0:
                # np.savez_compressed(f'{self.save_dir}/npz_fts/{i:05d}.npz', a=0)
                np.savez_compressed(f'{self.save_dir}/npz_fts_predgeo/{i:05d}.npz', a=0)

    def save_single_data(self, data, idx):
        Image.fromarray(data['image']).save(f'{self.save_dir}/rgb/{idx:03d}.jpg')
        np.savez_compressed(f'{self.save_dir}/depth/{idx:03d}.npz', depth=data['depth'])
        Image.fromarray(data['mask'].astype(np.uint8) * 255).save(f'{self.save_dir}/mask/{idx:03d}.png')
        np.savetxt(f'{self.save_dir}/pose/{idx:03d}.txt', data['pose'], fmt='%.9lf')
        np.savez_compressed(f'{self.save_dir}/npz/{idx:03d}.npz', rgb=data['rgbs'], pts=data['xyzs'])


if __name__ == '__main__':

    for sid in range(1, 22):
        if sid == 5:
            continue
        
        dataset = ExampleDataset(f'data_for_zuoyue/{sid:03d}')
        dataset.create_dataset_folder(f'ExampleScenesFused/{sid:03d}')

        all_xyzs, all_rgbs = [], []
        idx = 0
        for data in tqdm.tqdm(dataset):
            # dataset.save_single_data(data, idx)
            idx += 1

            all_xyzs.append(data['xyzs'])
            all_rgbs.append(data['rgbs'])
        all_xyzs = np.vstack(all_xyzs)
        all_rgbs = np.vstack(all_rgbs)

        np.savez_compressed(f'SilvanScenesFused/{sid:03d}/npz/00000.npz', rgb=all_rgbs, pts=all_xyzs)
        
        # print(all_xyzs.shape)
        # print(simple_downsample(all_xyzs, all_rgbs, 0.01)[0].shape)
        
        # with open(f'vis/pts_{sid:03d}.txt', 'w') as f:
        #     for (x, y, z), (r, g, b) in zip(*simple_downsample(all_xyzs, all_rgbs, 0.01)):
        #         f.write('%.2lf %.2lf %.2lf %d %d %d\n' % (x, y, z, r, g, b))

