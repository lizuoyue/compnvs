import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.vqvae2.dataset import ImageFileDataset, CodeRow
from models.vqvae2.vqvae import VQVAETop

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["DEBUG"] = "False"

def extract(loader, model, device, dataset, split):
    index = 0
    embeddings = []
    pbar = tqdm(loader)

    if dataset == 'replica':
        if split == 'val':
            scene_ids = [13, 14, 19, 20, 21, 42]
        elif split == 'train':
            scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]
        scene_embeddings = [np.zeros([300, 32, 32]) for sid in scene_ids]

        num_embeddings = set()
        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, _ = model.module.encode(img)
            id_t = id_t.detach().cpu().numpy()

            for file, top in zip(filename, id_t):
                index += 1
                pbar.set_description(f'inserted: {index}')
                view_name = file.split('/')[-1][:-4]
                num_embeddings.add(view_name)          
                embeddings.append(top)

                sid, vid = [int(num) for num in view_name.split('_')]
                scene_embeddings[scene_ids.index(sid)][vid] = top
        embeddings = np.stack(embeddings, axis=0)

    elif dataset == 'arkit':
        data_dir = '/home/lzq/lzy/ARKitScenes/Selected'
        split_file = f'{data_dir}/{split}_split.txt'
        with open(split_file, 'r') as txt_file:
            scene_names = [line.strip() for line in txt_file.readlines()]
        scene_embeddings = {scene_name: {} for scene_name in scene_names}

        num_embeddings = set()
        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, _ = model.module.encode(img)
            id_t = id_t.detach().cpu().numpy()

            for file, top in zip(filename, id_t):
                index += 1
                pbar.set_description(f'inserted: {index}')
                view_name = file.split('/')[-1][:-4]
                num_embeddings.add(view_name)          
                embeddings.append(top)

                scene_name, empty_view_name = view_name.split('_')
                scene_embeddings[scene_name][empty_view_name] = top
        embeddings = np.stack(embeddings, axis=0)
    
    elif dataset in ['fountain', 'westminster', 'notre', 'sacre', 'pantheon']:
        num_embeddings = set()
        scene_embeddings = {}
        for img, _, filename in pbar:
            img = img.to(device)
            _, _, _, id_t, _ = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            for file, top in zip(filename, id_t):
                index += 1
                pbar.set_description(f'inserted: {index}')
                embeddings.append(top)
                scene_embeddings[file] = top
        embeddings = np.stack(embeddings, axis=0)
    else:
        pass

    print('num embeddings',len(num_embeddings))

    try:
        os.makedirs('models/vqvae2/embeddings/', exist_ok=True)
        os.makedirs('models/vqvae2/scene_embeddings/', exist_ok=True)
    except:
        pass
    np.save('models/vqvae2/embeddings/%s_%s.npy' % (dataset, split), embeddings)
    np.savez_compressed('models/vqvae2/scene_embeddings/%s_%s' % (dataset, split), embeddings=scene_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, nargs='+', default=256)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=str)

    args = parser.parse_args()

    device = 'cuda'

    size = args.size if len(args.size) == 1 else tuple(args.size)
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = VQVAETop()

    torch_devices = [0,1]
    device = "cuda:" + str(torch_devices[0])
    
    from torch import nn
    # model = nn.DataParallel(model, torch_devices).to(device)
    moswl = model.to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    extract(loader, model, device, args.dataset, args.split)