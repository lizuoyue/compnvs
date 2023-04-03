import os
import tqdm

src_rgb_dir = '/home/lzq/lzy/replica_habitat_gen/ReplicaGenFt/all/rgb'

depth_dir = '/home/lzq/lzy/replica_habitat_gen/ReplicaGenFt/all/depth'  # {sid:02d}_{vid:03d}.npz
# load ['depth']

pytorch3d_rgb_dir = '/work/lzq/matterport3d/replica_pytorch3d_rgb'  # {sid:02d}_{vid:03d}.jpg

for sid in range(48):
    if sid in [13, 14, 19, 20, 21, 42]:
        phase = 'val'
    else:
        phase = 'train'
    dst_rgb_dir = f'gen_data/{phase}/rgb/'
    os.makedirs(dst_rgb_dir, exist_ok=True)
    for vid in tqdm.tqdm(range(300)):
        src_img = os.path.join(src_rgb_dir, f'{sid:02d}_{vid:03d}.png')
        os.system(f'ln -s {src_img} {dst_rgb_dir}')