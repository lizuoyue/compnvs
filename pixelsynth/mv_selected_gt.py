import os
import tqdm
from PIL import Image

# Replica
sltd_file_dir = '/home/lzq/lzy/NSVF/selected_replica_views.txt'
with open(sltd_file_dir, 'r') as f:
    sltd_view_names = [line.strip() for line in f.readlines()]

src_gt_dir = '/home/lzq/lzy/replica_habitat_gen/replica_val_video_gt/images/train'
rgb_dst_dir = 'res_multiviews_replica_gt_sltd/rgb'
gif_dst_dir = 'res_multiviews_replica_gt_sltd/gif'
os.makedirs(gif_dst_dir, exist_ok=True)

all_val_views = sorted(os.listdir('res_multiviews_replica'))
for view_name in tqdm.tqdm(sltd_view_names):
    view_idx = all_val_views.index(view_name)
    img_indices = list(range(view_idx*15, view_idx*15+15))

    view_rgb_dst_dir = os.path.join(rgb_dst_dir, view_name)
    os.makedirs(view_rgb_dst_dir, exist_ok=True)

    # New
    view_imgs = [Image.open(f'{src_gt_dir}/{img_idx:05d}.png').convert('RGB') for img_idx in img_indices]
    for idx, img in enumerate(view_imgs):
        img.save(os.path.join(view_rgb_dst_dir, f'{idx:02d}.jpg'))   # save images
    gif_img_indices = list(range(7,15)) + list(range(14,-1,-1)) + list(range(0,7))
    imgs_for_gif = [view_imgs[idx] for idx in gif_img_indices]
    imgs_for_gif[0].save(os.path.join(gif_dst_dir, f'{view_name}.gif'),     # save gif
                            append_images=imgs_for_gif[1:], duration=200, loop=0, save_all=True)

# ARKit
sltd_file_dir = '/home/lzq/lzy/NSVF/selected_arkit_views.txt'
with open(sltd_file_dir, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

src_gt_dir = '/home/lzq/lzy/arkit_video_gt'
rgb_dst_dir = 'res_multiviews_arkit_gt_sltd/rgb'
gif_dst_dir = 'res_multiviews_arkit_gt_sltd/gif'
os.makedirs(gif_dst_dir, exist_ok=True)

all_val_views = sorted([file[:-4] for file in os.listdir('/work/lzq/matterport3d/arkit_new_val_video_poses')])
for line in tqdm.tqdm(lines):
    view_name, angle = line.split(' ')
    view_rgb_dst_dir = os.path.join(rgb_dst_dir, view_name)
    os.makedirs(view_rgb_dst_dir, exist_ok=True)

    # New
    view_imgs = [Image.open(f'{src_gt_dir}/{view_name}/{img_idx:02d}.png').convert('RGB').rotate(int(angle), expand=1) for img_idx in range(15)]
    for idx, img in enumerate(view_imgs):
        img.save(os.path.join(view_rgb_dst_dir, f'{idx:02d}.jpg'))   # save images
    gif_img_indices = list(range(7,15)) + list(range(14,-1,-1)) + list(range(0,7))
    imgs_for_gif = [view_imgs[idx] for idx in gif_img_indices]
    imgs_for_gif[0].save(os.path.join(gif_dst_dir, f'{view_name}.gif'),     # save gif
                            append_images=imgs_for_gif[1:], duration=200, loop=0, save_all=True)