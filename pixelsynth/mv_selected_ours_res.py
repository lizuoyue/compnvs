import os
import tqdm
from PIL import Image

# Replica
# sltd_file_dir = '/home/lzq/lzy/NSVF/selected_replica_views.txt'
# with open(sltd_file_dir, 'r') as f:
#     sltd_view_names = [line.strip() for line in f.readlines()]
sltd_view_names = ['21_129_018_093', '14_170_076_206']

val_scene_ids = [13, 14, 19, 20, 21, 42]
scene_views_dict = {}
for scene_id in val_scene_ids:
    scene_views = sorted([img[:-7] for img in os.listdir(f'gen_data/{scene_id:02d}') if '_in' in img])
    scene_views_dict[f'{scene_id:02d}'] = scene_views

src_dir = '/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base'
# rgb_dst_dir = 'res_multiviews_replica_ours_sltd/rgb'
# gif_dst_dir = 'res_multiviews_replica_ours_sltd/gif'
rgb_dst_dir = 'res_multiviews_replica_ours_failure/rgb'
gif_dst_dir = 'res_multiviews_replica_ours_failure/gif'
os.makedirs(gif_dst_dir, exist_ok=True)
for view_name in tqdm.tqdm(sltd_view_names):
    view_rgb_dst_dir = f'{rgb_dst_dir}/{view_name}'
    os.makedirs(view_rgb_dst_dir, exist_ok=True)

    scene_id = view_name[:2]
    view_idx = scene_views_dict[scene_id].index(view_name[3:])
    view_img_indices = list(range(view_idx*15, view_idx*15+15))
    # New
    view_imgs = [Image.open(f'{src_dir}/output{view_name[:2]}/color/{img_idx:05d}.png').convert('RGB') for img_idx in view_img_indices]
    for idx, img in enumerate(view_imgs):
        img.save(os.path.join(view_rgb_dst_dir, f'{idx:02d}.jpg'))   # save images
    # imgs_for_gif = view_imgs[7:15] + view_imgs[14:-1:-1] + view_imgs[0:7]
    gif_img_indices = list(range(7,15)) + list(range(14,-1,-1)) + list(range(0,7))
    imgs_for_gif = [view_imgs[idx] for idx in gif_img_indices]
    imgs_for_gif[0].save(os.path.join(gif_dst_dir, f'{view_name}.gif'),     # save gif
                            append_images=imgs_for_gif[1:], duration=200, loop=0, save_all=True)

# ARKit
# sltd_file_dir = '/home/lzq/lzy/NSVF/selected_arkit_views.txt'
# with open(sltd_file_dir, 'r') as f:
#     lines = [line.strip() for line in f.readlines()]
lines = ['41069021_332566_328568_333566 0', '42898497_749910077_749909078_749911077 0']

all_val_views = sorted([file[:-4] for file in os.listdir('/work/lzq/matterport3d/arkit_new_val_video_poses')])

src_dir = '/home/lzq/lzy/NSVF/ARKitScenesTripletsNSVF/all/mink_nsvf_rgba_sep_field_base/output_video'
# rgb_dst_dir = 'res_multiviews_arkit_ours_sltd/rgb'
# gif_dst_dir = 'res_multiviews_arkit_ours_sltd/gif'
rgb_dst_dir = 'res_multiviews_arkit_ours_failure/rgb'
gif_dst_dir = 'res_multiviews_arkit_ours_failure/gif'
os.makedirs(gif_dst_dir, exist_ok=True)
for line in tqdm.tqdm(lines):
    view_name, angle = line.split(' ')
    view_rgb_dst_dir = f'{rgb_dst_dir}/{view_name}'
    os.makedirs(view_rgb_dst_dir, exist_ok=True)

    view_idx = all_val_views.index(view_name)
    view_img_indices = list(range(view_idx*15, view_idx*15+15))
    # New
    view_imgs = [Image.open(f'{src_dir}/color/{img_idx:05d}.png').convert('RGB').rotate(int(angle), expand=1) for img_idx in view_img_indices]
    for idx, img in enumerate(view_imgs):
        img.save(os.path.join(view_rgb_dst_dir, f'{idx:02d}.jpg'))   # save images
    # imgs_for_gif = view_imgs[7:15] + view_imgs[14:-1:-1] + view_imgs[0:7]
    gif_img_indices = list(range(7,15)) + list(range(14,-1,-1)) + list(range(0,7))
    imgs_for_gif = [view_imgs[idx] for idx in gif_img_indices]
    imgs_for_gif[0].save(os.path.join(gif_dst_dir, f'{view_name}.gif'),     # save gif
                            append_images=imgs_for_gif[1:], duration=200, loop=0, save_all=True)

    
