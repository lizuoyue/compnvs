import os
import tqdm
from PIL import Image

# Replica
# sltd_file_dir = '/home/lzq/lzy/NSVF/selected_replica_views.txt'
# with open(sltd_file_dir, 'r') as f:
#     sltd_view_names = [line.strip() for line in f.readlines()]
sltd_3queries = ['13_090_118_268_009_028_032', '13_182_227_001_006_186_027', '19_006_114_198_041_222_004']

val_scene_ids = [13, 14, 19, 20, 21, 42]
scene_views_dict = {}
for scene_id in val_scene_ids:
    scene_views = sorted([img[:-7] for img in os.listdir(f'gen_data/{scene_id:02d}') if '_in' in img])
    scene_views_dict[f'{scene_id:02d}'] = scene_views

src_dir = '/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base'
rgb_dst_dir = 'res_3query_replica_ours_sltd/rgb'
gif_dst_dir = 'res_3query_replica_ours_sltd/gif'
os.makedirs(gif_dst_dir, exist_ok=True)
for sltd_3query in tqdm.tqdm(sltd_3queries):
    video_rgb_dst_dir = f'{rgb_dst_dir}/{sltd_3query}'
    os.makedirs(video_rgb_dst_dir, exist_ok=True)

    s, n1, q1, n2, q2, n3, q3 = sltd_3query.split('_')
    refs = ['_'.join([q1, n1, n2]), '_'.join([q2, n2, n3]), '_'.join([q3, n3, n1])]
    view_names = ['_'.join([q1,]+sorted([n1, n2])), '_'.join([q2,]+sorted([n2, n3])), '_'.join([q3,]+sorted([n3, n1]))]
    orders = []
    for (ref, view_name) in zip(refs, view_names):
        if ref != view_name:
            orders.append(-1)
        else:
            orders.append(1)
    print(view_names)
    
    sltd_3query_imgs = []
    for i, view_name in enumerate(view_names):
        view_idx = scene_views_dict[s].index(view_name)
        view_img_indices = list(range(view_idx*15, view_idx*15+15))
        if orders[i] == -1:
            view_img_indices = view_img_indices[::-1]
        # New
        view_imgs = [Image.open(f'{src_dir}/output{s}/color/{img_idx:05d}.png').convert('RGB') for img_idx in view_img_indices]
        sltd_3query_imgs += view_imgs[:14]  # remove the replicated last frame
        
    for idx, img in enumerate(sltd_3query_imgs):
        img.save(os.path.join(video_rgb_dst_dir, f'{idx:02d}.jpg'))   # save images
    imgs_for_gif = sltd_3query_imgs
    imgs_for_gif[0].save(os.path.join(gif_dst_dir, f'{sltd_3query}.gif'),     # save gif
                            append_images=imgs_for_gif[1:], duration=200, loop=0, save_all=True)

# # ARKit
# # sltd_file_dir = '/home/lzq/lzy/NSVF/selected_arkit_views.txt'
# # with open(sltd_file_dir, 'r') as f:
# #     lines = [line.strip() for line in f.readlines()]
# lines = ['41069021_332566_328568_333566 0', '42898497_749910077_749909078_749911077 0']

# all_val_views = sorted([file[:-4] for file in os.listdir('/work/lzq/matterport3d/arkit_new_val_video_poses')])

# src_dir = '/home/lzq/lzy/NSVF/ARKitScenesTripletsNSVF/all/mink_nsvf_rgba_sep_field_base/output_video'
# # rgb_dst_dir = 'res_multiviews_arkit_ours_sltd/rgb'
# # gif_dst_dir = 'res_multiviews_arkit_ours_sltd/gif'
# rgb_dst_dir = 'res_multiviews_arkit_ours_failure/rgb'
# gif_dst_dir = 'res_multiviews_arkit_ours_failure/gif'
# os.makedirs(gif_dst_dir, exist_ok=True)
# for line in tqdm.tqdm(lines):
#     view_name, angle = line.split(' ')
#     view_rgb_dst_dir = f'{rgb_dst_dir}/{view_name}'
#     os.makedirs(view_rgb_dst_dir, exist_ok=True)

#     view_idx = all_val_views.index(view_name)
#     view_img_indices = list(range(view_idx*15, view_idx*15+15))
#     # New
#     view_imgs = [Image.open(f'{src_dir}/color/{img_idx:05d}.png').convert('RGB').rotate(int(angle), expand=1) for img_idx in view_img_indices]
#     for idx, img in enumerate(view_imgs):
#         img.save(os.path.join(view_rgb_dst_dir, f'{idx:02d}.jpg'))   # save images
#     # imgs_for_gif = view_imgs[7:15] + view_imgs[14:-1:-1] + view_imgs[0:7]
#     gif_img_indices = list(range(7,15)) + list(range(14,-1,-1)) + list(range(0,7))
#     imgs_for_gif = [view_imgs[idx] for idx in gif_img_indices]
#     imgs_for_gif[0].save(os.path.join(gif_dst_dir, f'{view_name}.gif'),     # save gif
#                             append_images=imgs_for_gif[1:], duration=200, loop=0, save_all=True)

    
