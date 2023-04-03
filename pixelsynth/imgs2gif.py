import os
from PIL import Image
from glob import glob
from tqdm import tqdm

# multiviews_save_dirs = glob('/home/lzq/lzy/PixelSynth/res_multiviews/*')
# num_views = 15
# for multiviews_save_dir in tqdm(multiviews_save_dirs):
#     ims = [Image.open(os.path.join(multiviews_save_dir, f'{view_idx:02d}.png')) for view_idx in range(num_views)]
#     ims[0].save(os.path.join(multiviews_save_dir, f'gif.gif'), 
#                 append_images=ims[1:], duration=200, loop=0, save_all=True)

src_rgb_dir = '/home/lzq/lzy/NSVF/ARKitScenesTripletsNSVF/all/mink_nsvf_rgba_sep_field_base/output_video/color'
dst_gif_dir = '/home/lzq/lzy/NSVF/ARKitScenesTripletsNSVF/all/mink_nsvf_rgba_sep_field_base/output_video/gif'
os.makedirs(dst_gif_dir, exist_ok=True)

all_val_view_names = sorted([file[:-4] for file in os.listdir('/work/lzq/matterport3d/arkit_new_val_video_poses')])
for view_idx, view_name in tqdm(enumerate(all_val_view_names)):
    img_indices = list(range(view_idx*15, view_idx*15+15))
    imgs = [Image.open(os.path.join(src_rgb_dir, f'{img_idx:05d}.png')) for img_idx in img_indices]
    imgs[0].save(os.path.join(dst_gif_dir, f'{view_name}.gif'), 
                append_images=imgs[1:], duration=200, loop=0, save_all=True)