import os
import tqdm
from PIL import Image

# Replica
sltd_file_dir = '/home/lzq/lzy/NSVF/selected_replica_views.txt'
with open(sltd_file_dir, 'r') as f:
    sltd_view_names = [line.strip() for line in f.readlines()]

ablation_name = 'norgba'  # choices = ['no2d', 'norgba', 'basic']

if ablation_name == 'no2d':
    src_dir = f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output_no2d_single_depth/color'
else:
    src_dir = f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_base/output_{ablation_name}_single/color'
rgb_dst_dir = f'res_multiviews_replica_ours_{ablation_name}_sltd'
os.makedirs(rgb_dst_dir, exist_ok=True)

all_val_views = sorted(os.listdir('res_multiviews_replica'))
for view_name in tqdm.tqdm(sltd_view_names):
    view_idx = all_val_views.index(view_name)

    # cmd = f'cp {src_dir}/{view_idx:05d}.png {rgb_dst_dir}/{view_name}.png'
    # os.system(cmd)
    img = Image.open(f'{src_dir}/{view_idx:05d}.png').convert('RGB')
    img.save(f'{rgb_dst_dir}/{view_name}.jpg')
