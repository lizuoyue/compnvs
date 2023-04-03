import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

depth_dir = '/home/lzq/lzy/NSVF/ARKitScenesSingleViewNSVF/all/depth'
sltd_views = ['42446156_79162312_79161313_79164311', '42446156_79162312_79160313_79163312']

for sltd_view in sltd_views:
    scene, k, i,j = sltd_view.split('_')
    for view in [k,i,j]:
        view_name = f'{scene}_{view}'
        depth = np.load(f'{depth_dir}/{view_name}.npz')['depth']
        plt.imshow(depth)
        plt.axis('off')
        plt.savefig(f'res_arkit_depth/{view_name}.jpg')