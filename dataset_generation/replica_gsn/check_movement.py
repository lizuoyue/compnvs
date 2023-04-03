import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

def sim(arr1, arr2):
    return np.abs(arr1[:3,:3] - arr2[:3,:3]).mean()

def get_ranges(li):
    res, tpl = [], []
    for num in li:
        if len(tpl) == 0:
            tpl = [num, num + 1]
        else:
            if num == tpl[1]:
                tpl[1] = num + 1
            else:
                res.append(tpl)
                tpl = [num, num + 1]
    return res


if __name__=='__main__':

    # phase, n_seq = 'train', 101
    phase, n_seq = 'test', 11

    num = [0, 0, 0]

    for i in range(n_seq):
        with open(f'{phase}/{i:02d}/cameras.json') as f:
            li = json.load(f)
            to_check = []
            for j in range(len(li)-1):
                s = sim(np.array(li[j]['Rt']), np.array(li[j+1]['Rt']))
                if s < 1e-3:
                    pass
                else:
                    if len(to_check) > 0 and to_check[-1] == j:
                        to_check.append(j+1)
                    else:
                        to_check.append(j)
                        to_check.append(j+1)
                # r = R.from_matrix(np.array(li[j]['Rt'])[:3, :3])
                # qx, qy, qz, qw = r.as_quat()
                # assert(np.abs(qx) < 1e-6)
                # assert(np.abs(qz) < 1e-6)
                # print(np.arctan2(qy, qw) * 180 / np.pi)
            
            for a, b in get_ranges(to_check):
                num[0] += (b - a >= 12) * (b - a - 12 + 1) # full unobserve
                num[1] += (b - a >= 10) * (b - a - 10 + 1) # 2/3 unobserve
                num[2] += (b - a >= 8) * (b - a - 8 + 1)  # 1/3 unobserve
            
            continue
            
            gif = []
            for idx in to_check:
                gif.append(Image.open(f'{phase}/{i:02d}/{idx:03d}_rgb.png').resize((256, 256), resample=Image.LANCZOS))
            gif[0].save(f'{phase}_rot_gif/{i:02d}.gif', append_images=gif[1:], loop=0, duration=250, save_all=True)

    print(num, sum(num))
