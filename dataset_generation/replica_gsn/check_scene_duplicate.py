import numpy as np
import tqdm

if __name__ == '__main__':

    di, li = {}, []
    for sid in tqdm.tqdm(list(range(101))):
        pts = np.loadtxt(f'ReplicaGSN/scene{sid:03d}/init_voxel/center_points_0.1.txt')
        a, b, c = np.round(pts.min(axis=0) * 100).astype(np.int).flatten().tolist()
        d, e, f = np.round(pts.max(axis=0) * 100).astype(np.int).flatten().tolist()
        li.append((a, b, c, d, e, f, sid))
        key = (int(a), int(b), int(c), int(d), int(e), int(f))
        if key in di:
            di[key].append(sid)
        else:
            di[key] = [sid]
    
    li.sort()
    for item in li:
        print(item)
    
    print()
    
    for key in sorted(di.keys()):
        print(key, di[key])
