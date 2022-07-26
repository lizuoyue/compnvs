import os

for i in range(0, 45):
    os.system(f'mv scene{i:02d}/init_voxel/fine_points_0.1.txt scene{i:02d}/init_voxel/fine_points_0.1_aligned.txt')
    continue
    for e, j in enumerate(range(i * 250, (i + 1) * 250)):
        os.system(f'mv scene{i:02d}/rgb/0_{j:04d}.png scene{i:02d}/rgb/0_{e:04d}.png')
        os.system(f'mv scene{i:02d}/depth/0_{j:04d}.png scene{i:02d}/depth/0_{e:04d}.png')
        os.system(f'mv scene{i:02d}/pose/0_{j:04d}.txt scene{i:02d}/pose/0_{e:04d}.txt')
