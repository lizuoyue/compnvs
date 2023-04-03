import numpy as np
import open3d as o3d

if __name__ == '__main__':

    scenes = ["apartment_0", "apartment_1", "apartment_2",
            "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
            "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
            "hotel_0", "office_0", "office_1", "office_2",
            "office_3", "office_4", "room_0", "room_1", "room_2"]

    d = {}
    for scene in scenes:
        pc = o3d.io.read_point_cloud(f'../Replica-Dataset/replica_v1/{scene}/mesh.ply')
        pc = np.asarray(pc.points)
        print(scene, pc.shape[0], 'points')
        d[scene] = np.concatenate([
            np.floor(pc.min(axis=0) * 10) / 10,
            np.ceil(pc.max(axis=0) * 10) / 10,
        ])
        print(d[scene])
        print()

    names = "apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_0,apartment_1,apartment_1,apartment_2,apartment_2,apartment_2,apartment_2,frl_apartment_0,frl_apartment_0,frl_apartment_0,frl_apartment_1,frl_apartment_1,frl_apartment_1,frl_apartment_2,frl_apartment_2,frl_apartment_2,frl_apartment_3,frl_apartment_3,frl_apartment_3,frl_apartment_4,frl_apartment_4,frl_apartment_4,frl_apartment_5,frl_apartment_5,frl_apartment_5,hotel_0,hotel_0,hotel_0,office_0,office_1,office_2,office_3,office_4,room_0,room_1,room_2".split(',')
    
    with open('ReplicaRawData/bboxes.txt', 'w') as f:
        for name in names:
            bbox = d[name][[0,2,4,3,5,1]]
            bbox[[2, 5]] *= -1
            f.write('%.1f %.1f %.1f %.1f %.1f %.1f\n' % tuple(bbox.tolist()))
    