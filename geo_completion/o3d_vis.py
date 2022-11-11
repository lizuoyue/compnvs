import open3d as o3d
import numpy as np

def create_o3d_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

if __name__ == '__main__':

    npz = np.load('/home/lzq/lzy/NSVF/ReplicaGenFt/all/npz/00_000.npz')
    pose = np.loadtxt('/home/lzq/lzy/NSVF/ReplicaGenFt/all/pose/00_000.txt')
    print(pose)

    pc = create_o3d_point_cloud(npz['pts'], npz['rgb'])
    pc.translate([0, 0, 0])
    pc.estimate_normals()
    pc.rotate(np.eye(3))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pc)
    vis.update_geometry(pc)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('test.png')
    vis.destroy_window()
    # o3d.visualization.draw_geometries([pc])



