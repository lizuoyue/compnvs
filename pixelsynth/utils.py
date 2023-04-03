import numpy as np
import matplotlib
matplotlib.use('agg')
from scipy.spatial.transform import Rotation as R

def pose2rotmat(pose):
    # Pose: B x 7
    b, n = pose.shape
    assert(n == 7)
    r = R.from_quat(pose[:,[4,5,6,3]])
    rotmat = r.as_matrix() # B x 3 x 3
    rotmat = np.concatenate([rotmat, pose[:,:3,np.newaxis]], axis=2) # B x 3 x 4
    to_cat = np.zeros((b, 1, 4))
    to_cat[:,:,-1] = 1
    rotmat = np.concatenate([rotmat, to_cat], axis=1) # B x 4 x 4
    # Replica coordinates
    neg_yz = np.diag([1.0,-1.0,-1.0,1.0]).astype(np.float32)
    return rotmat.astype(np.float32).dot(neg_yz)

def project(pts, int_mat, ext_mat, valid=(0, 256, 0, 256)):
    assert(len(pts.shape) == 2) # (N, 3)
    local = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=-1)
    local = local.dot(np.linalg.inv(ext_mat).T)[:, :3]
    local = local.dot(int_mat.T)
    local[:, :2] /= local[:, 2:]
    a, b, c, d = valid
    valid = np.ones((pts.shape[0],), np.bool)
    valid = valid & (local[:, 0] >= a)
    valid = valid & (local[:, 0] <= b)
    valid = valid & (local[:, 1] >= c)
    valid = valid & (local[:, 1] <= d)
    valid = valid & (local[:, 2] > 0)
    return valid

def project_depth(pts, int_mat, ext_mat):
    assert(len(pts.shape) == 2) # (N, 3)
    local = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=-1)
    local = local.dot(np.linalg.inv(ext_mat).T)[:, :3]
    local = local.dot(int_mat.T)
    local[:, :2] /= local[:, 2:]
    return local[:, :2], local[:, 2]

def unproject(dep, int_mat, ext_mat):
    # int_mat: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # ext_mat: local to world
    assert(len(dep.shape) == 2)
    h, w = dep.shape
    x, y = np.meshgrid(np.arange(w).astype(np.float32)+0.5, np.arange(h).astype(np.float32)+0.5)
    z = np.ones((h, w), np.float32)
    pts = np.stack([x, y, z], axis=-1)
    pts = pts.dot(np.linalg.inv(int_mat[:3, :3]).T)
    pts = pts * np.stack([dep] * 3, axis=-1) # local
    pts = np.concatenate([pts, np.ones((h, w, 1), np.float32)], axis=-1)
    pts = pts.dot(ext_mat.T)[..., :3]
    return pts

def voxelize_points(pts, info, voxel_size, ref=np.array([0, 0, 0])):
    # pts: shape of N, 3
    # info: shape of N, C
    # return voxel centers
    # ref: boundary reference
    half_voxel = voxel_size / 2.0
    offset = np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], np.int)

    center_idx = np.unique(np.floor((pts - ref) / voxel_size).astype(np.int), axis=0) * 2 + 1 # always odd

    vertex_idx = np.repeat(center_idx, 8, axis=0) + np.tile(offset, (center_idx.shape[0], 1)) # all voxel vertices index, always even
    vertex_idx, center_to_vertex = np.unique(vertex_idx, axis=0, return_inverse=True)
    center_to_vertex = center_to_vertex.reshape((-1, 8))

    center_pts = center_idx / 2.0 * voxel_size + ref # center coordinates
    vertex_pts = vertex_idx / 2.0 * voxel_size + ref # vertex coordinates

    num_c = info.shape[1]
    vertex_info = np.zeros((vertex_idx.shape[0], num_c + 1))

    vertex_idx_to_1d_idx = {}
    for d_idx, line in enumerate(vertex_idx):
        x, y, z = line
        vertex_idx_to_1d_idx[(x, y, z)] = d_idx

    pt_to_vertex = np.round((pts - ref) / voxel_size).astype(np.int) * 2 # always even
    for (x, y, z), pt_info in zip(pt_to_vertex, info):
        idx = vertex_idx_to_1d_idx[(x, y, z)]
        vertex_info[idx, :num_c] += pt_info
        vertex_info[idx, num_c] += 1

    return center_pts, center_to_vertex, (vertex_idx / 2.0).astype(np.int), vertex_info

def unproject_free_space(dep, int_mat, ext_mat, voxel_size):
    # int_mat: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # ext_mat: local to world
    dep = np.maximum(dep - voxel_size, 0)
    pts = []
    while dep.max() > 0:
        layer_pts = unproject(dep, int_mat, ext_mat)
        pts.append(layer_pts[dep > 0])
        dep = np.maximum(dep - voxel_size, 0)
    if len(pts) > 0:
        return np.concatenate(pts, axis=0)
    else:
        return None

def simple_voxelize_points(pts, voxel_size, ref=np.array([0, 0, 0]), unique=True):
    # ref: boundary reference
    # return center coord
    if pts is None:
        return None
    idx = np.floor((pts - ref) / voxel_size).astype(np.int)
    if unique:
        idx = np.unique(idx, axis=0)
    return (idx + 0.5) * voxel_size + ref


class OccupancyGrid(object):

    def __init__(self, spatial_size, voxel_size=None, reference=None, init_value=0, dtype=np.int32):
        assert(len(spatial_size) == 3)
        self.spatial_size = spatial_size
        self.voxel_size = voxel_size
        self.reference = reference # center reference, (0, 0, 0)'s coord
        self.e = np.ones(4, np.int32)
        self.e[1:] = np.cumprod(spatial_size[::-1])
        self.grid = np.ones((self.e[-1]), dtype) * init_value
        self.offset = np.array([
            [-1,-1,-1],
            [-1,-1, 1],
            [-1, 1,-1],
            [-1, 1, 1],
            [ 1,-1,-1],
            [ 1,-1, 1],
            [ 1, 1,-1],
            [ 1, 1, 1],
        ], np.int)
        return
    
    def get_grid(self):
        return self.grid.reshape(self.spatial_size)
    
    def get_occupied_index(self, condition=None):
        grid = self.get_grid()
        if condition is None:
            return np.stack(np.nonzero(grid != 0), axis=-1)
        else:
            return np.stack(np.nonzero(condition(grid)), axis=-1)

    def get_occupied_coord(self, condition=None):
        return self.get_occupied_index(condition) * self.voxel_size + self.reference

    def get_all_grid_coord(self):
        return self.get_occupied_coord(condition=lambda x: x < np.inf)
    
    def get_all_vertex_coord(self):
        grid_coord = self.get_all_grid_coord()
        return np.repeat(grid_coord, 8, axis=0) + np.tile(self.offset, (grid_coord.shape[0], 1)) * self.voxel_size / 2
    
    def get_occupancy_status_by_coord(self, coord):
        if coord is None or len(coord.shape) == 0:
            return
        assert(self.reference is not None) # coord: N x 3
        raw_index = (coord - self.reference) / self.voxel_size
        index = np.round(raw_index).astype(np.int32)
        assert(np.abs(raw_index - index).mean() < 1e-3)
        return self.get_occupancy_status_by_index(index)

    def get_occupancy_status_by_index(self, index):
        index = self.filter_invalid_index(index)
        one_dim_index = index.dot(self.e[:3][::-1])
        return self.grid[one_dim_index]

    def filter_invalid_index(self, index):
        mask = np.ones((index.shape[0]), np.bool)
        for dim in range(3):
            mask &= (index[:, dim] >= 0)
            mask &= (index[:, dim] < self.spatial_size[dim])
        assert(mask.mean() > 0.999999)
        return index[mask]

    def set_occupancy_by_index(self, index, value=1):
        index = self.filter_invalid_index(index)
        one_dim_index = index.dot(self.e[:3][::-1])
        self.grid[one_dim_index] = value
        return

    def set_occupancy_by_coord(self, coord, value=1):
        if coord is None or len(coord.shape) == 0 or coord.shape[0] == 0:
            return
        assert(self.reference is not None) # coord: N x 3
        raw_index = (coord - self.reference) / self.voxel_size
        index = np.round(raw_index).astype(np.int32)
        assert(np.abs(raw_index - index).mean() < 1e-4)
        self.set_occupancy_by_index(index, value)
        return
    
    def index_to_coord(self, index):
        return index * self.voxel_size + self.reference

    def set_occupancy_by_index_with_condition(self, index, value=1, condition=None):
        to_change = condition(self.grid)
        to_set = np.zeros(self.grid.shape, np.bool)
        index = self.filter_invalid_index(index)
        one_dim_index = index.dot(self.e[:3][::-1])
        to_set[one_dim_index] = True
        self.grid[to_change & to_set] = value
        return
    
    def set_occupancy_by_coord_with_condition(self, coord, value=1, condition=None):
        if coord is None or len(coord.shape) == 0:
            return
        assert(self.reference is not None) # coord: N x 3
        raw_index = (coord - self.reference) / self.voxel_size
        index = np.round(raw_index).astype(np.int32)
        assert(np.abs(raw_index - index).mean() < 1e-3)
        self.set_occupancy_by_index_with_condition(index, value, condition)
        return




class OccupancyGridMultiDim(OccupancyGrid):

    def __init__(self, spatial_size, dim, voxel_size=None, reference=None, init_value=0, dtype=np.float32):
        super().__init__(spatial_size, voxel_size, reference, init_value, dtype)
        self.dim = dim
        self.grid = np.ones((self.e[-1], dim), dtype) * init_value
        return
    
    def get_grid(self):
        return self.grid.reshape(self.spatial_size + (self.dim,))
    
    def get_occupied_index(self, condition=None):
        assert(False)
        grid = self.get_grid()
        if condition is None:
            return np.stack(np.nonzero(grid != 0), axis=-1)
        else:
            return np.stack(np.nonzero(condition(grid)), axis=-1)

    def get_occupied_coord(self, condition=None):
        assert(False)
        return self.get_occupied_index(condition) * self.voxel_size + self.reference

    def get_all_grid_coord(self):
        assert(False)
        return self.get_occupied_coord(condition=lambda x: x < np.inf)


class PseudoColorConverter(object):

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = matplotlib.pyplot.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        return

    def convert(self, val):
        return np.round(self.scalarMap.to_rgba(val)[..., :3] * 255).astype(np.uint8)

def filter_points(pts, bbox, init_mask, return_mask=False):
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    mask = init_mask
    mask &= pts[..., 0] >= x_min
    mask &= pts[..., 0] <= x_max
    mask &= pts[..., 1] >= y_min
    mask &= pts[..., 1] <= y_max
    mask &= pts[..., 2] >= z_min
    mask &= pts[..., 2] <= z_max
    # print(' filter points mask mean', mask.mean())
    if return_mask:
        return pts[mask], mask
    else:
        return pts[mask]

def closest(vec, arr):
    # vec:     d
    # arr: n x d
    diff = np.abs(arr - vec)
    idx = np.argmin(diff, axis=0)
    return arr[idx, np.array(range(arr.shape[0]))]

def interplate_poses(p1, p2, num):
    # p1, p2: 4 x 4
    r1 = R.from_matrix(p1[:3, :3]).as_euler('zxy', degrees=True)
    r2 = R.from_matrix(p2[:3, :3]).as_euler('zxy', degrees=True)
    r2 = closest(r1, np.stack([r2-360, r2, r2+360]))
    q1 = np.concatenate([r1, p1[:3, 3]])
    q2 = np.concatenate([r2, p2[:3, 3]])
    lines = np.linspace(q1, q2, num=num)
    result = np.zeros((num, 4, 4), np.float32)
    for i in range(num):
        result[i, :3, :3] = R.from_euler('zxy', lines[i, :3], degrees=True).as_matrix()
        result[i, :3, 3] = lines[i, 3:]
        result[i, 3, 3] = 1.0
    return result
