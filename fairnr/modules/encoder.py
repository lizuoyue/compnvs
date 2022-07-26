# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
import math
import sys
import os
import math
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from plyfile import PlyData, PlyElement

from fairnr.data.data_utils import load_matrix
from fairnr.data.geometry import (
    trilinear_interp, splitting_points, offset_points,
    get_edge, build_easy_octree, discretize_points
)
from fairnr.clib import (
    aabb_ray_intersect, triangle_ray_intersect, svo_ray_intersect,
    uniform_ray_sampling, inverse_cdf_sampling
)
from fairnr.modules.module_utils import (
    FCBlock, Linear, Embedding,
    InvertableMapping
)
MAX_DEPTH = 10000.0
ENCODER_REGISTRY = {}

def register_encoder(name):
    def register_encoder_cls(cls):
        if name in ENCODER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        ENCODER_REGISTRY[name] = cls
        return cls
    return register_encoder_cls


def get_encoder(name):
    if name not in ENCODER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return ENCODER_REGISTRY[name]


@register_encoder('abstract_encoder')
class Encoder(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass


@register_encoder('volume_encoder')
class VolumeEncoder(Encoder):
    
    def __init__(self, args):
        super().__init__(args)

        self.context = None

    @staticmethod
    def add_args(parser):
        parser.add_argument('--near', type=float, help='near distance of the volume')
        parser.add_argument('--far',  type=float, help='far distance of the volume')

    def precompute(self, id=None, context=None, *args, **kwargs):
        self.context = context  # save context which maybe useful later
        return {}   # we do not use encoder for NeRF

    def ray_intersect(self, ray_start, ray_dir, encoder_states, near=None, far=None):
        S, V, P, _ = ray_dir.size()
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()
        near = near if near is not None else self.args.near
        far = far if far is not None else self.args.far
        intersection_outputs = {
            "min_depth": ray_dir.new_ones(S, V * P, 1) * near,
            "max_depth": ray_dir.new_ones(S, V * P, 1) * far,
            "probs": ray_dir.new_ones(S, V * P, 1),
            "steps": ray_dir.new_ones(S, V * P) * self.args.fixed_num_samples,
            "intersected_voxel_idx": ray_dir.new_zeros(S, V * P, 1).int()}
        hits = ray_dir.new_ones(S, V * P).bool()
        return ray_start, ray_dir, intersection_outputs, hits

    def ray_sample(self, intersection_outputs):
        sampled_idx, sampled_depth, sampled_dists = inverse_cdf_sampling(
            intersection_outputs['intersected_voxel_idx'], 
            intersection_outputs['min_depth'], 
            intersection_outputs['max_depth'], 
            intersection_outputs['probs'],
            intersection_outputs['steps'], -1, (not self.training))
        return {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,  # dummy index (to match raymarcher)
        }

    def forward(self, samples, encoder_states):
        inputs = {
            'pos': samples['sampled_point_xyz'].requires_grad_(True),
            'ray': samples['sampled_point_ray_direction'],
            'dists': samples['sampled_point_distance']
        }
        if self.context is not None:
            inputs.update({'context': self.context})
        return inputs

@register_encoder('infinite_volume_encoder')
class InfiniteVolumeEncoder(VolumeEncoder):

    def __init__(self, args):
        super().__init__(args)
        self.imap = InvertableMapping(style='simple')
        self.nofixdz = getattr(args, "no_fix_dz", False)
        self.sample_msi = getattr(args, "sample_msi", False)

    @staticmethod
    def add_args(parser):
        VolumeEncoder.add_args(parser)
        parser.add_argument('--no-fix-dz', action='store_true', help='do not fix dz.')
        parser.add_argument('--sample-msi', action='store_true')

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        S, V, P, _ = ray_dir.size()
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()
        
        # ray sphere (unit) intersection (assuming all camera is inside sphere):
        p_v = (ray_start * ray_dir).sum(-1)
        p_p = (ray_start * ray_start).sum(-1)
        d_u = -p_v + torch.sqrt(p_v ** 2 - p_p + 1)
        
        intersection_outputs = {
            "min_depth": torch.arange(-1, 1, 1, dtype=ray_dir.dtype, device=ray_dir.device)[None, None, :].expand(S, V * P, 2),
            "max_depth": torch.arange( 0, 2, 1, dtype=ray_dir.dtype, device=ray_dir.device)[None, None, :].expand(S, V * P, 2),
            "probs": ray_dir.new_ones(S, V * P, 2) * .5,
            "steps": ray_dir.new_ones(S, V * P, 1) * self.args.fixed_num_samples,
            "intersected_voxel_idx": torch.arange( 0, 2, 1, device=ray_dir.device)[None, None, :].expand(S, V * P, 2).int(),
            "unit_sphere_depth": d_u,
            "p_v": p_v, "p_p": p_p}
        hits = ray_dir.new_ones(S, V * P).bool()
        return ray_start, ray_dir, intersection_outputs, hits
        
    def ray_sample(self, intersection_outputs):
        samples = super().ray_sample(intersection_outputs)   # HACK: < 1, unit sphere;  > 1, outside the sphere
        
        # map from (0, 1) to (0, +inf) with invertable mapping
        samples['original_point_distance'] = samples['sampled_point_distance'].clone()
        samples['original_point_depth'] = samples['sampled_point_depth'].clone()
        
        # assign correct depth
        in_depth = intersection_outputs['unit_sphere_depth'][:, None] * (
            samples['original_point_depth'].clamp(max=0.0) + 1.0).masked_fill(samples['sampled_point_voxel_idx'].ne(0), 0)
        if not self.sample_msi:
            out_depth = (intersection_outputs['unit_sphere_depth'][:, None] + 1 / (1 - samples['original_point_depth'].clamp(min=0.0) + 1e-7) - 1
                ).masked_fill(samples['sampled_point_voxel_idx'].ne(1), 0)
        else:
            p_v, p_p = intersection_outputs['p_v'][:, None], intersection_outputs['p_p'][:, None]
            out_depth = (-p_v + torch.sqrt(p_v ** 2 - p_p + 1. / (1. - samples['original_point_depth'].clamp(min=0.0) + 1e-7) ** 2)
                ).masked_fill(samples['sampled_point_voxel_idx'].ne(1), 0)
        samples['sampled_point_depth'] = in_depth + out_depth

        if not self.nofixdz:
            # raise NotImplementedError("need to re-compute later")
            in_dists = 1 / intersection_outputs['unit_sphere_depth'][:, None] * (samples['original_point_distance']).masked_fill(
                samples['sampled_point_voxel_idx'].ne(0), 0)
            alpha = 1. if not self.sample_msi else 1. / torch.sqrt(1. + (p_v ** 2 - p_p) * (1. - samples['original_point_depth'].clamp(min=0.0) + 1e-7) ** 2)
            out_dists = alpha / ((1 - samples['original_point_depth'].clamp(min=0.0)) ** 2 + 1e-7) * (samples['original_point_distance']).masked_fill(
                samples['sampled_point_voxel_idx'].ne(1), 0)
            samples['sampled_point_distance'] = in_dists + out_dists
        else:
            samples['sampled_point_distance'] = samples['sampled_point_distance'].scatter(1, 
                samples['sampled_point_voxel_idx'].ne(-1).sum(-1, keepdim=True) - 1, 1e8)
        
        return samples

    def forward(self, samples, encoder_states):
        field_inputs = super().forward(samples, encoder_states)

        r = field_inputs['pos'].norm(p=2, dim=-1, keepdim=True) # .clamp(min=1.0)
        field_inputs['pos'] = torch.cat([field_inputs['pos'] / (r + 1e-8), r / (1.0 + r)], dim=-1)
        return field_inputs


@register_encoder('sparsevoxel_encoder')
class SparseVoxelEncoder(Encoder):

    def __init__(self, args, voxel_path=None, bbox_path=None, shared_values=None):
        super().__init__(args)
        # read initial voxels or learned sparse voxels
        self.voxel_path = voxel_path if voxel_path is not None else args.voxel_path
        self.bbox_path = bbox_path if bbox_path is not None else getattr(args, "initial_boundingbox", None)
        assert (self.bbox_path is not None) or (self.voxel_path is not None), \
            "at least initial bounding box or pretrained voxel files are required."
        self.voxel_index = None
        self.scene_scale = getattr(args, "scene_scale", 1.0)

        if self.voxel_path is not None:
            # read voxel file
            assert os.path.exists(self.voxel_path), "voxel file must exist"
            
            if Path(self.voxel_path).suffix == '.ply':
                from plyfile import PlyData, PlyElement
                plyvoxel = PlyData.read(self.voxel_path)
                elements = [x.name for x in plyvoxel.elements]
                
                assert 'vertex' in elements
                plydata = plyvoxel['vertex']
                fine_points = torch.from_numpy(
                    np.stack([plydata['x'], plydata['y'], plydata['z']]).astype('float32').T)

                if 'face' in elements:
                    # read voxel meshes... automatically detect voxel size
                    faces = plyvoxel['face']['vertex_indices']
                    t = fine_points[faces[0].astype('int64')]
                    voxel_size = torch.abs(t[0] - t[1]).max()

                    # indexing voxel vertices
                    fine_points = torch.unique(fine_points, dim=0)

                    # vertex_ids, _ = discretize_points(fine_points, voxel_size)
                    # vertex_ids_offset = vertex_ids + 1
                    
                    # # simple hashing
                    # vertex_ids = vertex_ids[:, 0] * 1000000 + vertex_ids[:, 1] * 1000 + vertex_ids[:, 2]
                    # vertex_ids_offset = vertex_ids_offset[:, 0] * 1000000 + vertex_ids_offset[:, 1] * 1000 + vertex_ids_offset[:, 2]

                    # vertex_ids = {k: True for k in vertex_ids.tolist()}
                    # vertex_inside = [v in vertex_ids for v in vertex_ids_offset.tolist()]
                    
                    # # get voxel centers
                    # fine_points = fine_points[torch.tensor(vertex_inside)] + voxel_size * .5
                    # fine_points = fine_points + voxel_size * .5   --> use all corners as centers
                
                else:
                    # voxel size must be provided
                    assert getattr(args, "voxel_size", None) is not None, "final voxel size is essential."
                    voxel_size = args.voxel_size

                if 'quality' in elements:
                    self.voxel_index = torch.from_numpy(plydata['quality']).long()
               
            else:
                # supporting the old style .txt voxel points
                # fine_points = torch.from_numpy(np.loadtxt(self.voxel_path)[:, 3:].astype('float32'))
                fine_points = torch.from_numpy(np.loadtxt(self.voxel_path)[:, :3].astype('float32'))
                voxel_size = args.voxel_size
        else:
            # read bounding-box file
            bbox = np.loadtxt(self.bbox_path)
            voxel_size = bbox[-1] if getattr(args, "voxel_size", None) is None else args.voxel_size
            fine_points = torch.from_numpy(bbox2voxels(bbox[:6], voxel_size))
        
        half_voxel = voxel_size * .5
        
        # transform from voxel centers to voxel corners (key/values)
        fine_coords, _ = discretize_points(fine_points, half_voxel)
        fine_keys0 = offset_points(fine_coords, 1.0).reshape(-1, 3)
        fine_keys, fine_feats = torch.unique(fine_keys0, dim=0, sorted=True, return_inverse=True)
        fine_feats = fine_feats.reshape(-1, 8)
        num_keys = torch.scalar_tensor(fine_keys.size(0)).long()
        
        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * voxel_size
        else:
            step_size = args.raymarching_stepsize
        
        # register parameters (will be saved to checkpoints)
        self.register_buffer("points", fine_points)          # voxel centers
        self.register_buffer("keys", fine_keys.long())       # id used to find voxel corners/embeddings
        self.register_buffer("feats", fine_feats.long())     # for each voxel, 8 voxel corner ids
        self.register_buffer("num_keys", num_keys)
        self.register_buffer("keep", fine_feats.new_ones(fine_feats.size(0)).long())  # whether the voxel will be pruned

        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        logger.info("loaded {} voxel centers, {} voxel corners".format(fine_points.size(0), num_keys))

        # set-up other hyperparameters and initialize running time caches
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)    
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        # sparse voxel embeddings     
        if shared_values is None and self.embed_dim > 0:
            self.values = Embedding(num_keys, self.embed_dim, None)
        else:
            self.values = shared_values
        
        if getattr(args, "fix_embedding", False) is True:
            for name, param in self.values.named_parameters():
                param.requires_grad = False
    
    def init_embedding(self, init_file):
        import MinkowskiEngine as ME
        
        mat = np.loadtxt(init_file).astype(np.int32)
        coord = torch.from_numpy(mat[:, :3]).to(self.keys.device).float()
        feat = torch.from_numpy(mat[:, 3:]).to(self.keys.device).float()
        
        coords, feats = ME.utils.sparse_collate([coord], [feat])
        q_coords, _ = ME.utils.sparse_collate([self.keys], [self.keys])
        mat = ME.SparseTensor(features=feats, coordinates=coords)
        init_feats = mat.features_at_coordinates(q_coords.float()) # N, 4
        init_feats = init_feats / torch.maximum(init_feats[:,3:], init_feats[:,3:]*0+1e-6)
        init_feats[:,:3] /= 255.0
        init_feats *= 2
        init_feats -= 1

        nn.init.constant_(self.values.weight, 0.0)
        with torch.no_grad():
            res = []
            for i in range(self.embed_dim // 8):
                res.append(torch.cos(init_feats * (2**i) * np.pi))
                res.append(torch.sin(init_feats * (2**i) * np.pi))
            res = torch.cat(res, dim=-1)
            self.values.weight += res
        return

    def upgrade_state_dict_named(self, state_dict, name):
        # update the voxel embedding shapes
        if self.values is not None:
            loaded_values = state_dict[name + '.values.weight']
            self.values.weight = nn.Parameter(self.values.weight.new_zeros(*loaded_values.size()))
            self.values.num_embeddings = self.values.weight.size(0)
            self.total_size = self.values.weight.size(0)
            self.num_keys = self.num_keys * 0 + self.total_size
        
        if self.voxel_index is not None:
            state_dict[name + '.points'] = state_dict[name + '.points'][self.voxel_index]
            state_dict[name + '.feats'] = state_dict[name + '.feats'][self.voxel_index]
            state_dict[name + '.keep'] = state_dict[name + '.keep'][self.voxel_index]
        
        # update the buffers shapes
        if name + '.points' in state_dict:
            self.points = self.points.new_zeros(*state_dict[name + '.points'].size())
            self.feats  = self.feats.new_zeros(*state_dict[name + '.feats'].size())
            self.keys   = self.keys.new_zeros(*state_dict[name + '.keys'].size())
            self.keep   = self.keep.new_zeros(*state_dict[name + '.keep'].size())
        
        else:
            # this usually happens when loading a NeRF checkpoint to NSVF
            # use initialized values
            state_dict[name + '.points'] = self.points
            state_dict[name + '.feats'] = self.feats
            state_dict[name + '.keys'] = self.keys
            state_dict[name + '.keep'] = self.keep
    
            state_dict[name + '.voxel_size'] = self.voxel_size
            state_dict[name + '.step_size'] = self.step_size
            state_dict[name + '.max_hits'] = self.max_hits
            state_dict[name + '.num_keys'] = self.num_keys

    @staticmethod
    def add_args(parser):
        parser.add_argument('--initial-boundingbox', type=str, help='the initial bounding box to initialize the model')
        parser.add_argument('--voxel-size', type=float, metavar='D', help='voxel size of the input points (initial')
        parser.add_argument('--voxel-path', type=str, help='path for pretrained voxel file. if provided no update')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-stepsize-ratio', type=float, metavar='D',
                            help='if the concrete step size is not given (=0), we use the ratio to the voxel size as step size.')
        parser.add_argument('--use-octree', action='store_true', help='if set, instead of looping over the voxels, we build an octree.')
        parser.add_argument('--track-max-probs', action='store_true', help='if set, tracking the maximum probability in ray-marching.')
        parser.add_argument('--scene-scale', type=float, default=1.0)

    def reset_runtime_caches(self):
        logger.info("reset chache")
        if self.use_octree:
            points = self.points[self.keep.bool()]
            centers, children = build_easy_octree(points, self.voxel_size / 2.0)
            self._runtime_caches['flatten_centers'] = centers
            self._runtime_caches['flatten_children'] = children
        if self.track_max_probs:
            self._runtime_caches['max_voxel_probs'] = self.points.new_zeros(self.points.size(0))

    def clean_runtime_caches(self):
        logger.info("clean chache")
        for name in self._runtime_caches:
            self._runtime_caches[name] = None

    def precompute(self, id=None, *args, **kwargs):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        # points[:, 0] += (self.voxel_size / 10)
        values = self.values.weight[: self.num_keys] if self.values is not None else None
        
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values
        }

        if self.use_octree:
            flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
            if id is not None:
                flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
                flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
            encoder_states['voxel_octree_center_xyz'] = flatten_centers
            encoder_states['voxel_octree_children_idx'] = flatten_children
        return encoder_states

    @torch.no_grad()
    def export_voxels(self, return_mesh=False):
        logger.info("exporting learned sparse voxels...")
        voxel_idx = torch.arange(self.keep.size(0), device=self.keep.device)
        voxel_idx = voxel_idx[self.keep.bool()]
        voxel_pts = self.points[self.keep.bool()]
        if not return_mesh:
            # HACK: we export the original voxel indices as "quality" in case for editing
            points = [
                (voxel_pts[k, 0], voxel_pts[k, 1], voxel_pts[k, 2], voxel_idx[k])
                for k in range(voxel_idx.size(0))
            ]
            vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4')])
            return PlyData([PlyElement.describe(vertex, 'vertex')])
        
        else:
            # generate polygon for voxels
            center_coords, residual = discretize_points(voxel_pts, self.voxel_size / 2)
            offsets = torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1],[1,1,1]], device=center_coords.device)
            vertex_coords = center_coords[:, None, :] + offsets[None, :, :]
            vertex_points = vertex_coords.type_as(residual) * self.voxel_size / 2 + residual
            
            faceidxs = [[1,6,7,5],[7,6,2,4],[5,7,4,3],[1,0,2,6],[1,5,3,0],[0,3,4,2]]
            all_vertex_keys, all_vertex_idxs  = {}, []
            for i in range(vertex_coords.shape[0]):
                for j in range(8):
                    key = " ".join(["{}".format(int(p)) for p in vertex_coords[i,j]])
                    if key not in all_vertex_keys:
                        all_vertex_keys[key] = vertex_points[i,j]
                        all_vertex_idxs += [key]
            all_vertex_dicts = {key: u for u, key in enumerate(all_vertex_idxs)}
            all_faces = torch.stack([torch.stack([vertex_coords[:, k] for k in f]) for f in faceidxs]).permute(2,0,1,3).reshape(-1,4,3)
    
            all_faces_keys = {}
            for l in range(all_faces.size(0)):
                key = " ".join(["{}".format(int(p)) for p in all_faces[l].sum(0) // 4])
                if key not in all_faces_keys:
                    all_faces_keys[key] = all_faces[l]

            vertex = np.array([tuple(all_vertex_keys[key].cpu().tolist()) for key in all_vertex_idxs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            face = np.array([([all_vertex_dicts["{} {} {}".format(*b)] for b in a.cpu().tolist()],) for a in all_faces_keys.values()],
                dtype=[('vertex_indices', 'i4', (4,))])
            return PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')])

    @torch.no_grad()
    def export_surfaces(self, field_fn, th, bits):
        """
        extract triangle-meshes from the implicit field using marching cube algorithm
            Lewiner, Thomas, et al. "Efficient implementation of marching cubes' cases with topological guarantees." 
            Journal of graphics tools 8.2 (2003): 1-15.
        """
        logger.info("marching cube...")
        encoder_states = self.precompute(id=None)
        points = encoder_states['voxel_center_xyz']

        scores = self.get_scores(field_fn, th=th, bits=bits, encoder_states=encoder_states)
        coords, residual = discretize_points(points, self.voxel_size)
        A, B, C = [s + 1 for s in coords.max(0).values.cpu().tolist()]
    
        # prepare grids
        full_grids = points.new_ones(A * B * C, bits ** 3)
        full_grids[coords[:, 0] * B * C + coords[:, 1] * C + coords[:, 2]] = scores
        full_grids = full_grids.reshape(A, B, C, bits, bits, bits)
        full_grids = full_grids.permute(0, 3, 1, 4, 2, 5).reshape(A * bits, B * bits, C * bits)
        full_grids = 1 - full_grids

        # marching cube
        from skimage import measure
        space_step = self.voxel_size.item() / bits
        verts, faces, normals, _ = measure.marching_cubes_lewiner(
            volume=full_grids.cpu().numpy(), level=0.5,
            spacing=(space_step, space_step, space_step)
        )
        verts += (residual - (self.voxel_size / 2)).cpu().numpy()
        verts = np.array([tuple(a) for a in verts.tolist()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        faces = np.array([(a, ) for a in faces.tolist()], dtype=[('vertex_indices', 'i4', (3,))])
        return PlyData([PlyElement.describe(verts, 'vertex'), PlyElement.describe(faces, 'face')])

    def get_edge(self, ray_start, ray_dir, samples, encoder_states):
        outs = get_edge(
            ray_start + ray_dir * samples['sampled_point_depth'][:, :1], 
            encoder_states['voxel_center_xyz'].reshape(-1, 3)[samples['sampled_point_voxel_idx'][:, 0].long()], 
            self.voxel_size).type_as(ray_dir)   # get voxel edges/depth (for visualization)
        outs = (1 - outs[:, None].expand(outs.size(0), 3)) * 0.7
        return outs

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()

        if self.use_octree:  # ray-voxel intersection with SVO
            flatten_centers = encoder_states['voxel_octree_center_xyz']
            flatten_children = encoder_states['voxel_octree_children_idx']
            pts_idx, min_depth, max_depth = svo_ray_intersect(
                self.voxel_size, self.max_hits, flatten_centers, flatten_children,
                ray_start, ray_dir)
        else:   # ray-voxel intersection with all voxels
            pts_idx, min_depth, max_depth = aabb_ray_intersect(
                self.voxel_size, self.max_hits, point_xyz, ray_start, ray_dir)

        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        
        if S > 1:  # extend the point-index to multiple shapes (just in case)
            pts_idx = (pts_idx + H * torch.arange(S, 
                device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
                ).masked_fill_(pts_idx.eq(-1), -1)

        intersection_outputs = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "intersected_voxel_idx": pts_idx
        }
        return ray_start, ray_dir, intersection_outputs, hits

    def ray_sample(self, intersection_outputs):
        # sample points and use middle point approximation
        sampled_idx, sampled_depth, sampled_dists = inverse_cdf_sampling(
            intersection_outputs['intersected_voxel_idx'],
            intersection_outputs['min_depth'], 
            intersection_outputs['max_depth'], 
            intersection_outputs['probs'],
            intersection_outputs['steps'], 
            -1, self.deterministic_step or (not self.training))
        sampled_dists = sampled_dists.clamp(min=0.0)
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        sampled_dists.masked_fill_(sampled_idx.eq(-1), 0.0)
        
        samples = {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,
        }
        return samples

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        # encoder states
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']

        # ray point samples
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        sampled_xyz = samples['sampled_point_xyz'].requires_grad_(True)
        sampled_dir = samples['sampled_point_ray_direction']
        sampled_dis = samples['sampled_point_distance']

        # prepare inputs for implicit field
        #  / self.scene_scale
        inputs = {
            'pos': sampled_xyz, 
            'ray': sampled_dir, 
            'dists': sampled_dis}

        # --- just for debugging ---- #
        # r = inputs['pos'].norm(p=2, dim=-1, keepdim=True)
        # inputs['pos'] = torch.cat([inputs['pos'] / (r + 1e-8), r / (1 + r)], dim=-1)

        if values is not None:
            # resample point features
            point_xyz = F.embedding(sampled_idx, point_xyz)
            point_feats = F.embedding(F.embedding(sampled_idx, point_feats), values).view(point_xyz.size(0), -1)

            # tri-linear interpolation
            p = ((sampled_xyz - point_xyz) / self.voxel_size + .5).unsqueeze(1)
            q = offset_points(p, .5, offset_only=True).unsqueeze(0) + .5   # BUG (FIX)
            inputs.update({'emb': trilinear_interp(p, q, point_feats)})

        return inputs

    @torch.no_grad()
    def track_voxel_probs(self, voxel_idxs, voxel_probs):
        voxel_idxs = voxel_idxs.masked_fill(voxel_idxs.eq(-1), self.max_voxel_probs.size(0))
        chunk_size = 4096
        for start in range(0, voxel_idxs.size(0), chunk_size):
            end = start + chunk_size
            end = end if end < voxel_idxs.size(0) else voxel_idxs.size(0)
            max_voxel_probs = self.max_voxel_probs.new_zeros(end-start, self.max_voxel_probs.size(0) + 1).scatter_add_(
                dim=-1, index=voxel_idxs[start:end], src=voxel_probs[start:end]).max(0)[0][:-1].data        
            self.max_voxel_probs = torch.max(self.max_voxel_probs, max_voxel_probs)
    
    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, encoder_states=None, train_stats=False):
        if not train_stats:
            logger.info("pruning...")
            scores = self.get_scores(field_fn, th=th, bits=16, encoder_states=encoder_states)
            keep = (1 - scores.min(-1)[0]) > th
        else:
            logger.info("pruning based on training set statics (e.g. probs)...")
            if dist.is_initialized() and dist.get_world_size() > 1:  # sync on multi-gpus
                dist.all_reduce(self.max_voxel_probs, op=dist.ReduceOp.MAX)
            keep = self.max_voxel_probs > th
            
        self.keep.masked_scatter_(self.keep.bool(), keep.long())
        logger.info("pruning done. # of voxels before: {}, after: {} voxels".format(keep.size(0), keep.sum()))
    
    def get_scores(self, field_fn, th=0.5, bits=16, encoder_states=None):
        if encoder_states is None:
            encoder_states = self.precompute(id=None)
        
        feats = encoder_states['voxel_vertex_idx'] 
        points = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']
        chunk_size = 64

        def get_scores_once(feats, points, values):
            # sample points inside voxels
            sampled_xyz = offset_points(points, self.voxel_size / 2.0, bits=bits)
            sampled_idx = torch.arange(points.size(0), device=points.device)[:, None].expand(*sampled_xyz.size()[:2])
            sampled_xyz, sampled_idx = sampled_xyz.reshape(-1, 3), sampled_idx.reshape(-1)
            
            field_inputs = self.forward(
                {'sampled_point_xyz': sampled_xyz, 
                 'sampled_point_voxel_idx': sampled_idx,
                 'sampled_point_ray_direction': None,
                 'sampled_point_distance': None}, 
                {'voxel_vertex_idx': feats,
                 'voxel_center_xyz': points,
                 'voxel_vertex_emb': values})  # get field inputs
            if encoder_states.get('context', None) is not None:
                field_inputs['context'] = encoder_states['context']
            
            # evaluation with density
            field_outputs = field_fn(field_inputs, outputs=['sigma'])
            free_energy = -torch.relu(field_outputs['sigma']).reshape(-1, bits ** 3)
            
            # return scores
            return torch.exp(free_energy)

        return torch.cat([get_scores_once(feats[i: i + chunk_size], points[i: i + chunk_size], values) 
            for i in range(0, points.size(0), chunk_size)], 0)

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        encoder_states = self.precompute(id=None)
        feats, points, values = encoder_states['voxel_vertex_idx'], encoder_states['voxel_center_xyz'], encoder_states['voxel_vertex_emb']
        new_points, new_feats, new_values, new_keys = splitting_points(points, feats, values, self.voxel_size / 2.0)
        new_num_keys = new_keys.size(0)
        new_point_length = new_points.size(0)
        
        # set new voxel embeddings
        if new_values is not None:
            self.values.weight = nn.Parameter(new_values)
            self.values.num_embeddings = self.values.weight.size(0)
        
        self.total_size = new_num_keys
        self.num_keys = self.num_keys * 0 + self.total_size

        self.points = new_points
        self.feats = new_feats
        self.keep = self.keep.new_ones(new_point_length)
        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(points.size(0), self.keep.sum()))
        
    @property
    def flatten_centers(self):
        if self._runtime_caches['flatten_centers'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_centers']
    
    @property
    def flatten_children(self):
        if self._runtime_caches['flatten_children'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_children']

    @property
    def max_voxel_probs(self):
        if self._runtime_caches['max_voxel_probs'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['max_voxel_probs']

    @max_voxel_probs.setter
    def max_voxel_probs(self, x):
        self._runtime_caches['max_voxel_probs'] = x

    @property
    def feature_dim(self):
        return self.embed_dim

    @property
    def dummy_loss(self):
        if self.values is not None:
            return self.values.weight[0,0] * 0.0
        return 0.0
    
    @property
    def num_voxels(self):
        return self.keep.long().sum()


@register_encoder('multi_sparsevoxel_encoder')
class MultiSparseVoxelEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)
        # try:
        self.all_voxels = nn.ModuleList(
            [SparseVoxelEncoder(args, vox.strip()) for vox in open(args.voxel_path).readlines()]
        )

        self.init_method = getattr(args, "embedding_init", "normal")
        if self.init_method == "normal":
            pass
        elif self.init_method == "const":
            for voxel in self.all_voxels:
                nn.init.constant_(voxel.values.weight, 0.0)
        else:
            assert(self.init_method.endswith(".txt"))
            for voxel, init_file in zip(self.all_voxels, open(self.init_method).readlines()):
                print(f'init voxel from {init_file} ...')
                voxel.init_embedding(init_file.strip())

        # except TypeError:
        #     bbox_path = getattr(args, "bbox_path", "/private/home/jgu/data/shapenet/disco_dataset/bunny_point.txt")
        #     self.all_voxels = nn.ModuleList(
        #         [SparseVoxelEncoder(args, None, g.strip() + '/bbox.txt') for g in open(bbox_path).readlines()])
        
        # properties
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False) 

        self.cid = None
        if getattr(self.args, "global_embeddings", None) is not None:
            self.global_embed = torch.zeros(*eval(self.args.global_embeddings)).normal_(mean=0, std=0.01)
            self.global_embed = nn.Parameter(self.global_embed, requires_grad=True)
        else:
            self.global_embed = None

    @staticmethod
    def add_args(parser):
        SparseVoxelEncoder.add_args(parser)
        parser.add_argument('--bbox-path', type=str, default=None)
        parser.add_argument('--global-embeddings', type=str, default=None,
            help="""set global embeddings if provided in global.txt. We follow this format:
                (N, D) or (K, N, D) if we have multi-dimensional global features. 
                D is the global feature dimentions. 
                N is the number of indices of this feature, 
                and K is the number of features if provided.""")

    def reset_runtime_caches(self):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].reset_runtime_caches()
    
    def clean_runtime_caches(self):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].clean_runtime_caches()

    def precompute(self, id, global_index=None, *args, **kwargs):
        # TODO: this is a HACK for simplicity
        assert id.size(0) == 1, "for now, only works for one object"
        
        # HACK
        # id = id * 0 + 2
        self.cid = id[0]
        encoder_states = self.all_voxels[id[0]].precompute(id, *args, **kwargs)
        if (global_index is not None) and (self.global_embed is not None):
            encoder_states['context'] = torch.stack([
                F.embedding(global_index[:, i], self.global_embed[i])
                for i in range(self.global_embed.size(0))], 1)
        return encoder_states

    def export_surfaces(self, field_fn, th, bits):
        raise NotImplementedError("does not support for now.")

    def export_voxels(self, return_mesh=False):
        raise NotImplementedError("does not support for now.")
    
    def get_edge(self, *args, **kwargs):
        return self.all_voxels[self.cid].get_edge(*args, **kwargs)

    def ray_intersect(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_intersect(*args, **kwargs)

    def ray_sample(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_sample(*args, **kwargs)

    def forward(self, samples, encoder_states):
        inputs = self.all_voxels[self.cid].forward(samples, encoder_states)
        if encoder_states.get('context', None) is not None:
            inputs['context'] = encoder_states['context']
        return inputs

    def track_voxel_probs(self, voxel_idxs, voxel_probs):
        return self.all_voxels[self.cid].track_voxel_probs(voxel_idxs, voxel_probs)

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, train_stats=False):
        for id in range(len(self.all_voxels)):
           self.all_voxels[id].pruning(field_fn, th, train_stats=train_stats)
    
    @torch.no_grad()
    def splitting(self):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].splitting()

    @property
    def feature_dim(self):
        return self.all_voxels[0].embed_dim

    @property
    def dummy_loss(self):
        return sum([d.dummy_loss for d in self.all_voxels])

    @property
    def voxel_size(self):
        return self.all_voxels[0].voxel_size

    @voxel_size.setter
    def voxel_size(self, x):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].voxel_size = x

    @property
    def step_size(self):
        return self.all_voxels[0].step_size

    @step_size.setter
    def step_size(self, x):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].step_size = x

    @property
    def max_hits(self):
        return self.all_voxels[0].max_hits

    @max_hits.setter
    def max_hits(self, x):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].max_hits = x

    @property
    def num_voxels(self):
        return self.all_voxels[self.cid].num_voxels


@register_encoder('shared_sparsevoxel_encoder')
class SharedSparseVoxelEncoder(MultiSparseVoxelEncoder):
    """
    Different from MultiSparseVoxelEncoder, we assume a shared list 
    of voxels across all models. Usually useful to learn a video sequence.  
    """
    def __init__(self, args):
        super(MultiSparseVoxelEncoder, self).__init__(args)

        # using a shared voxel
        self.voxel_path = args.voxel_path
        self.num_frames = args.num_frames
        self.all_voxels = [SparseVoxelEncoder(args, self.voxel_path)]
        self.all_voxels =  nn.ModuleList(self.all_voxels + [
            SparseVoxelEncoder(args, self.voxel_path, shared_values=self.all_voxels[0].values)
            for i in range(self.num_frames - 1)])
        self.context_embed_dim = args.context_embed_dim
        self.contexts = nn.Embedding(self.num_frames, self.context_embed_dim, None)
        self.cid = None

    @staticmethod
    def add_args(parser):
        SparseVoxelEncoder.add_args(parser)
        parser.add_argument('--num-frames', type=int, help='the total number of frames')
        parser.add_argument('--context-embed-dim', type=int, help='context embedding for each view')

    def forward(self, samples, encoder_states):
        inputs = self.all_voxels[self.cid].forward(samples, encoder_states)
        inputs.update({'context': self.contexts(self.cid).unsqueeze(0)})
        return inputs

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, train_stats=False):
        for cid in range(len(self.all_voxels)):
           id = torch.tensor([cid], device=self.contexts.weight.device)
           encoder_states = {name: v[0] if v is not None else v 
                    for name, v in self.precompute(id).items()}
           encoder_states['context'] = self.contexts(id)
           self.all_voxels[cid].pruning(field_fn, th, 
                encoder_states=encoder_states,
                train_stats=train_stats)

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        all_feats, all_points = [], []
        for id in range(len(self.all_voxels)):
            encoder_states = self.all_voxels[id].precompute(id=None)
            feats = encoder_states['voxel_vertex_idx']
            points = encoder_states['voxel_center_xyz']
            values = encoder_states['voxel_vertex_emb']

            all_feats.append(feats)
            all_points.append(points)
        
        feats, points = torch.cat(all_feats, 0), torch.cat(all_points, 0)
        unique_feats, unique_idx = torch.unique(feats, dim=0, return_inverse=True)
        unique_points = points[
            unique_feats.new_zeros(unique_feats.size(0)).scatter_(
                0, unique_idx, torch.arange(unique_idx.size(0), device=unique_feats.device)
        )]
        new_points, new_feats, new_values, new_keys = splitting_points(unique_points, unique_feats, values, self.voxel_size / 2.0)
        new_num_keys = new_keys.size(0)
        new_point_length = new_points.size(0)

        # set new voxel embeddings (shared voxels)
        if values is not None:
            self.all_voxels[0].values.weight = nn.Parameter(new_values)
            self.all_voxels[0].values.num_embeddings = new_num_keys

        for id in range(len(self.all_voxels)):
            self.all_voxels[id].total_size = new_num_keys
            self.all_voxels[id].num_keys = self.all_voxels[id].num_keys * 0 + self.all_voxels[id].total_size

            self.all_voxels[id].points = new_points
            self.all_voxels[id].feats = new_feats
            self.all_voxels[id].keep = self.all_voxels[id].keep.new_ones(new_point_length)

        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(
            unique_points.size(0), new_point_length))

    @property
    def feature_dim(self):
        return self.all_voxels[0].embed_dim + self.context_embed_dim


@register_encoder('triangle_mesh_encoder')
class TriangleMeshEncoder(SparseVoxelEncoder):
    """
    Training on fixed mesh model. Cannot pruning..
    """
    def __init__(self, args, mesh_path=None, shared_values=None):
        super(SparseVoxelEncoder, self).__init__(args)
        self.mesh_path = mesh_path if mesh_path is not None else args.mesh_path
        assert (self.mesh_path is not None) and os.path.exists(self.mesh_path)
        
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
        faces = torch.from_numpy(np.asarray(mesh.triangles, dtype=np.long))

        step_size = args.raymarching_stepsize
        if getattr(args, "raymarching_margin", None) is None:
            margin = step_size * 10  # truncated space around the triangle surfaces
        else:
            margin = args.raymarching_margin
        
        self.register_buffer("margin", torch.scalar_tensor(margin))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        self.vertices = nn.Parameter(vertices, requires_grad=getattr(args, "trainable_vertices", False))
        self.faces = nn.Parameter(faces, requires_grad=False)

        # set-up other hyperparameters
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.values = None
        self.blur_ratio = getattr(args, "blur_ratio", 0.0)

    def upgrade_state_dict_named(self, state_dict, name):
        pass
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--mesh-path', type=str, help='path for initial mesh file')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-margin', type=float, default=None,
                            help='margin around the surface.')
        parser.add_argument('--blur-ratio', type=float, default=0,
                            help="it is possible to shoot outside the triangle. default=0")
        parser.add_argument('--trainable-vertices', action='store_true',
                            help='if set, making the triangle trainable. experimental code. not ideal.')

    def precompute(self, id=None, *args, **kwargs):
        feats, points, values = self.faces, self.vertices, self.values
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None
            
            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + points.size(1) * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]

        encoder_states = {
            'mesh_face_vertex_idx': feats,
            'mesh_vertex_xyz': points,
        }
        return encoder_states

    def get_edge(self, ray_start, ray_dir, *args, **kwargs):
        return torch.ones_like(ray_dir) * 0.7

    @property
    def voxel_size(self):
        return self.margin

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        point_xyz = encoder_states['mesh_vertex_xyz']
        point_feats = encoder_states['mesh_face_vertex_idx']
        
        S, V, P, _ = ray_dir.size()
        F, G = point_feats.size(1), point_xyz.size(1)
  
        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()
        pts_idx, depth, uv = triangle_ray_intersect(
            self.margin, self.blur_ratio, self.max_hits, point_xyz, point_feats, ray_start, ray_dir)
        min_depth = (depth[:,:,:,0] + depth[:,:,:,1]).masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth = (depth[:,:,:,0] + depth[:,:,:,2]).masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object

        if S > 1:  # extend the point-index to multiple shapes (just in case)
            pts_idx = (pts_idx + G * torch.arange(S, 
                device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
                ).masked_fill_(pts_idx.eq(-1), -1)

        intersection_outputs = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "intersected_voxel_idx": pts_idx
        }
        return ray_start, ray_dir, intersection_outputs, hits

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        return {
            'pos': samples['sampled_point_xyz'].requires_grad_(True),
            'ray': samples['sampled_point_ray_direction'],
            'dists': samples['sampled_point_distance']
        }

    @property
    def num_voxels(self):
        return self.vertices.size(0)


def bbox2voxels(bbox, voxel_size):
    vox_min, vox_max = bbox[:3], bbox[3:]
    steps = ((vox_max - vox_min) / voxel_size).round().astype('int64') + 1
    x, y, z = [c.reshape(-1).astype('float32') for c in np.meshgrid(np.arange(steps[0]), np.arange(steps[1]), np.arange(steps[2]))]
    x, y, z = x * voxel_size + vox_min[0], y * voxel_size + vox_min[1], z * voxel_size + vox_min[2]
    
    return np.stack([x, y, z]).T.astype('float32')





































import sparseconvnet as scn
# sys.path.append('/home/lzq/lzy/NSVF/fairnr/modules/')
# from randlanet import RandLANet

class SparseConvUNet(nn.Module):

    def __init__(self, input_dim, output_dim, spatial, reps=1, n_planes=[i * 32 for i in [1, 2, 3, 4, 5]]):
        # input_dim: number of input feature channels
        # spatial: spatial size of the voxel grid
        # In case of repetition in coords:
        # mode == 0 if the input is guaranteed to have no duplicates
        # mode == 1 to use the last item at each spatial location
        # mode == 2 to keep the first item at each spatial location
        # mode == 3 to sum feature vectors sharing one spatial location
        # mode == 4 to average feature vectors at each spatial location
        nn.Module.__init__(self)
        self.dim = 3 # 3D
        self.reps = reps # conv block repetition factor
        self.n_planes = n_planes # unet number of features per level
        self.unet = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4)).add(
            scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0], filter_size=3, bias=False)).add(
            scn.UNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2])).add(
            scn.BatchNormReLU(self.n_planes[0])).add(
            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(self.n_planes[0], output_dim)
    
    def forward(self, x):
        x = self.unet(x)
        x = self.linear(x)
        return x

class SparseConvFCN(nn.Module):

    def __init__(self, input_dim, output_dim, spatial, reps=1, n_planes=[i * 64 for i in [1, 2, 3, 4]]):
        nn.Module.__init__(self)
        self.dim = 3 # 3D
        self.reps = reps # conv block repetition factor
        self.n_planes = n_planes # number of features per level
        self.fcn = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4)).add(
            scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0], filter_size=3, bias=False)).add(
            scn.FullyConvolutionalNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2])).add(
            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(sum(self.n_planes), output_dim)

    def forward(self, x):
        x = self.fcn(x)
        x = self.linear(x)
        return x



class GatedSparseConvUNet(nn.Module):

    def __init__(self, input_dim, output_dim, spatial):
        # input_dim: number of input feature channels
        # spatial: spatial size of the voxel grid
        # In case of repetition in coords:
        # mode == 0 if the input is guaranteed to have no duplicates
        # mode == 1 to use the last item at each spatial location
        # mode == 2 to keep the first item at each spatial location
        # mode == 3 to sum feature vectors sharing one spatial location
        # mode == 4 to average feature vectors at each spatial location
        nn.Module.__init__(self)
        self.dim = 3 # 3D
        self.reps = 1 # conv block repetition factor
        self.n_planes = [i * 32 for i in [2, 4, 8, 16]] # unet number of features per level
        self.unet = scn.Sequential().add(
            scn.InputLayer(self.dim, spatial, mode=4)).add(
            scn.Gated(scn.SubmanifoldConvolution(self.dim, input_dim, self.n_planes[0] * 2, filter_size=3, bias=True), 0.1, True)).add(
            scn.GatedUNet(self.dim, self.reps, self.n_planes, residual_blocks=True, downsample=[2, 2], leakiness=0.1, batchnorm=True)).add(
            scn.OutputLayer(self.dim))
        self.linear = nn.Linear(self.n_planes[0], output_dim)
    
    def forward(self, x):
        x = self.unet(x)
        x = self.linear(x)
        return x



























































@register_encoder('reg_sparsevoxel_encoder')
class RegularizedSparseVoxelEncoder(SparseVoxelEncoder):

    def __init__(self, args, voxel_path=None, bbox_path=None, shared_values=None):
        super(SparseVoxelEncoder, self).__init__(args)

        self.dim_compressed = args.dim_compressed

        # read initial voxels or learned sparse voxels
        self.voxel_path = voxel_path if voxel_path is not None else args.voxel_path
        self.bbox_path = bbox_path if bbox_path is not None else getattr(args, "initial_boundingbox", None)
        assert (self.bbox_path is not None) or (self.voxel_path is not None), \
            "at least initial bounding box or pretrained voxel files are required."
        self.voxel_index = None
        self.scene_scale = getattr(args, "scene_scale", 1.0)

        fine_points = torch.from_numpy(np.loadtxt(self.voxel_path)[:, :3].astype('float32'))
        voxel_size = args.voxel_size
        half_voxel = voxel_size * .5
        
        # transform from voxel centers to voxel corners (key/values)
        fine_coords, _ = discretize_points(fine_points, half_voxel)
        fine_keys0 = offset_points(fine_coords, 1.0).reshape(-1, 3)
        fine_keys, fine_feats = torch.unique(fine_keys0, dim=0, sorted=True, return_inverse=True)

        self.locations = ((fine_keys + 1.0) / 2.0).long().contiguous()
        self.activation = torch.nn.Tanh()

        fine_feats = fine_feats.reshape(-1, 8)
        num_keys = torch.scalar_tensor(fine_keys.size(0)).long()
        
        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * voxel_size
        else:
            step_size = args.raymarching_stepsize
        
        # register parameters (will be saved to checkpoints)
        self.register_buffer("points", fine_points)          # voxel centers
        self.register_buffer("keys", fine_keys.long())       # id used to find voxel corners/embeddings
        self.register_buffer("feats", fine_feats.long())     # for each voxel, 8 voxel corner ids
        self.register_buffer("num_keys", num_keys)
        self.register_buffer("keep", fine_feats.new_ones(fine_feats.size(0)).long())  # whether the voxel will be pruned

        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        logger.info("loaded {} voxel centers, {} voxel corners".format(fine_points.size(0), num_keys))

        # set-up other hyperparameters and initialize running time caches
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)    
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        # sparse voxel embeddings     
        if shared_values is None and self.embed_dim > 0:
            self.values = Embedding(num_keys, self.dim_compressed, None)
        else:
            self.values = shared_values

    def precompute(self, id=None, *args, **kwargs):
        assert('scn_model' in kwargs)
        scn_model = kwargs['scn_model']

        print('Regularizer does not require grad')
        for name, param in scn_model.named_parameters():
            param.requires_grad = False

        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        init_values = self.values.weight[: self.num_keys] if self.values is not None else None
        init_values = self.activation(init_values)

        if kwargs['reg_net'].startswith('scn'):
            values = scn_model((self.locations, init_values, 1))
        else:
            to_input = torch.cat([self.locations.to(init_values.device), init_values], dim=-1)
            values = scn_model(to_input.unsqueeze(0))

        values = self.activation(values)
        
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values
        }

        if self.use_octree:
            flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
            if id is not None:
                flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
                flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
            encoder_states['voxel_octree_center_xyz'] = flatten_centers
            encoder_states['voxel_octree_children_idx'] = flatten_children
        return encoder_states



@register_encoder('reg_multi_sparsevoxel_encoder')
class RegularizedMultiSparseVoxelEncoder(MultiSparseVoxelEncoder):
    def __init__(self, args):
        super(MultiSparseVoxelEncoder, self).__init__(args)
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.spatial = torch.ones(3).long() * 1024

        self.reg_net = getattr(args, "reg_net", None)
        assert(self.reg_net in ['scn', 'scn_randlanet', 'randlanet'])
        if self.reg_net == 'scn':
            self.scn_model = SparseConvUNet(args.dim_compressed, self.embed_dim, self.spatial, 3, [i * 64 for i in [1, 2, 3, 4, 5]])
        if self.reg_net == 'randlanet':
            self.scn_model = RandLANet(args.dim_compressed, self.embed_dim, device=torch.device('cuda:0'))
        if self.reg_net == 'scn_randlanet':
            self.scn_model = nn.Sequential().add(
                SparseConvUNet(args.dim_compressed, self.embed_dim, self.spatial, 3, [i * 64 for i in [1, 2, 3, 4, 5]])).add(
                RandLANet(self.embed_dim, self.embed_dim, device=torch.device('cuda:0')))

        self.all_voxels = nn.ModuleList(
            [RegularizedSparseVoxelEncoder(args, vox.strip()) for vox in open(args.voxel_path).readlines()]
        )

        # properties
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False) 

        self.cid = None
        if getattr(self.args, "global_embeddings", None) is not None:
            self.global_embed = torch.zeros(*eval(self.args.global_embeddings)).normal_(mean=0, std=0.01)
            self.global_embed = nn.Parameter(self.global_embed, requires_grad=True)
        else:
            self.global_embed = None

    def precompute(self, id, global_index=None, *args, **kwargs):
        # TODO: this is a HACK for simplicity
        assert id.size(0) == 1, "for now, only works for one object"

        kwargs['scn_model'] = self.scn_model
        kwargs['reg_net'] = self.reg_net
        
        # HACK
        # id = id * 0 + 2
        self.cid = id[0]
        encoder_states = self.all_voxels[id[0]].precompute(id, *args, **kwargs)
        if (global_index is not None) and (self.global_embed is not None):
            encoder_states['context'] = torch.stack([
                F.embedding(global_index[:, i], self.global_embed[i])
                for i in range(self.global_embed.size(0))], 1)
        return encoder_states


















































@register_encoder('dict_sparsevoxel_encoder')
class DictSparseVoxelEncoder(SparseVoxelEncoder):

    def __init__(self, args, voxel_path=None, bbox_path=None, shared_values=None):
        super(SparseVoxelEncoder, self).__init__(args)

        self.dim_compressed = 4

        # read initial voxels or learned sparse voxels
        self.voxel_path = voxel_path if voxel_path is not None else args.voxel_path
        self.bbox_path = bbox_path if bbox_path is not None else getattr(args, "initial_boundingbox", None)
        assert (self.bbox_path is not None) or (self.voxel_path is not None), \
            "at least initial bounding box or pretrained voxel files are required."
        self.voxel_index = None
        self.scene_scale = getattr(args, "scene_scale", 1.0)

        fine_points = torch.from_numpy(np.loadtxt(self.voxel_path)[:, :3].astype('float32'))
        voxel_size = args.voxel_size
        half_voxel = voxel_size * .5
        
        # transform from voxel centers to voxel corners (key/values)
        fine_coords, _ = discretize_points(fine_points, half_voxel)
        fine_keys0 = offset_points(fine_coords, 1.0).reshape(-1, 3)
        fine_keys, fine_feats = torch.unique(fine_keys0, dim=0, sorted=True, return_inverse=True)

        fine_feats = fine_feats.reshape(-1, 8)
        num_keys = torch.scalar_tensor(fine_keys.size(0)).long()
        
        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * voxel_size
        else:
            step_size = args.raymarching_stepsize
        
        # register parameters (will be saved to checkpoints)
        self.register_buffer("points", fine_points)          # voxel centers
        self.register_buffer("keys", fine_keys.long())       # id used to find voxel corners/embeddings
        self.register_buffer("feats", fine_feats.long())     # for each voxel, 8 voxel corner ids
        self.register_buffer("num_keys", num_keys)
        self.register_buffer("keep", fine_feats.new_ones(fine_feats.size(0)).long())  # whether the voxel will be pruned

        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        logger.info("loaded {} voxel centers, {} voxel corners".format(fine_points.size(0), num_keys))

        # set-up other hyperparameters and initialize running time caches
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)    
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        # sparse voxel embeddings     
        if shared_values is None and self.embed_dim > 0:
            self.values = Embedding(num_keys, self.dim_compressed, None)
        else:
            self.values = shared_values

    def precompute(self, id=None, *args, **kwargs):
        assert('dictionary' in kwargs)
        assert('dictionary_w' in kwargs)
        dictionary = kwargs['dictionary'] # 1024 * 32
        dictionary_w = kwargs['dictionary_w'] # 32 * 4
        d_hat = dictionary_w(dictionary) # 1024 * 4

        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        init_values = self.values.weight[: self.num_keys] if self.values is not None else None # N * 4

        prob = torch.mm(init_values, d_hat.transpose(0, 1)) # N * 1024
        prob = torch.nn.Softmax(dim=1)(prob)
        values = torch.mm(prob, dictionary)
        
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values
        }

        if self.use_octree:
            flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
            if id is not None:
                flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
                flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
            encoder_states['voxel_octree_center_xyz'] = flatten_centers
            encoder_states['voxel_octree_children_idx'] = flatten_children
        return encoder_states



@register_encoder('dict_multi_sparsevoxel_encoder')
class DictMultiSparseVoxelEncoder(MultiSparseVoxelEncoder):
    def __init__(self, args):
        super(MultiSparseVoxelEncoder, self).__init__(args)
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.dict_num = 1024

        self.dictionary = Embedding(self.dict_num, self.embed_dim, None)
        self.dictionary_w = nn.Linear(self.embed_dim, 4)

        self.all_voxels = nn.ModuleList(
            [DictSparseVoxelEncoder(args, vox.strip()) for vox in open(args.voxel_path).readlines()]
        )

        # properties
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False) 

        self.cid = None
        if getattr(self.args, "global_embeddings", None) is not None:
            self.global_embed = torch.zeros(*eval(self.args.global_embeddings)).normal_(mean=0, std=0.01)
            self.global_embed = nn.Parameter(self.global_embed, requires_grad=True)
        else:
            self.global_embed = None

    def precompute(self, id, global_index=None, *args, **kwargs):
        # TODO: this is a HACK for simplicity
        assert id.size(0) == 1, "for now, only works for one object"

        kwargs['dictionary'] = self.dictionary.weight
        kwargs['dictionary_w'] = self.dictionary_w
        
        # HACK
        # id = id * 0 + 2
        self.cid = id[0]
        encoder_states = self.all_voxels[id[0]].precompute(id, *args, **kwargs)
        if (global_index is not None) and (self.global_embed is not None):
            encoder_states['context'] = torch.stack([
                F.embedding(global_index[:, i], self.global_embed[i])
                for i in range(self.global_embed.size(0))], 1)
        return encoder_states


































import MinkowskiEngine as ME
sys.path.append('/home/lzq/lzy/minkowski_3d_completion')
# sys.path.append('/home/lzq/lzy/minkowski_3d_completion/examples')
# from minkowski_network import MinkowskiDeepFillv2
from minkowski_utils import replace_features
import resnet
from examplesMink import minkunet as minkunet_org
import minkunet as minkunet_instnorm
import time

@register_encoder('mink_sparsevoxel_encoder')
class MinkowskiSparseVoxelEncoder(SparseVoxelEncoder):

    @staticmethod
    def add_args(parser):

        parser.add_argument('--voxel-size', type=float, metavar='D', help='voxel size of the input points (initial)')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-stepsize-ratio', type=float, metavar='D',
                            help='if the concrete step size is not given (=0), we use the ratio to the voxel size as step size.')
        parser.add_argument('--use-octree', action='store_true', help='if set, instead of looping over the voxels, we build an octree.')
        parser.add_argument('--track-max-probs', action='store_true', help='if set, tracking the maximum probability in ray-marching.')
        parser.add_argument('--scene-scale', type=float, default=1.0)

        return


    def __init__(self, args):
        Encoder.__init__(self, args)

        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * args.voxel_size
        else:
            step_size = args.raymarching_stepsize
        
        self.register_buffer("voxel_size", torch.scalar_tensor(args.voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits)) # default 60?
        self.register_buffer("num_voxels", torch.scalar_tensor(0)) # dummy, not used

        self.embed_dim = getattr(args, "voxel_embed_dim", None) # default 32?
        self.deterministic_step = getattr(args, "deterministic_step", False) # default False?
        self.use_octree = False # getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)  # default False?
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        backbone_name = getattr(args, "backbone_name", "MinkUNet34C")
        if backbone_name is None:
            backbone_name = "MinkUNet34C"
        if backbone_name.endswith('_inst'):
            backbone = getattr(minkunet_instnorm, backbone_name[:-5], None)
        else:
            backbone = getattr(minkunet_org, backbone_name, None)
        assert(backbone is not None)

        if backbone_name == "MinkUNet34C":
            self.replace_ft = True
        elif backbone_name == "MinkUNet50":
            self.replace_ft = False
        else:
            assert(False)
        
        input_dim = self.embed_dim + 1
        if 'basic' in args.load_npz:
            input('input dim 3')
            input_dim = 3
            self.replace_ft = None

        self.mink_net = backbone(input_dim, self.embed_dim, D=3)
        # self.mink_net = MinkUNet34C(17, self.embed_dim)
        # self.mink_net = MinkowskiDeepFillv2(33, self.embed_dim, 64)

        return

    def upgrade_state_dict_named(self, state_dict, name):

        # When init shape different from loaded one
        # update to the loaded one's shape and init with 0
        return

    def reset_runtime_caches(self):
        print('--- reset_runtime_caches')
        return super().reset_runtime_caches()

    def clean_runtime_caches(self):
        print('--- clean_runtime_caches')
        return super().clean_runtime_caches()

    def precompute(self, id=None, *args, **kwargs):

        # should input point cloud here

        # requires voxel index of self.values.weight 

        # args: ()
        # kwargs: dict()
        # ['colors', 'intrinsics', 'extrinsics', 'uv', 'shape', 'view', 'size', 'step', 'field_fn', 'input_fn']
        # [
        #     'voxel_pts',
        #     'voxel_to_vertex',
        #     'voxel_batch',
        #     'vertex_idx',
        #     'vertex_info',
        #     'vertex_batch',
        # ]

        print('--- precompute')
        # print(list(kwargs.keys()))
        # print(kwargs['colors'].shape) # torch.Size([1, 4, 65536, 3])
        # print(kwargs['voxel_batch'].shape) # torch.Size([1, 1, 61396])
        # print(kwargs['vertex_batch'].shape) # torch.Size([1, 1, 152440])
        # print(kwargs['vertex_idx'].shape) # torch.Size([1, 3, 152440])

        # for i in range(4):
        #     print((kwargs['voxel_batch'] == i).sum())
        #     print((kwargs['vertex_batch'] == i).sum())
        # 8136 17327
        # 21353 57820
        # 18646 47259
        # 13261 30034

        locations = kwargs['vertex_idx'][0].transpose(0, 1)

        # loc_bias = kwargs['vertex_idx'][0].min(dim=-1)[0]
        # locations = kwargs['vertex_idx'][0].transpose(0, 1) - loc_bias
        # locations = torch.cat([locations, kwargs['vertex_batch'][0].transpose(0, 1)], dim=1)

        locations = locations.contiguous().int()
        locations = torch.cat([locations[:,:1]*0, locations], dim=1)
        features = kwargs['vertex_input'][0].transpose(0, 1).float()

        feats = kwargs['center_to_vertex'][0].transpose(0, 1)
        points = kwargs['center_pts'][0].transpose(0, 1)

        features[:, -1] *= -1
        features[:, -1] += 1

        net_in = ME.SparseTensor(coordinates=locations, features=features)
        try:
            # values = self.mink_net(net_in).F
            values = self.mink_net(net_in)
        except Exception as error:
            torch.cuda.empty_cache()
            print(error)
            time.sleep(5)
            with torch.no_grad():
                values = self.mink_net(net_in)
            # values = replace_features(net_in, net_in.F[:,:-1] * 0)
        
        if self.replace_ft == True:
            print('Replace Feature')
            if type(values) == tuple:
                values = values[1].F
            else:
                values = values.F
            values = values * features[:, -1:] + features[:, :-1] * (1-features[:, -1:])
            emb_loss_mask = features[:, -1]#torch.round(features[:, -1]).bool()
        elif self.replace_ft == False:
            values += replace_features(net_in, net_in.F[:,:-1])
            if type(values) == tuple:
                values = values[1].F
            else:
                values = values.F
            emb_loss_mask = (features[:, -1] * 2.0 + 1.0).unsqueeze(1).expand(*values.shape).clone()
            emb_loss_mask[:, :values.shape[1]//4] = emb_loss_mask[:, :values.shape[1]//4] / 4.0
            #torch.round(features[:, -1] * 0.0 + 1.0).bool()
        else:
            values = values.F
            emb_loss_mask = (values[:, -1].detach() * 0.0 + 1.0).unsqueeze(1).expand(*values.shape).clone()
        # values = kwargs['vertex_output'][0].transpose(0, 1)

        # print(locations.is_contiguous())
        # values = self.scn_unet((locations, features, batch_size))

        # values_gt = kwargs['vertex_feature'][0].transpose(0, 1)
        # values = values_gt

        # print('------------')

        # print(values.shape, values)
        # print(values_gt.shape, values_gt)
        # print(torch.abs(values-values_gt).mean())
        # quit()

        # if kwargs['vertex_feature'] is None:
        #     values = self.scn_model((locations, features, batch_size))
        # else:
        #     print('here')
        #     values = kwargs['vertex_feature'][0].transpose(0, 1)

        if False:
            # from sklearn.decomposition import PCA
            # pca = PCA(n_components=3)
            # values_vis = pca.fit_transform(values.cpu().numpy())
            # vmin = values_vis.min(axis=0)
            # vmax = values_vis.max(axis=0)
            # values_vis = (values_vis - vmin) / (vmax - vmin)
            
            # For debugging, write to point cloud txt file
            with open('pc_emb.txt', 'w') as f:
                pts = kwargs['vertex_idx'][0].transpose(0, 1).cpu().numpy() * 0.1
                # pts += kwargs['vertex_ref'][0].transpose(0, 1).cpu().numpy()
                for pt, c in zip(pts, features.cpu().numpy()):
                    x, y, z = pt
                    r, g, b, n, m = c.astype(np.int)
                    if n > 0:
                        f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
            # with open('pc_scn.txt', 'w') as f:
            #     pts = kwargs['vertex_idx'][0].transpose(0, 1).cpu().numpy() * 0.05
            #     for pt, c in zip(pts, features.cpu().numpy()):
            #         x, y, z = pt
            #         r, g, b, a = ((c+1) * 127.5).astype(np.int)
            #         if a > 130:
            #             f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
            quit()

        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats.contiguous().clone(),
            'voxel_center_xyz': points.contiguous().clone(),
            'voxel_vertex_emb': values.contiguous().clone(),
            'emb_loss_mask': emb_loss_mask,
            'emb_loss_ft': values,
            'dis_real_rgb': kwargs['colors'],
            # 'batch_size': batch_size,
        }

        # if self.use_octree:
        #     flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
        #     if id is not None:
        #         flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
        #         flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
        #     encoder_states['voxel_octree_center_xyz'] = flatten_centers
        #     encoder_states['voxel_octree_children_idx'] = flatten_children

        return encoder_states

    @torch.no_grad()
    def export_voxels(self, return_mesh=False):
        print('--- export_voxels')
        return super().export_voxels(return_mesh)

    @torch.no_grad()
    def export_surfaces(self, field_fn, th, bits):
        print('--- export_surfaces')
        return super().export_surfaces(field_fn, th, bits)

    def get_edge(self, ray_start, ray_dir, samples, encoder_states):
        print('--- forward x N')
        print('--- get_edge')
        return super().get_edge(ray_start, ray_dir, samples, encoder_states)

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        # first call this 
        print('--- ray_intersect')
        print('--- should add batch size in this function')
        return super().ray_intersect(ray_start, ray_dir, encoder_states)

    def ray_sample(self, intersection_outputs):
        # second call this
        print('--- ray_sample')
        return super().ray_sample(intersection_outputs)

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        # number of calls not fixed furing rendering?
        return super().forward(samples, encoder_states)

    @torch.no_grad()
    def track_voxel_probs(self, voxel_idxs, voxel_probs):
        print('--- track_voxel_probs')
        return super().track_voxel_probs(voxel_idxs, voxel_probs)

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, encoder_states=None, train_stats=False):
        print('--- pruning function not used')
        return
        # return super().track_voxel_probs(field_fn, th, encoder_states, train_stats)

    def get_scores(self, field_fn, th=0.5, bits=16, encoder_states=None):
        print('--- get_scores')
        return super().get_scores(field_fn, th, bits, encoder_states)

    @torch.no_grad()
    def splitting(self):
        print('--- splitting')
        return super().splitting()

@register_encoder('minkpc_sparsevoxel_encoder')
class MinkowskiPointCloudSparseVoxelEncoder(MinkowskiSparseVoxelEncoder):

    def __init__(self, args):
        Encoder.__init__(self, args)

        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * args.voxel_size
        else:
            step_size = args.raymarching_stepsize
        
        self.register_buffer("voxel_size", torch.scalar_tensor(args.voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits)) # default 60?
        self.register_buffer("num_voxels", torch.scalar_tensor(0)) # dummy, not used

        self.embed_dim = getattr(args, "voxel_embed_dim", None) # default 32?
        self.deterministic_step = getattr(args, "deterministic_step", False) # default False?
        self.use_octree = False # getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)  # default False?
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        backbone_name = getattr(args, "backbone_name", "MinkResNetBackbone")
        backbone = getattr(resnet, backbone_name, None)
        assert(backbone is not None)
        self.mink_net = backbone(3, self.embed_dim, D=3)
        self.pool = ME.MinkowskiSumPooling(kernel_size=2, stride=1, dimension=3).cuda()
        self.offset = torch.from_numpy(np.array([
            [-1,-1,-1],
            [-1,-1, 1],
            [-1, 1,-1],
            [-1, 1, 1],
            [ 1,-1,-1],
            [ 1,-1, 1],
            [ 1, 1,-1],
            [ 1, 1, 1],
        ], np.int)).cuda()
        return

    def precompute(self, id=None, *args, **kwargs):

        # should input point cloud here

        # requires voxel index of self.values.weight 

        # args: ()
        # kwargs: dict()
        # ['colors', 'intrinsics', 'extrinsics', 'uv', 'shape', 'view', 'size', 'step', 'field_fn', 'input_fn']
        # [
        #     'voxel_pts',
        #     'voxel_to_vertex',
        #     'voxel_batch',
        #     'vertex_idx',
        #     'vertex_info',
        #     'vertex_batch',
        # ]

        print('--- precompute')
        # print(kwargs['colors'].shape) # torch.Size([1, 4, 65536, 3])
        # print(kwargs['voxel_batch'].shape) # torch.Size([1, 1, 61396])
        # print(kwargs['vertex_batch'].shape) # torch.Size([1, 1, 152440])
        # print(kwargs['vertex_idx'].shape) # torch.Size([1, 3, 152440])

        # for i in range(4):
        #     print((kwargs['init_batch'] == i).sum())
        #     print((kwargs['voxel_batch'] == i).sum())
        #     print((kwargs['vertex_batch'] == i).sum())
        # 8136 17327
        # 21353 57820
        # 18646 47259
        # 13261 30034

        init_pts = kwargs['init_pts'][0].transpose(0, 1)
        init_rgb = kwargs['init_rgb'][0].transpose(0, 1).float() / 256.0
        init_batch = kwargs['init_batch'][0].transpose(0, 1)

        center_idx = torch.floor(init_pts / self.voxel_size * 16).int() * 2 + 1 # always odd
        vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + self.offset.repeat(center_idx.shape[0], 1) # always even
        assert((vertex_idx % 2).sum() == 0)
        vertex_idx //= 2
        vertex_idx = torch.cat([torch.repeat_interleave(init_batch, 8, dim=0), vertex_idx], dim=1)
        vertex_info = torch.repeat_interleave(init_rgb, 8, dim=0)

        in_field = ME.TensorField(
            features=vertex_info,
            coordinates=vertex_idx,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=vertex_info.device,
        )
        sparse_input = in_field.sparse()

        # with open('sparse_input.txt', 'w') as f:
        #     for (_,x,y,z),rgb in zip(sparse_input.C.detach().cpu().numpy(), sparse_input.F.detach().cpu().numpy()):
        #         r, g, b = np.round(rgb * 255).astype(np.int32)
        #         f.write(f'{x};{y};{z};{r};{g};{b}\n')
        
        if True:
            # with torch.no_grad():
            mink_values = self.mink_net(sparse_input)
            # mink_values has stride = 16

            #
            mink_ones = replace_features(mink_values, mink_values.F[:,:1] * 0 + 1)
            center_nums = self.pool(mink_ones)
            center_idx = center_nums.C[torch.round(center_nums.F[:,0]) == 8, 1:] + 8 # half stride

            # with open('vis/center.txt', 'w') as f:
            #     for (x,y,z) in center_idx:
            #         f.write(f'{x};{y};{z}\n')

            # TODO: Here batch size do not support
            vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + 8 * self.offset.repeat(center_idx.shape[0], 1)
            vertex_idx, feats = torch.unique(vertex_idx, dim=0, return_inverse=True)
            
            feats = feats.reshape(-1, 8)
            points = center_idx * self.voxel_size / 16
            values = mink_values.features_at_coordinates(torch.cat([vertex_idx[:,:1]*0, vertex_idx], dim=1).float())

            # see whether we get all valid values
            # print(((torch.round(values).int() == 0).int().sum(dim=1) == 0).int().sum())

        else:
            import glob
            sid_k = os.path.basename(kwargs['path'][0][0]).replace('.png', '')
            # files = glob.glob(f'ReplicaGenEncFtTriplets/easy/npz_triple/{sid_k}*')
            # files = glob.glob(f'ReplicaGenEncFtTriplets/easy/npz/{sid_k}*')
            files = glob.glob(f'ARKitScenesSingleViewNSVF/all/npz_predft/{sid_k}*')
            if len(files) > 0:
                # print('READ FROM npz_triple', files[0])
                print('READ FROM npz', files[0])
                npz = np.load(files[0])
                # feats = torch.from_numpy(npz['feats']).long().cuda()
                # points = torch.from_numpy(npz['points']).float().cuda()
                # values = torch.from_numpy(npz['values']).float().cuda()
                # feats = torch.from_numpy(npz['center_to_vertex']).long().cuda()
                # points = torch.from_numpy(npz['center_pts']).float().cuda()
                # values = torch.from_numpy(npz['vertex_output'][:,:32]).float().cuda()




        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values,
            # 'emb_loss_mask': torch.round(features[:, -1]).bool(),
            # 'emb_loss_ft': values,
            # 'batch_size': batch_size,
        }

        return encoder_states

























@register_encoder('scn_sparsevoxel_encoder')
class SparseConvNetSparseVoxelEncoder(SparseVoxelEncoder):

    @staticmethod
    def add_args(parser):

        parser.add_argument('--voxel-size', type=float, metavar='D', help='voxel size of the input points (initial)')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-stepsize-ratio', type=float, metavar='D',
                            help='if the concrete step size is not given (=0), we use the ratio to the voxel size as step size.')
        parser.add_argument('--use-octree', action='store_true', help='if set, instead of looping over the voxels, we build an octree.')
        parser.add_argument('--track-max-probs', action='store_true', help='if set, tracking the maximum probability in ray-marching.')
        parser.add_argument('--scene-scale', type=float, default=1.0)

        return


    def __init__(self, args):
        Encoder.__init__(self, args)

        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * args.voxel_size
        else:
            step_size = args.raymarching_stepsize
        
        self.register_buffer("voxel_size", torch.scalar_tensor(args.voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits)) # default 60?
        self.register_buffer("num_voxels", torch.scalar_tensor(0)) # dummy, not used

        self.embed_dim = getattr(args, "voxel_embed_dim", None) # default 32?
        self.deterministic_step = getattr(args, "deterministic_step", False) # default False?
        self.use_octree = False # getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)  # default False?
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        self.spatial = torch.ones(3).long() * 1024
        # self.scn_unet = SparseConvUNet(4, self.embed_dim, self.spatial)
        self.scn_model = SparseConvUNet(4, self.embed_dim, self.spatial)

        return

    def upgrade_state_dict_named(self, state_dict, name):

        # When init shape different from loaded one
        # update to the loaded one's shape and init with 0
        return

        print('--- upgrade_state_dict_named')

        print(name) # encoder
        print(list(state_dict.keys())) # ckpt's state_dict

        print(self.feats.shape)
        print(self.points.shape)
        print(self.values.weight.shape)
        print()

        # update the voxel embedding shapes
        if self.values is not None:
            loaded_values = state_dict[name + '.values.weight']
            self.values.weight = nn.Parameter(self.values.weight.new_zeros(*loaded_values.size()))
            self.values.num_embeddings = self.values.weight.size(0)
            self.total_size = self.values.weight.size(0)
            self.num_keys = self.num_keys * 0 + self.total_size
        
        if self.voxel_index is not None:
            state_dict[name + '.points'] = state_dict[name + '.points'][self.voxel_index]
            state_dict[name + '.feats'] = state_dict[name + '.feats'][self.voxel_index]
            state_dict[name + '.keep'] = state_dict[name + '.keep'][self.voxel_index]
        
        print(self.feats.shape)
        print(self.points.shape)
        print(self.values.weight.shape)
        print()
        
        # update the buffers shapes
        if name + '.points' in state_dict:
            self.points = self.points.new_zeros(*state_dict[name + '.points'].size())
            self.feats  = self.feats.new_zeros(*state_dict[name + '.feats'].size())
            self.keys   = self.keys.new_zeros(*state_dict[name + '.keys'].size())
            self.keep   = self.keep.new_zeros(*state_dict[name + '.keep'].size())
        
        else:
            # this usually happens when loading a NeRF checkpoint to NSVF
            # use initialized values
            state_dict[name + '.points'] = self.points
            state_dict[name + '.feats'] = self.feats
            state_dict[name + '.keys'] = self.keys
            state_dict[name + '.keep'] = self.keep
    
            state_dict[name + '.voxel_size'] = self.voxel_size
            state_dict[name + '.step_size'] = self.step_size
            state_dict[name + '.max_hits'] = self.max_hits
            state_dict[name + '.num_keys'] = self.num_keys
        

        print(self.feats.shape)
        print(self.points.shape)
        print(self.values.weight.shape)
        print(self.points[0])
        print(self.points[1])

    

    def reset_runtime_caches(self):
        print('--- reset_runtime_caches')
        return super().reset_runtime_caches()

    def clean_runtime_caches(self):
        print('--- clean_runtime_caches')
        return super().clean_runtime_caches()

    def precompute(self, id=None, *args, **kwargs):

        # should input point cloud here

        # requires voxel index of self.values.weight 

        # args: ()
        # kwargs: dict()
        # ['colors', 'intrinsics', 'extrinsics', 'uv', 'shape', 'view', 'size', 'step', 'field_fn', 'input_fn']
        # [
        #     'voxel_pts',
        #     'voxel_to_vertex',
        #     'voxel_batch',
        #     'vertex_idx',
        #     'vertex_info',
        #     'vertex_batch',
        # ]

        print('--- precompute')
        # print(list(kwargs.keys()))
        # print(kwargs['colors'].shape) # torch.Size([1, 4, 65536, 3])
        # print(kwargs['voxel_batch'].shape) # torch.Size([1, 1, 61396])
        # print(kwargs['vertex_batch'].shape) # torch.Size([1, 1, 152440])
        # print(kwargs['vertex_idx'].shape) # torch.Size([1, 3, 152440])

        # for i in range(4):
        #     print((kwargs['voxel_batch'] == i).sum())
        #     print((kwargs['vertex_batch'] == i).sum())
        # 8136 17327
        # 21353 57820
        # 18646 47259
        # 13261 30034

        locations = kwargs['vertex_idx'][0].transpose(0, 1)

        # loc_bias = kwargs['vertex_idx'][0].min(dim=-1)[0]
        # locations = kwargs['vertex_idx'][0].transpose(0, 1) - loc_bias
        # locations = torch.cat([locations, kwargs['vertex_batch'][0].transpose(0, 1)], dim=1)

        locations = locations.long()
        features = kwargs['vertex_info'][0].transpose(0, 1).float()
        batch_size = kwargs['colors'].shape[1]

        feats = kwargs['voxel_to_vertex'][0].transpose(0, 1)
        points = kwargs['voxel_pts'][0].transpose(0, 1)

        values = self.scn_model((locations.contiguous(), features, batch_size))
        # print(locations.is_contiguous())
        # values = self.scn_unet((locations, features, batch_size))

        # values_gt = kwargs['vertex_feature'][0].transpose(0, 1)
        # values = values_gt

        # print('------------')

        # print(values.shape, values)
        # print(values_gt.shape, values_gt)
        # print(torch.abs(values-values_gt).mean())
        # quit()

        # if kwargs['vertex_feature'] is None:
        #     values = self.scn_model((locations, features, batch_size))
        # else:
        #     print('here')
        #     values = kwargs['vertex_feature'][0].transpose(0, 1)

        if False:
            # from sklearn.decomposition import PCA
            # pca = PCA(n_components=3)
            # values_vis = pca.fit_transform(values.cpu().numpy())
            # vmin = values_vis.min(axis=0)
            # vmax = values_vis.max(axis=0)
            # values_vis = (values_vis - vmin) / (vmax - vmin)
            
            # For debugging, write to point cloud txt file
            with open('pc_emb.txt', 'w') as f:
                pts = kwargs['vertex_idx'][0].transpose(0, 1).cpu().numpy() * 0.1
                # pts += kwargs['vertex_ref'][0].transpose(0, 1).cpu().numpy()
                for pt, c in zip(pts, features.cpu().numpy()):
                    x, y, z = pt
                    r, g, b, n, m = c.astype(np.int)
                    if n > 0:
                        f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
            # with open('pc_scn.txt', 'w') as f:
            #     pts = kwargs['vertex_idx'][0].transpose(0, 1).cpu().numpy() * 0.05
            #     for pt, c in zip(pts, features.cpu().numpy()):
            #         x, y, z = pt
            #         r, g, b, a = ((c+1) * 127.5).astype(np.int)
            #         if a > 130:
            #             f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
            quit()

        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values,
            # 'batch_size': batch_size,
        }

        # if self.use_octree:
        #     flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
        #     if id is not None:
        #         flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
        #         flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
        #     encoder_states['voxel_octree_center_xyz'] = flatten_centers
        #     encoder_states['voxel_octree_children_idx'] = flatten_children

        return encoder_states

    @torch.no_grad()
    def export_voxels(self, return_mesh=False):
        print('--- export_voxels')
        return super().export_voxels(return_mesh)

    @torch.no_grad()
    def export_surfaces(self, field_fn, th, bits):
        print('--- export_surfaces')
        return super().export_surfaces(field_fn, th, bits)

    def get_edge(self, ray_start, ray_dir, samples, encoder_states):
        print('--- forward x N')
        print('--- get_edge')
        return super().get_edge(ray_start, ray_dir, samples, encoder_states)

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        # first call this 
        print('--- ray_intersect')
        print('--- should add batch size in this function')
        return super().ray_intersect(ray_start, ray_dir, encoder_states)

    def ray_sample(self, intersection_outputs):
        # second call this
        print('--- ray_sample')
        return super().ray_sample(intersection_outputs)

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        # number of calls not fixed furing rendering?
        return super().forward(samples, encoder_states)

    @torch.no_grad()
    def track_voxel_probs(self, voxel_idxs, voxel_probs):
        print('--- track_voxel_probs')
        return super().track_voxel_probs(voxel_idxs, voxel_probs)

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, encoder_states=None, train_stats=False):
        print('--- pruning function not used')
        return
        # return super().track_voxel_probs(field_fn, th, encoder_states, train_stats)

    def get_scores(self, field_fn, th=0.5, bits=16, encoder_states=None):
        print('--- get_scores')
        return super().get_scores(field_fn, th, bits, encoder_states)

    @torch.no_grad()
    def splitting(self):
        print('--- splitting')
        return super().splitting()






@register_encoder('gated_scn_sparsevoxel_encoder')
class GatedSparseConvNetSparseVoxelEncoder(SparseConvNetSparseVoxelEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.scn_model = GatedSparseConvUNet(4, self.embed_dim, self.spatial)

@register_encoder('advanced_scn_sparsevoxel_encoder')
class AdvancedSparseConvNetSparseVoxelEncoder(SparseConvNetSparseVoxelEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.scn_model = SparseConvUNet(4, self.embed_dim, self.spatial, 3, [i * 64 for i in [1, 2, 3, 4, 5]])

@register_encoder('fcn_scn_sparsevoxel_encoder')
class FCNSparseConvNetSparseVoxelEncoder(SparseConvNetSparseVoxelEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.scn_model = SparseConvFCN(4, self.embed_dim, self.spatial, 2, [i * 64 for i in [1, 2, 3, 4]])









sys.path.append('/home/lzq/lzy/spsg/torch')
from model import GeoGeneratorDFv2

@register_encoder('geo_scn_sparsevoxel_encoder')
class GeoSparseConvNetSparseVoxelEncoder(SparseConvNetSparseVoxelEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.spatial = torch.ones(3).long() * 1024
        # self.scn_model = SparseConvUNet(33, self.embed_dim, self.spatial, reps=2, n_planes=[i * 48 for i in [1, 2, 3, 4, 5]])
        self.scn_model = SparseConvUNet(33, self.embed_dim, self.spatial, reps=2, n_planes=[i * 32 for i in [1, 2, 3, 4, 5]])
        # self.scn_model = SparseConvUNet(5, 4, self.spatial, reps=2, n_planes=[i * 32 for i in [1, 2, 3, 4, 5]])
        self.geo_model = GeoGeneratorDFv2(nf_in_geo=5, nf=64, max_data_size=[160,80,192])
        self.offset = torch.Tensor([
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],
        ]).long()
        self.e = torch.Tensor([
            80*192,
            192,
            1,
        ]).long()

        # self.reg_scn_model = SparseConvUNet(4, 32, self.spatial, 3, [i * 64 for i in [1, 2, 3, 4, 5]])
    




    # def precompute_backup(self, id=None, *args, **kwargs):

    #     # should input point cloud here

    #     # requires voxel index of self.values.weight 

    #     # args: ()
    #     # kwargs: dict()
    #     # ['colors', 'intrinsics', 'extrinsics', 'uv', 'shape', 'view', 'size', 'step', 'field_fn', 'input_fn']
    #     # [
    #     #     'voxel_pts',
    #     #     'voxel_to_vertex',
    #     #     'voxel_batch',
    #     #     'vertex_idx',
    #     #     'vertex_info',
    #     #     'vertex_batch',
    #     # ]

    #     print('--- precompute of geo')
    #     geo_x = kwargs['geo_x'][0]
    #     geo_y = kwargs['geo_y'][0]
    #     geo_mask = kwargs['geo_mask'][0]
    #     info_grid = kwargs['info_grid'][0, 0].transpose(0, 1)

    #     pred_y = self.geo_model(geo_x)[:, 0]
    #     pred_y_01 = torch.nn.Sigmoid()(pred_y) > 0.9
    #     geo_y = geo_y[:, 0]

    #     pred_y_01 = pred_y_01 * geo_mask.long() + geo_y * (1 - geo_mask.long())
    #     pred_y_01 = pred_y_01.bool().detach()





    #     # ptx, pty, ptz = np.meshgrid(np.arange(128), np.arange(64), np.arange(128), indexing='ij')
    #     # pts = np.stack([ptx, pty, ptz], axis=-1)
    #     # pts = np.round(pts).astype(np.int32)

    #     # sel5 = pred_y_01.detach().cpu().numpy() # (1, 128, 64, 128)

    #     # with open(f'pc/pc_5.txt', 'w') as f:
    #     #     for pt in pts[sel5[0]]:
    #     #         f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))



    #     center_idx = torch.nonzero(pred_y_01[0, :-1, :-1, :-1])
    #     vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + self.offset.to(center_idx.device).repeat(center_idx.shape[0], 1)
    #     vertex_idx, feats = torch.unique(vertex_idx, dim=0, return_inverse=True)

    #     locations = vertex_idx.long()
    #     locations_1d = locations[:, 0] * self.e[0] + locations[:, 1] * self.e[1] + locations[:, 2] * self.e[2]
    #     features = info_grid[locations_1d].float()

    #     # feats = kwargs['voxel_to_vertex'][0].transpose(0, 1)
    #     # points = kwargs['voxel_pts'][0].transpose(0, 1)
    #     feats = feats.reshape((-1, 8))
    #     points = (center_idx.float() + 0.5) * 0.1 + kwargs['vertex_ref'][0, :, 0]

    #     values = self.scn_model((locations.contiguous(), features, 1))

    #     # print(locations.is_contiguous())
    #     # values = self.scn_unet((locations, features, batch_size))

    #     # values_gt = kwargs['vertex_feature'][0].transpose(0, 1)
    #     # values = values_gt

    #     # print('------------')

    #     # print(values.shape, values)
    #     # print(values_gt.shape, values_gt)
    #     # print(torch.abs(values-values_gt).mean())
    #     # quit()

    #     # if kwargs['vertex_feature'] is None:
    #     #     values = self.scn_model((locations, features, batch_size))
    #     # else:
    #     #     print('here')
    #     #     values = kwargs['vertex_feature'][0].transpose(0, 1)

    #     if False:
    #         # from sklearn.decomposition import PCA
    #         # pca = PCA(n_components=3)
    #         # values_vis = pca.fit_transform(values.cpu().numpy())
    #         # vmin = values_vis.min(axis=0)
    #         # vmax = values_vis.max(axis=0)
    #         # values_vis = (values_vis - vmin) / (vmax - vmin)
            
    #         # For debugging, write to point cloud txt file
    #         with open('pc_emb.txt', 'w') as f:
    #             pts = kwargs['vertex_idx'][0].transpose(0, 1).cpu().numpy() * 0.1
    #             # pts += kwargs['vertex_ref'][0].transpose(0, 1).cpu().numpy()
    #             for pt, c in zip(pts, features.cpu().numpy()):
    #                 x, y, z = pt
    #                 r, g, b, n, m = c.astype(np.int)
    #                 if n > 0:
    #                     f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
    #         # with open('pc_scn.txt', 'w') as f:
    #         #     pts = kwargs['vertex_idx'][0].transpose(0, 1).cpu().numpy() * 0.05
    #         #     for pt, c in zip(pts, features.cpu().numpy()):
    #         #         x, y, z = pt
    #         #         r, g, b, a = ((c+1) * 127.5).astype(np.int)
    #         #         if a > 130:
    #         #             f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
    #         quit()

    #     if id is not None:
    #         # extend size to support multi-objects
    #         feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
    #         points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
    #         values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

    #         # moving to multiple objects
    #         if id.size(0) > 1:
    #             feats = feats + self.num_keys * torch.arange(id.size(0), 
    #                 device=feats.device, dtype=feats.dtype)[:, None, None]
        
    #     encoder_states = {
    #         'voxel_vertex_idx': feats,
    #         'voxel_center_xyz': points,
    #         'voxel_vertex_emb': values,
    #         'geo': pred_y,
    #         # 'batch_size': batch_size,
    #     }

    #     # if self.use_octree:
    #     #     flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
    #     #     if id is not None:
    #     #         flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
    #     #         flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
    #     #     encoder_states['voxel_octree_center_xyz'] = flatten_centers
    #     #     encoder_states['voxel_octree_children_idx'] = flatten_children

    #     return encoder_states

    def precompute(self, id=None, *args, **kwargs):

        # should input point cloud here

        # requires voxel index of self.values.weight 

        # args: ()
        # kwargs: dict()
        # ['colors', 'intrinsics', 'extrinsics', 'uv', 'shape', 'view', 'size', 'step', 'field_fn', 'input_fn']
        # [
        #     'voxel_pts',
        #     'voxel_to_vertex',
        #     'voxel_batch',
        #     'vertex_idx',
        #     'vertex_info',
        #     'vertex_batch',
        # ]

        print('--- precompute of geo, eccv 2022')

        # geo_x = kwargs['geo_x'][0]
        # geo_y = kwargs['geo_y'][0]
        # geo_mask = kwargs['geo_mask'][0]
        # info_grid = kwargs['info_grid'][0, 0].transpose(0, 1)

        # pred_y = self.geo_model(geo_x)[:, 0]
        # pred_y_01 = torch.nn.Sigmoid()(pred_y) > 0.9
        # geo_y = geo_y[:, 0]

        # pred_y_01 = pred_y_01 * geo_mask.long() + geo_y * (1 - geo_mask.long())
        # pred_y_01 = pred_y_01.bool().detach()

        # ptx, pty, ptz = np.meshgrid(np.arange(128), np.arange(64), np.arange(128), indexing='ij')
        # pts = np.stack([ptx, pty, ptz], axis=-1)
        # pts = np.round(pts).astype(np.int32)

        # sel5 = pred_y_01.detach().cpu().numpy() # (1, 128, 64, 128)

        # with open(f'pc/pc_5.txt', 'w') as f:
        #     for pt in pts[sel5[0]]:
        #         f.write('%d;%d;%d\n' % (pt[0], pt[1], pt[2]))



        # center_idx = torch.nonzero(pred_y_01[0, :-1, :-1, :-1])
        # vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + self.offset.to(center_idx.device).repeat(center_idx.shape[0], 1)
        # vertex_idx, feats = torch.unique(vertex_idx, dim=0, return_inverse=True)

        # locations = vertex_idx.long()
        # locations_1d = locations[:, 0] * self.e[0] + locations[:, 1] * self.e[1] + locations[:, 2] * self.e[2]
        # features = info_grid[locations_1d].float()

        # feats = feats.reshape((-1, 8))
        # points = (center_idx.float() + 0.5) * 0.1 + kwargs['vertex_ref'][0, :, 0]

        # values = self.scn_model((locations.contiguous(), features, 1))

        # print(locations.is_contiguous())
        # values = self.scn_unet((locations, features, batch_size))

        # values_gt = kwargs['vertex_feature'][0].transpose(0, 1)
        # values = values_gt

        # print('------------')

        # print(values.shape, values)
        # print(values_gt.shape, values_gt)
        # print(torch.abs(values-values_gt).mean())
        # quit()

        if True:
            locations = kwargs['vertex_idx'][0].long().transpose(0, 1).contiguous()
            features = kwargs['vertex_input'][0].transpose(0, 1)
            # sigma = kwargs['vertex_sigma'][0].transpose(0, 1)
            # mu = kwargs['vertex_mu'][0].transpose(0, 1)
            # scn_values = kwargs['vertex_output'][0].transpose(0, 1) * sigma + mu

            # print(features[features[:,-1] < 0.1, :])

            scn_values = self.scn_model((locations, features, 1))
            scn_values = 3 * nn.Tanh()(scn_values)

            # scn_values = torch.from_numpy(np.load('SparseConvNetDeepImagePrior/npz/5000_posenc_l2.npz')['dip']).float().cuda()
            # to_vis = (scn_values.cpu().numpy() + 1.0) / 2.0 * 255.0
            # to_vis = np.round(np.clip(to_vis, 0.0, 255.0)).astype(np.uint8)
            # for i in range(3):
            #     with open(f'pts/posencl2_{i:02d}.txt', 'w') as f:
            #         for (x,y,z), (r,g,b) in zip(locations.detach().cpu().numpy(), to_vis[:,i*3:i*3+3]):
            #             f.write('%d;%d;%d;%d;%d;%d\n' % (x,y,z,r,g,b))

            values = scn_values * (1-features[:, -1:]) + features[:, :-1] * features[:, -1:]

            # values = values * sigma + mu
            # values = kwargs['vertex_input'][0].transpose(0, 1)[:,:-1]
            # emb_mask = torch.round(features[..., -1]).bool()

            # ckpt = np.load('/home/lzq/lzy/NSVF/ReplicaGen/reg_multi_scene_nsvf_dim4in.npz',allow_pickle=True) # index not shifted
            # feats = torch.from_numpy(ckpt[f'center_to_vertex_00']).cuda().long()
            # points = torch.from_numpy(ckpt[f'center_point_00']).cuda()
            # values = torch.from_numpy(ckpt[f'vertex_feature_00']).cuda()

            # locations = ckpt[f'vertex_point_x10_00']
            # locations -= locations.min(axis=0)
            # locations = torch.from_numpy(locations).cuda().long()

            # values = kwargs['vertex_output'][0].transpose(0, 1)
            # values = nn.Tanh()(values)
            # values = self.reg_scn_model((locations, values, 1))
            # values = nn.Tanh()(values)

        else:
            print('using ground truth NSVF feature')
            values = kwargs['vertex_output'][0].transpose(0, 1)
            # values = kwargs['vertex_input'][0].transpose(0, 1)
            # values = values[:, :32]

        feats = kwargs['center_to_vertex'][0].transpose(0, 1)
        points = kwargs['center_pts'][0].transpose(0, 1)


        # ckpt = np.load('/home/lzq/lzy/NSVF/ReplicaGen/reg_multi_scene_nsvf_dim32.npz',allow_pickle=True) # index not shifted
        # feats = torch.from_numpy(ckpt[f'center_to_vertex_00']).to(feats.device).long()
        # points = torch.from_numpy(ckpt[f'center_point_00']).to(points.device)
        # values = torch.from_numpy(ckpt[f'vertex_feature_00']).to(values.device)
        

        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]

        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values,
            'dis_real_rgb': kwargs['colors']
            # 'emb_loss_ft': scn_values,
            # 'emb_loss_mask': emb_mask,
            # 'geo': pred_y,
            # 'batch_size': batch_size,
        }

        # if self.use_octree:
        #     flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
        #     if id is not None:
        #         flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
        #         flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
        #     encoder_states['voxel_octree_center_xyz'] = flatten_centers
        #     encoder_states['voxel_octree_children_idx'] = flatten_children

        return encoder_states




























