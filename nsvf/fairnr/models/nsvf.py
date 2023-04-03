# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
logger = logging.getLogger(__name__)

import cv2, math, time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.utils import item
from fairnr.data.geometry import compute_normal_map, fill_in
from fairnr.models.nerf import NeRFModel

def set_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad
    return

@register_model('nsvf')
class NSVFModel(NeRFModel):

    READER = 'image_reader'
    ENCODER = 'sparsevoxel_encoder'
    FIELD = 'radiance_field'
    RAYMARCHER = 'volume_rendering'

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--fine-num-sample-ratio', type=float, default=0,
            help='raito of samples compared to the first pass')
        parser.add_argument('--inverse-distance-coarse-sampling', type=str, 
            choices=['none', 'camera', 'origin'], default='none',
            help='if set, we do not sample points uniformly through voxels.')

    def intersecting(self, ray_start, ray_dir, encoder_states, **kwargs):
        S = ray_dir.size(0)
        ray_start, ray_dir, intersection_outputs, hits, _ = \
            super().intersecting(ray_start, ray_dir, encoder_states, **kwargs)

        if self.reader.no_sampling and self.training and (self.discriminator is None):  # sample points after ray-voxel intersection
            uv, size = kwargs['uv'], kwargs['size']
            mask = hits.reshape(*uv.size()[:2], uv.size(-1))

            # sample rays based on voxel intersections
            sampled_uv, sampled_masks = self.reader.sample_pixels(
                uv, size, mask=mask, return_mask=True)
            sampled_masks = sampled_masks.reshape(uv.size(0), -1).bool()
            hits, sampled_masks = hits[sampled_masks].reshape(S, -1), sampled_masks.unsqueeze(-1)
            intersection_outputs = {name: outs[sampled_masks.expand_as(outs)].reshape(S, -1, outs.size(-1)) 
                for name, outs in intersection_outputs.items()}
            ray_start = ray_start[sampled_masks.expand_as(ray_start)].reshape(S, -1, 3)
            ray_dir = ray_dir[sampled_masks.expand_as(ray_dir)].reshape(S, -1, 3)
        
        else:
            sampled_uv = None
        
        min_depth = intersection_outputs['min_depth']
        max_depth = intersection_outputs['max_depth']
        pts_idx = intersection_outputs['intersected_voxel_idx']
        dists = (max_depth - min_depth).masked_fill(pts_idx.eq(-1), 0)
        intersection_outputs['probs'] = dists / dists.sum(dim=-1, keepdim=True)
        if getattr(self.args, "fixed_num_samples", 0) > 0:
            intersection_outputs['steps'] = intersection_outputs['min_depth'].new_ones(
                *intersection_outputs['min_depth'].size()[:-1], 1) * self.args.fixed_num_samples
        else:
            intersection_outputs['steps'] = dists.sum(-1) / self.encoder.step_size
        return ray_start, ray_dir, intersection_outputs, hits, sampled_uv
        
    def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False):
        samples, all_results = super().raymarching(ray_start, ray_dir, intersection_outputs, encoder_states, fine)
        all_results['vertex_emb'] = encoder_states['voxel_vertex_emb']
        if 'geo' in encoder_states:
            all_results['geo'] = encoder_states['geo']
        if 'emb_loss_mask' in encoder_states:
            all_results['emb_loss_mask'] = encoder_states['emb_loss_mask']
            all_results['emb_loss_ft'] = encoder_states['emb_loss_ft']
        if 'dis_real_rgb' in encoder_states:
            all_results['dis_real_rgb'] = encoder_states['dis_real_rgb']
        all_results['voxel_edges'] = self.encoder.get_edge(ray_start, ray_dir, samples, encoder_states)
        all_results['voxel_depth'] = samples['sampled_point_depth'][:, 0]
        return samples, all_results

    def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
        intersection_outputs = super().prepare_hierarchical_sampling(intersection_outputs, samples, all_results)
        if getattr(self.args, "fine_num_sample_ratio", 0) > 0:
            intersection_outputs['steps'] = samples['sampled_point_voxel_idx'].ne(-1).sum(-1).float() * self.args.fine_num_sample_ratio
        return intersection_outputs

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes, kwargs):
         # we need fill_in for NSVF for background
        S, V, P = sizes
        fullsize = S * V * P
        
        all_results['missed'] = fill_in((fullsize, ), hits, all_results['missed'], 1.0).view(S, V, P)
        all_results['colors'] = fill_in((fullsize, 3), hits, all_results['colors'], 0.0).view(S, V, P, 3)
        all_results['depths'] = fill_in((fullsize, ), hits, all_results['depths'], 0.0).view(S, V, P)
        
        BG_DEPTH = self.field.bg_color.depth
        bg_color = self.field.bg_color(all_results['colors'])
        all_results['colors'] += all_results['missed'].unsqueeze(-1) * bg_color.reshape(fullsize, 3).view(S, V, P, 3)
        all_results['depths'] += all_results['missed'] * BG_DEPTH

        if self.discriminator is not None:

            img_h, img_w = kwargs['size'][0,0,:2].int().cpu().numpy()

            nft = all_results['features'].shape[-1] # emb + 3

            all_results['features'] = fill_in((fullsize, nft), hits, all_results['features'], 0.0).view(S, V, P, nft)
            all_results['features'][..., :3] = all_results['features'][..., :3] * 0 + all_results['colors']

            pred_rgb_lowres = all_results['colors'].reshape(S*V, img_h//2, img_w//2, 3).permute(0,3,1,2).contiguous()
            pred_fts_lowres = all_results['features'].reshape(S*V, img_h//2, img_w//2, nft).permute(0,3,1,2).contiguous()
            pred_rgb = self.upsampler(pred_fts_lowres)
            pred_rgb += torch.nn.functional.interpolate(pred_rgb_lowres, scale_factor=2, mode='bicubic')
            all_results['colors_low'] = all_results['colors']
            all_results['colors'] = pred_rgb.reshape(S, V, 3, P*4).permute(0,1,3,2).contiguous()

            set_requires_grad(self.discriminator, False)
            # fix D and let G generate as real as possible
            pred_rgb = torch.clamp(pred_rgb, -1, 1)
            all_results['dis_gen_real'] = self.discriminator(pred_rgb)            

            # from PIL import Image
            # Image.fromarray(np.round(127.5*pred_rgb_lowres.detach().cpu().numpy()[0].transpose([1,2,0])+127.5).astype(np.uint8)).save('pred_rgb0.png')
            # Image.fromarray(np.round(127.5*pred_rgb.detach().cpu().numpy()[0].transpose([1,2,0])+127.5).astype(np.uint8)).save('pred_rgb.png')
            if self.training:
                print('train discriminator')
                set_requires_grad(self.discriminator, True)
            else:
                print('val discriminator no grad')
                set_requires_grad(self.discriminator, False)
            # detach G and train D as disciminative as possible
            # real_rgb = all_results['dis_real_rgb'].reshape(S*V, img_h, img_w, 3).permute(0,3,1,2).contiguous()
            real_rgb = all_results['dis_real_rgb'].reshape(S*V, img_h, img_w, 3).permute(0,3,1,2).contiguous()
            all_results['dis_dis_fake'] = self.discriminator(pred_rgb.detach())
            all_results['dis_dis_real'] = self.discriminator(real_rgb)
            # Image.fromarray(np.round(+127.5*real_rgb.detach().cpu().numpy()[0].transpose([1,2,0])+127.5).astype(np.uint8)).save('real_rgb.png')
            # quit()


        if 'normal' in all_results:
            all_results['normal'] = fill_in((fullsize, 3), hits, all_results['normal'], 0.0).view(S, V, P, 3)
        if 'voxel_depth' in all_results:
            all_results['voxel_depth'] = fill_in((fullsize, ), hits, all_results['voxel_depth'], BG_DEPTH).view(S, V, P)
        if 'voxel_edges' in all_results:
            all_results['voxel_edges'] = fill_in((fullsize, 3), hits, all_results['voxel_edges'], 1.0).view(S, V, P, 3)
        if 'feat_n2' in all_results:
            all_results['feat_n2'] = fill_in((fullsize,), hits, all_results['feat_n2'], 0.0).view(S, V, P)
        return all_results

    def add_other_logs(self, all_results):
        return {'voxs_log': item(self.encoder.voxel_size),
                'stps_log': item(self.encoder.step_size),
                'nvox_log': item(self.encoder.num_voxels)}

    def _visualize(self, images, sample, output, state, **kwargs):
        img_id, shape, view, width, name = state
        images = super()._visualize(images, sample, output, state, **kwargs)
        if 'voxel_edges' in output and output['voxel_edges'] is not None:
            # voxel hitting visualization
            images['{}_voxel/{}:HWC'.format(name, img_id)] = {
                'img': output['voxel_edges'][shape, view].float(), 
                'min_val': 0, 
                'max_val': 1,
                'weight':
                    compute_normal_map(
                        sample['ray_start'][shape, view].float(),
                        sample['ray_dir'][shape, view].float(),
                        output['voxel_depth'][shape, view].float(),
                        sample['extrinsics'][shape, view].float().inverse(),
                        width, proj=True)
                }
        
        if 'feat_n2' in output and output['feat_n2'] is not None:
            images['{}_featn2/{}:HWC'.format(name, img_id)] = {
                'img': output['feat_n2'][shape, view].float(),
                'min_val': 0,
                'max_val': 1
            }
        return images
    
    @torch.no_grad()
    def prune_voxels(self, th=0.5, train_stats=False):
        self.encoder.pruning(self.field, th, train_stats=train_stats)
        self.clean_caches()

    @torch.no_grad()
    def split_voxels(self):
        logger.info("half the global voxel size {:.4f} -> {:.4f}".format(
            self.encoder.voxel_size.item(), self.encoder.voxel_size.item() * .5))
        self.encoder.splitting()
        self.encoder.voxel_size *= .5
        self.encoder.max_hits *= 1.5
        self.clean_caches()

    @torch.no_grad()
    def reduce_stepsize(self):
        logger.info("reduce the raymarching step size {:.4f} -> {:.4f}".format(
            self.encoder.step_size.item(), self.encoder.step_size.item() * .5))
        self.encoder.step_size *= .5

    def clean_caches(self, reset=False):
        self.encoder.clean_runtime_caches()
        if reset:
            self.encoder.reset_runtime_caches()

@register_model_architecture("nsvf", "nsvf_base")
def base_architecture(args):
    # parameter needs to be changed
    args.voxel_size = getattr(args, "voxel_size", None)
    args.max_hits = getattr(args, "max_hits", 60)
    args.raymarching_stepsize = getattr(args, "raymarching_stepsize", 0.01)
    args.raymarching_stepsize_ratio = getattr(args, "raymarching_stepsize_ratio", 0.0)
    
    # encoder default parameter
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 32)
    args.voxel_path = getattr(args, "voxel_path", None)
    args.initial_boundingbox = getattr(args, "initial_boundingbox", None)

    # field
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:6:32")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    args.feature_embed_dim = getattr(args, "feature_embed_dim", 256)
    args.density_embed_dim = getattr(args, "density_embed_dim", 128)
    args.texture_embed_dim = getattr(args, "texture_embed_dim", 256)

    # API Update: fix the number of layers
    args.feature_layers = getattr(args, "feature_layers", 1)
    args.texture_layers = getattr(args, "texture_layers", 3)
    
    args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
    args.background_depth = getattr(args, "background_depth", 5.0)
    
    # raymarcher
    args.discrete_regularization = getattr(args, "discrete_regularization", False)
    args.deterministic_step = getattr(args, "deterministic_step", False)
    args.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0)
    args.use_octree = getattr(args, "use_octree", False)

    # reader
    args.pixel_per_view = getattr(args, "pixel_per_view", 2048)
    args.sampling_on_mask = getattr(args, "sampling_on_mask", 0.0)
    args.sampling_at_center = getattr(args, "sampling_at_center", 1.0)
    args.sampling_on_bbox = getattr(args, "sampling_on_bbox", False)
    args.sampling_patch_size = getattr(args, "sampling_patch_size", 1)
    args.sampling_skipping_size = getattr(args, "sampling_skipping_size", 1)

    # others
    args.chunk_size = getattr(args, "chunk_size", 64)
    args.valid_chunk_size = getattr(args, "valid_chunk_size", 64)


@register_model_architecture("nsvf", "nsvf_xyz")
def nerf2_architecture(args):
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 0)
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:10, ray:4")
    base_architecture(args)


@register_model_architecture("nsvf", "nsvf_nerf")
def nerf_style_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    args.feature_layers = getattr(args, "feature_layers", 6)
    args.texture_layers = getattr(args, "texture_layers", 0)
    args.feature_field_skip_connect = getattr(args, "feature_field_skip_connect", 3)
    args.no_layernorm_mlp = getattr(args, "no_layernorm_mlp", True)
    nerf2_architecture(args)

@register_model_architecture("nsvf", "nsvf_nerf_nov")
def nerf_noview_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256")
    nerf_style_architecture(args)

@register_model_architecture("nsvf", "nsvf_xyzn")
def nerf3_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:10, normal:4, ray:4")
    nerf2_architecture(args)


@register_model_architecture("nsvf", "nsvf_xyz_nope")
def nerf3nope_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:0:3, sigma:0:1, ray:4")
    nerf2_architecture(args)

@register_model_architecture("nsvf", "nsvf_xyzn_old")
def nerfold_architecture(args):
    args.feature_layers = getattr(args, "feature_layers", 6)
    args.feature_field_skip_connect = getattr(args, "feature_field_skip_connect", 3)
    args.no_layernorm_mlp = getattr(args, "no_layernorm_mlp", True)
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, normal:0:3, sigma:0:1, ray:4")
    nerf2_architecture(args)

@register_model_architecture("nsvf", "nsvf_xyzn_nope")
def nerf2nope_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:0:3, normal:0:3, sigma:0:1, ray:4")
    nerf2_architecture(args)

@register_model_architecture("nsvf", "nsvf_xyzn_noz")
def nerf3noz_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "pos:10, normal:4, ray:4")
    nerf2_architecture(args)

@register_model_architecture("nsvf", "nsvf_embn")
def nerf4_architecture(args):
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:6:32")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, normal:4, ray:4")
    base_architecture(args)


@register_model_architecture("nsvf", "nsvf_emb0")
def nerf5_architecture(args):
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 384)
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:0:384")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    base_architecture(args)


@register_model('disco_nsvf')
class DiscoNSVFModel(NSVFModel):

    FIELD = "disentangled_radiance_field"


@register_model_architecture("disco_nsvf", "disco_nsvf")
def disco_nsvf_architecture(args):
    args.compressed_light_dim = getattr(args, "compressed_light_dim", 64)
    nerf3_architecture(args)


@register_model('multi_disco_nsvf')
class mDiscoNSVFModel(NSVFModel):

    ENCODER = "multi_sparsevoxel_encoder"
    FIELD = "disentangled_radiance_field"


@register_model_architecture("multi_disco_nsvf", "multi_disco_nsvf")
def mdisco_nsvf_architecture(args):
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10, context:0:256")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:10, normal:4, ray:4, context:0:256")
    disco_nsvf_architecture(args)


@register_model('sdf_nsvf')
class SDFNSVFModel(NSVFModel):

    FIELD = "sdf_radiance_field"


@register_model_architecture("sdf_nsvf", "sdf_nsvf")
def sdf_nsvf_architecture(args):
    args.feature_layers = getattr(args, "feature_layers", 6)
    args.feature_field_skip_connect = getattr(args, "feature_field_skip_connect", 3)
    args.no_layernorm_mlp = getattr(args, "no_layernorm_mlp", True)
    nerf2nope_architecture(args)


@register_model('sdf_nsvf_sfx')
class SDFSFXNSVFModel(SDFNSVFModel):

    FIELD = "sdf_radiance_field"
    RAYMARCHER = "surface_volume_rendering"


@register_model_architecture("sdf_nsvf_sfx", "sdf_nsvf_sfx")
def sdf_nsvfsfx_architecture(args):
    sdf_nsvf_architecture(args)




@register_model('scn_nsvf')
class SCNNSVFModel(NSVFModel):
    ENCODER = "scn_sparsevoxel_encoder"

@register_model_architecture("scn_nsvf", "scn_nsvf_base")
def scn_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('gated_scn_nsvf')
class GatedSCNNSVFModel(NSVFModel):
    ENCODER = "gated_scn_sparsevoxel_encoder"

@register_model_architecture("gated_scn_nsvf", "gated_scn_nsvf_base")
def gated_scn_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('advanced_scn_nsvf')
class AdvancedSCNNSVFModel(NSVFModel):
    ENCODER = "advanced_scn_sparsevoxel_encoder"

@register_model_architecture("advanced_scn_nsvf", "advanced_scn_nsvf_base")
def advanced_scn_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('multi_scene_nsvf')
class MultiSceneNSVFModel(NSVFModel):
    ENCODER = "multi_sparsevoxel_encoder"

@register_model_architecture("multi_scene_nsvf", "multi_scene_nsvf_base")
def multi_scene_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('multi_scene_nsvf_rgba_field')
class MultiSceneNSVFRGBAFieldModel(NSVFModel):
    ENCODER = "multi_sparsevoxel_encoder"
    FIELD = "rgba_radiance_field"

@register_model_architecture("multi_scene_nsvf_rgba_field", "multi_scene_nsvf_rgba_field_base")
def multi_scene_nsvf_rgba_field_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('reg_multi_scene_nsvf')
class RegMultiSceneNSVFModel(NSVFModel):
    ENCODER = "reg_multi_sparsevoxel_encoder"

@register_model_architecture("reg_multi_scene_nsvf", "reg_multi_scene_nsvf_base")
def reg_multi_scene_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('dict_multi_scene_nsvf')
class DictMultiSceneNSVFModel(NSVFModel):
    ENCODER = "dict_multi_sparsevoxel_encoder"

@register_model_architecture("dict_multi_scene_nsvf", "dict_multi_scene_nsvf_base")
def dict_multi_scene_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('fcn_scn_nsvf')
class FCNSCNNSVFModel(NSVFModel):
    ENCODER = "fcn_scn_sparsevoxel_encoder"

@register_model_architecture("fcn_scn_nsvf", "fcn_scn_nsvf_base")
def fcn_scn_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('geo_scn_nsvf')
class FCNSCNNSVFModel(NSVFModel):
    ENCODER = "geo_scn_sparsevoxel_encoder"

@register_model_architecture("geo_scn_nsvf", "geo_scn_nsvf_base")
def fcn_scn_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('mink_nsvf')
class MinkNSVFModel(NSVFModel):
    ENCODER = "mink_sparsevoxel_encoder"

@register_model_architecture("mink_nsvf", "mink_nsvf_base")
def mink_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('mink_nsvf_rgba_sep_field')
class MinkNSVFModel(NSVFModel):
    ENCODER = "mink_sparsevoxel_encoder"
    FIELD = "rgba_sep_radiance_field"

@register_model_architecture("mink_nsvf_rgba_sep_field", "mink_nsvf_rgba_sep_field_base")
def mink_nsvf_rgba_sep_field_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('minkpc_nsvf')
class MinkPCNSVFModel(NSVFModel):
    ENCODER = "minkpc_sparsevoxel_encoder"

@register_model_architecture("minkpc_nsvf", "minkpc_nsvf_base")
def minkpc_nsvf_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('minkpc_nsvf_rgba_field')
class MinkPCNSVFRGBAFieldModel(NSVFModel):
    ENCODER = "minkpc_sparsevoxel_encoder"
    FIELD = "rgba_radiance_field"

@register_model_architecture("minkpc_nsvf_rgba_field", "minkpc_nsvf_rgba_field_base")
def minkpc_nsvf_rgba_field_base_architecture(args):
    base_architecture(args)

#################################################################

@register_model('minkpc_nsvf_rgbasep_field')
class MinkPCNSVFRGBASepFieldModel(NSVFModel):
    ENCODER = "minkpc_sparsevoxel_encoder"
    FIELD = "rgba_sep_radiance_field"

@register_model_architecture("minkpc_nsvf_rgbasep_field", "minkpc_nsvf_rgbasep_field_base")
def minkpc_nsvf_rgbasep_field_base_architecture(args):
    base_architecture(args)
