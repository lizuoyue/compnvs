# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
for i in $(seq -f "%03g" 1 21)
do
    # ln -s /home/lzq/lzy/NSVF/ARKitScenesTripletsNSVF/all/mink_nsvf_rgba_sep_field_basev1_disc_pred SilvanScenesFused/$i/mink_nsvf_rgba_sep_field_basev1_disc_pred
    # ln -s /home/lzq/lzy/NSVF/ARKitScenesTripletsNSVF/all/mink_nsvf_rgba_sep_field_basev1_pred SilvanScenesFused/$i/mink_nsvf_rgba_sep_field_basev1_pred
    DATA="SilvanScenesFused/$i"
    RES="192x256" # hxw 
    ARCH="mink_nsvf_rgba_sep_field_base"
    SUFFIX="v1_disc_pred"
    # SUFFIX="v1_pred"
    DATASET=/home/lzq/lzy/NSVF/${DATA}
    SAVE=/home/lzq/lzy/NSVF/${DATA}
    MODEL=$ARCH$SUFFIX
    MODEL_PATH=$SAVE/$MODEL/checkpoint19.pt # checkpoint_last.pt

    # additional rendering args
    MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"use_octree":True}'
    MODELARGS=$(printf "$MODELTEMP" 256 0.0)

    # rendering with pre-defined testing trajectory
    CUDA_VISIBLE_DEVICES=1 python3 render.py ${DATASET} \
        --user-dir fairnr \
        --task single_object_rendering \
        --load-npz "npz_fts_predgeo/" \
        --path ${MODEL_PATH} \
        --backbone-name "MinkUNet50" \
        --use-discriminator \
        --render-save-fps 24 \
        --render-camera-poses ${DATASET}/pose_video/"*" \
        --render-views "0..1" \
        --test-views "0..1" \
        --model-overrides $MODELARGS \
        --render-resolution $RES \
        --render-output ${SAVE}/$ARCH/output_video \
        --render-output-types "color" \
        --log-format "simple" \
        --render-combine-output 
done
