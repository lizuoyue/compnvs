# Prepare dataset
cd data_example
python3 make_nsvf_dataset.py
cd ..


# Geometry completion
cd geo_completion
mkdir mink_comp_pred
CUDA_VISIBLE_DEVICES=1 python3 test_comp_example.py
cd ..


# Feature encoding and for inputs and geometry merging
CUDA_VISIBLE_DEVICES=1 python3 make_pred_ft.py
CUDA_VISIBLE_DEVICES=1 python3 make_pred_geo.py


# # Texture completion and rendering
for i in $(seq -f "%03g" 1 21)
do
    DATA="ExampleScenesFused/$i"
    RES="192x256" # hxw 
    ARCH="mink_nsvf_rgba_sep_field_base"
    SUFFIX="v1_disc_pred"
    # SUFFIX="v1_pred"
    DATASET=${DATA}
    SAVE=${DATA}
    MODEL=$ARCH$SUFFIX

    # additional rendering args
    MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"use_octree":True}'
    MODELARGS=$(printf "$MODELTEMP" 256 0.0)

    # rendering with pre-defined testing trajectory
    CUDA_VISIBLE_DEVICES=1 python3 render.py ${DATASET} \
        --user-dir fairnr \
        --task single_object_rendering \
        --load-npz "npz_fts_predgeo/" \
        --path "ckpt/ckpt_nsvf.pt" \
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
