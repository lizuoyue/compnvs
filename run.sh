# # Prepare dataset
# cd data_example
# python3 make_nsvf_dataset.py
# cd ..


# # Geometry completion
# cd geo_completion
# mkdir mink_comp_pred
# CUDA_VISIBLE_DEVICES=1 python3 test_comp_example.py
# cd ..


# # Feature encoding and for inputs and geometry merging
# CUDA_VISIBLE_DEVICES=1 python3 make_pred_ft.py
# CUDA_VISIBLE_DEVICES=1 python3 make_pred_geo.py


# # Texture completion and rendering
