# CompNVS

## Requested dependencies
* PyTorch
* MinkowskiEngine
* NSVF

## Inference
1. You may first create checkpoint folders by runing in the path `./compnvs` and also build.
```
mkdir geo_completion/ckpt
mkdir ckpt
python3 setup.py build_ext --inplace
```
2. Place the whole `data_for_zuoyue` folder inside the `data_example` folder.
3. Place `ckpt_geo_comp.pt` in the `geo_completion/ckpt` folder.
4. Place `ckpt_encoder.pt` and `ckpt_nsvf.pt` in the `ckpt` folder.
5. Then in the path `./compnvs`, run `sh run.sh`.
6. You will find results in the folders `compnvs/ExampleScenesFused/*/mink_nsvf_rgba_sep_field_base/output_video/color/*.png`.