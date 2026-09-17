<p align="center">
  <h1 align="center">DepthSplat: Connecting Gaussian Splatting and Depth</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://pengsongyou.github.io/">Songyou Peng</a>
    ·
    <a href="https://fangjinhuawang.github.io/">Fangjinhua Wang</a>
    ·
    <a href="https://hermannblum.net/">Hermann Blum</a>
    ·
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">Daniel Barath</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>
    ·
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
  </p>
  <h3 align="center">CVPR 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2410.13862">Paper</a> | <a href="https://haofeixu.github.io/depthsplat/">Project Page</a> | <a href="https://huggingface.co/haofeixu/depthsplat">Models</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/depthsplat/assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>


<p align="center">
<strong>DepthSplat enables cross-task interactions between Gaussian splatting and depth estimation.</strong> <br>
Left: Better depth leads to improved novel view synthesis with Gaussian splatting. <br>
Right: Unsupervised depth pre-training with Gaussian splatting leads to reduced depth prediction error.
</p>


## Updates

- 2026-03-31: Check out [ReSplat](https://haofeixu.github.io/resplat/) for more compact and robust feed-forward Gaussian splatting models!

- 2025-03-27: We simplified our model architecture while preparing the CVPR camera-ready version. The models have been re-trained, and the [paper](https://arxiv.org/abs/2410.13862) has been updated accordingly. [The new models](MODEL_ZOO.md) are now simpler, faster, and perform as well as or better than the previous version.

## Installation

Our code is developed using PyTorch 2.4.0, CUDA 12.4, and Python 3.10. 

We recommend setting up a virtual environment using either [conda](https://docs.anaconda.com/miniconda/) or [venv](https://docs.python.org/3/library/venv.html) before installation:

```bash
# conda
conda create -y -n depthsplat python=3.10
conda activate depthsplat

# or venv
# python -m venv /path/to/venv/depthsplat
# source /path/to/venv/depthsplat/bin/activate

# installation
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Model Zoo

Our pre-trained models are hosted on [Hugging Face 🤗](https://huggingface.co/haofeixu/depthsplat).

Model details can be found at [MODEL_ZOO.md](MODEL_ZOO.md).


## Camera Conventions

The camera intrinsic matrices are normalized, with the first row divided by the image width and the second row divided by the image height.

The camera extrinsic matrices follow the OpenCV convention for camera-to-world transformation (+X right, +Y down, +Z pointing into the screen).

## Datasets

For dataset preparation, please refer to [DATASETS.md](DATASETS.md).



## Gaussian Splatting


### Useful configs


<!-- <details>
<summary>Click to expand</summary> -->



- `dataset.test_chunk_interval=1`: Running on the full test set can be time-consuming due to the large number of scenes. You can run on a fraction of the test set for debugging or validation purposes. For example, setting `dataset.test_chunk_interval=10` will evaluate on 1/10 of the full test set.
- `output_dir=outputs/depthsplat`: Directory to save the results.
- `test.save_image=true`: Save the rendered images.
- `test.save_gt_image=true`: Save the ground truth (GT) images.
- `test.save_input_images=true`: Save the input images.
- `test.save_depth=true`: Save the predicted depths.
- `test.save_depth_concat_img=true`: Save the concatenated images and depths.
- `test.save_depth_npy=true`: Save the raw depth predictions in `.npy`.
- `test.save_gaussian=true`: Save the reconstructed Gaussians in `.ply` files, which can be viewed using online viewers like [SuperSplat](https://superspl.at/editor), [Antimatter15](https://antimatter15.com/splat/), etc.

<!-- </details> -->


### Rendering Video

DepthSplat enables feed-forward reconstruction from 12 input views (512x960 resolutions) in 0.6 seconds on a single A100 GPU.

#### RealEstate10K


<details>
<summary>6 input views at 512x960 resolutions: click to expand the script</summary>

- A preprocessed subset is provided to quickly run inference with our model, please refer to the details in [DATASETS.md](DATASETS.md).

```
# render video on re10k (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
dataset.roots=[datasets/re10k_720p] \
dataset.image_shape=[512,960] \
dataset.ori_image_shape=[720,1280] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/re10k_ctx_6v_video.json \
test.save_video=true \
test.compute_scores=false \
test.render_chunk_size=10 \
output_dir=outputs/depthsplat-re10k-512x960
```

</details>



https://github.com/user-attachments/assets/3f228a3f-8d54-4a90-9db4-ff0874150883



<details>
<summary>2 input views at 256x256 resolutions:</summary>


```
# render video on re10k (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=100 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.compute_scores=false
output_dir=outputs/depthsplat-re10k
```

</details>



#### DL3DV

<details>
<summary>12 input views at 512x960 resolutions:</summary>

- A preprocessed subset is provided to quickly run inference with our model, please refer to the details in [DATASETS.md](DATASETS.md).

- Tip: use `test.stablize_camera=true` to stablize the camera trajectory.

```
# render video on dl3dv (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
dataset.roots=[datasets/dl3dv_960p] \
dataset.image_shape=[512,960] \
dataset.ori_image_shape=[540,960] \
model.encoder.upsample_factor=8 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=12 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_100_ctx_12v_video.json \
test.save_video=true \
test.stablize_camera=true \
test.compute_scores=false \
test.render_chunk_size=10 \
output_dir=outputs/depthsplat-dl3dv-512x960
```

</details>




https://github.com/user-attachments/assets/ea6d3b9c-af80-43e6-9a12-36c67e874366




### Evaluation



#### RealEstate10K

<details>
<summary>Evaluation scripts (small, base, and large models)</summary>

Please note that the numbers may differ slightly from those reported in the paper, as the models have been re-trained.

- To evalute the large model:
```
# Table 1 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
mode=test \
dataset/view_sampler=evaluation
```

<!-- </details>

<details>
<summary><b>To evaluate the base model, use:</b></summary> -->

- To evaluate the base model:

```
# Table 1 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth \
mode=test \
dataset/view_sampler=evaluation
```

<!-- </details>


<details>
<summary><b>To evaluate the small model, use:</b></summary> -->

- To evaluate the small model: 

```
# Table 1 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
mode=test \
dataset/view_sampler=evaluation
```
</details>


#### DL3DV

<details>
<summary>Evaluation scripts (6, 4, 2 input views, and zero-shot generalization)</summary>

- 6 input views:

```
# Table 7 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
```


<!-- <details>
<summary><b>4 input views:</b></summary> -->

- 4 input views:

```
# Table 7 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=4 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
```

<!-- </details>


<details>
<summary><b>2 input views:</b></summary> -->

- 2 input views:

```
# Table 7 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_2v_video_0_50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
```

<!-- </details>


<details>
<summary><b>Zero-shot generalization from RealEstate10K to DL3DV:</b></summary> -->

- Zero-shot generalization from RealEstate10K to DL3DV:

```
# Table 8 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth
```


</details>




#### ACID


<details>
<summary>Evaluation scripts (zero-shot generalization)</summary>

- Zero-shot generalization from RealEstate10K to ACID:

```
# Table 8 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[datasets/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth
```


</details>


### Training

- Before training, you need to download the pre-trained [UniMatch](https://github.com/autonomousvision/unimatch) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) weights, and set up your [wandb account](config/main.yaml) (in particular, by setting `wandb.entity=YOUR_ACCOUNT`) for logging.

```
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P pretrained
```

- By default, we train our models using four GH200 GPUs (96GB VRAM each). However, this is not a strict requirement—our model can be trained on different GPUs as well. For example, we have verified that configurations such as four RTX 4090 GPUs (24GB VRAM each) or a single A100 GPU (80GB VRAM) can achieve very similar results, with a PSNR difference of at most 0.1 dB. Just ensure that the total number of training samples, calculated as (number of GPUs &times; `data_loader.train.batch_size` &times; `trainer.max_steps`), remains the same. Check out the scripts [scripts/re10k_depthsplat_train.sh](scripts/re10k_depthsplat_train.sh) and [scripts/dl3dv_depthsplat_train.sh](scripts/dl3dv_depthsplat_train.sh) for details.



## Depth Prediction

We fine-tune our Gaussian Splatting pre-trained depth model using ground-truth depth supervision. The depth models are trained with a randomly selected number of input images (ranging from 2 to 8) and can be used for depth prediction from multi-view posed images. For more details, please refer to [scripts/inference_depth.sh](scripts/inference_depth.sh).


<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/depthsplat/assets/depth/img_depth_c31a5a509ab9c526.png" alt="Logo" width="100%">
  </a>
</p>


## nuScenes Wide-View Inference

`scripts/inference_nuscenes_wide.py` is the nuScenes entry point for the
standalone single-frame inference path. It reconstructs Gaussians from cameras
`5, 4, 3` of a frame and renders a horizontally widened image from camera `5`,
without going through the Lightning `ModelWrapper.test_step` (which assumes a
ground-truth target view). By default it processes every scene in the scene list
and every valid frame; extrinsics come from the static camera-to-ego rig (see
[Extrinsics source](#extrinsics-source) below).

The implementation is shared: `scripts/wide_inference_core.py` holds all the
logic (helpers, `DatasetPreset` table, CLI), and the per-dataset entry points are
thin wrappers that pin a preset:

| Single-frame entry point | Pinned `--dataset` |
| --- | --- |
| `scripts/inference_nuscenes_wide.py` | `nuscenes` |
| `scripts/inference_lyft1920_wide.py` | `lyft1920` |
| `scripts/inference_lyft1224_wide.py` | `lyft1224` |
| `scripts/inference_ddad_wide.py` | `ddad` |

`--dataset` remains available on every wrapper as an explicit override, so
`python scripts/inference_lyft1920_wide.py ...` is equivalent to
`python scripts/wide_inference_core.py --dataset lyft1920 ...`. The examples
below use the nuScenes wrapper; swap the script name for another dataset.

### Usage

```bash
# Defaults: 256x448 model, every scene in the scene list, every valid frame,
# no input images saved.
python scripts/inference_nuscenes_wide.py \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide

# Select the model preset and the input resize size explicitly (these are the
# same as the defaults, shown for clarity):
python scripts/inference_nuscenes_wide.py \
  --model 256x448 \
  --input-size 256x448 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide

# The larger local checkpoint, rendered at its native 448x768 input:
python scripts/inference_nuscenes_wide.py \
  --model 448x768 \
  --input-size 448x768 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_448x768

# Render a single scene/frame instead:
python scripts/inference_nuscenes_wide.py \
  --model 256x448 \
  --input-size 256x448 \
  --scene 037 \
  --frame 0 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --width-factor 3 \
  --output-dir outputs/nuscenes_wide

# Ego-car masking is on by default (cameras 4 and 3 masked, render camera 5
# preserved); mask the render view too for a diagnostic, or disable entirely:
python scripts/inference_nuscenes_wide.py \
  --mask-render-view \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_all_masks_check
python scripts/inference_nuscenes_wide.py \
  --disable-car-mask \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide
```

- The model defaults to the `256x448` preset (whose default input is therefore
  `256x448`; the `448x768` model is available via `--model 448x768`); context
  cameras default to `5,4,3`, the render camera to `5`, and the output width
  factor to `3` (`256x448 -> 256x1344`, `448x768 -> 448x2304`).
- `--model` selects the **architecture preset**: it fixes `vitb`,
  `num_scales`, `upsample_factor`, `lowest_feature_resolution` and
  `gaussian_scale_max`, and chooses the matching default checkpoint. It is
  independent of the input size.
- `--input-size HxW` sets the **input resize size** (e.g. `448x768`, also
  accepted as `448,768` or `448 768`). It defaults to the `--model` preset size
  and is rounded down to a multiple of the effective patch size (`64` for the
  `dl3dv` experiment). `--height` / `--width` still work and override the
  corresponding dimension of `--input-size` (highest priority).
- `--resolution` is kept as a **legacy alias for `--model`** (same option, same
  values), so existing commands like `--resolution 448x768` keep working.
- Scene ids default to `datasets/nuscenes/processed_10Hz/trainval2/nuScenes_Val2.txt`
  (`--scene-list` overrides this; `--data-root` sets the scene folder root and,
  while the default list is used, is where `<data-root>/nuScenes_Val2.txt` is
  read from). Every scene in the list and **every valid frame per scene** are
  processed by default; `--max-frames N` limits frames per scene (`-1`, the
  default, means all), and `--scene` / `--frame` select a single scene/frame.
- Input images are **not** saved by default; pass `--save-inputs` to also write
  the resized context images.
- Extrinsics default to the static camera-to-ego rig
  (`--extrinsics-source cam2ego`); see [Extrinsics source](#extrinsics-source)
  for the `per_frame` A/B alternative.
- `--dry-run` validates and loads a frame (images, intrinsics, extrinsics)
  without building the model, which is useful on machines without the CUDA
  rasterizer extension.
- Output is written to `<output-dir>/<scene>/rgb/{frame}_{render_cam}_wide.jpg`
  at JPEG quality 95. With `--save-inputs` the resized context images are also
  written to `<output-dir>/<scene>/inputs/{frame}_{cam}.jpg`.

### Extrinsics source

Each context camera needs an OpenCV **camera-to-world** `4x4` matrix, and the
processed scene folders ship two candidates:

| File | Meaning |
| --- | --- |
| `cam2ego_extrinsics/{cam}.txt` | static per-camera rig transform (camera → ego) |
| `extrinsics/{frame}_{cam}.txt` | per-frame global camera-to-world (ego pose baked in) |

`--extrinsics-source` selects which one is used:

- `cam2ego` (**default**) reads `cam2ego_extrinsics/{cam}.txt` and uses it
  directly as OpenCV camera-to-world. In this processed data the per-frame ego
  frame **is** the world, so the rig is frame-independent: every frame of a scene
  reconstructs in the same ego/world frame, which is what independent
  single-frame processing should do.
- `per_frame` reads the per-frame `extrinsics/{frame}_{cam}.txt` global matrix
  instead. That source is not always exactly `ego_pose @ cam2ego` in this
  processed data (for example, scene `037` frame `008`: the relative
  camera-3-to-camera-5 transform implied by `per_frame` differs from the
  `ego_pose @ cam2ego` composition by ~`0.7 m`), so it can yield a
  frame-dependent rig and a slightly different reconstruction. It is provided
  **only for explicit A/B comparison**, never as a silent fallback, and the
  script prints a warning when it is selected.

If the selected file is missing the run fails with an error naming the missing
path, the active source, and the flag to switch to the other source; the two
sources are never mixed implicitly.

### Ego-car masking

Single-frame inference masks the ego-car pixels by default (enabled, same
policy and implementation as the multi-frame path). Each selected camera has a
static mask under `--car-mask-root` (default
`datasets/nuscenes/processed_10Hz/nuscenes_mask`), mapped by nuScenes camera id:

| Camera id | Camera | Mask file |
| --- | --- | --- |
| 0 | `CAM_FRONT` | `CAM_FRONT_mask.png` |
| 1 | `CAM_FRONT_LEFT` | `CAM_FRONT_LEFT_mask.png` |
| 2 | `CAM_FRONT_RIGHT` | `CAM_FRONT_RIGHT_mask.png` |
| 3 | `CAM_BACK_LEFT` | `CAM_BACK_LEFT_mask.png` |
| 4 | `CAM_BACK_RIGHT` | `CAM_BACK_RIGHT_mask.png` |
| 5 | `CAM_BACK` | `CAM_BACK_mask.png` |

- **Polarity**: black (`< 128`) pixels are removed; white (`>= 128`) pixels are
  kept.
- **Transform**: each mask is loaded as PIL `L` and transformed with *exactly*
  the images' resize/crop plan — a NEAREST resize to the scaled size, then the
  same centre crop. It is **not** plain-resized straight to the destination, so
  mask pixels stay aligned with the resized/cropped images. The mask's source
  resolution must equal the source images' resolution.
- **Which views (default, `all_except_render_view`)**: each camera's mask is
  applied to its own context view, **except the render camera which is fully
  preserved**. For the default cameras `5,4,3` with render camera `5`, cameras
  **4 and 3** are masked and camera **5** is kept whole.
- **Diagnostic (`--mask-render-view`, `all_views`)**: additionally applies each
  camera's mask to the render view (camera 5), so the preset ego-car region is
  removed there too.
- **Disable (`--disable-car-mask`)**: highest precedence; keeps every Gaussian
  from every view and overrides both policies above.
- **Precedence**: `--disable-car-mask` > `--mask-render-view` > default.
- **Failure mode**: a missing required mask for any selected camera is a hard
  error (there is no silent all-ones fallback).
- Pruning drops the affected Gaussians from `means`, `covariances`, `harmonics`
  and `opacities` identically (rather than only zeroing opacity) after the
  encoder and before the decoder. This is the same shared implementation used by
  the multi-frame path.

The current masks only have black pixels in `CAM_BACK_mask.png` (camera 5);
cameras 3 and 4 are white placeholders, so the default single-frame run reports
them as "applied but 0 removed" today.

### Checkpoints and input resolutions

The `--model` preset selects the matching locally shipped base checkpoint and
its architecture; `--input-size` chooses the input resize independently:

| `--model` (alias `--resolution`) | Default checkpoint | Default input | Output (factor 3) | `gaussian_scale_max` |
| --- | --- | --- | --- | --- |
| `256x448` (default) | `pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth` | 256x448 | 256x1344 | 3.0 |
| `448x768` | `pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth` | 448x768 | 448x2304 | 0.1 |

Both are 117M `vitb` models, so the script sets
`monodepth_vit_type=vitb`, `num_scales=2`, `upsample_factor=4` and
`lowest_feature_resolution=8`. `gaussian_scale_max` is part of the trained
architecture rather than a free rendering knob, so it is carried on the preset
and is always composed into the encoder config (the `448x768` model was trained
with `0.1`, the `256x448` dl3dv model with the `3.0` default); use
`--gaussian-scale-max` to override it. Use `--checkpoint` to load another
compatible checkpoint; the encoder architecture is loaded strictly, so a
mismatched checkpoint fails loudly (missing *or* unexpected keys) instead of
silently leaving parts of the encoder randomly initialized. As an early guard,
passing one preset's *own* checkpoint file together with a different `--model`
(e.g. `--model 256x448 --checkpoint
pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth`) is
rejected before the model is built.

### Offline DINOv2 requirement

The monodepth branch uses the DINOv2 ViT backbone. `MultiViewUniMatch` normally
loads it with `torch.hub.load("facebookresearch/dinov2", ...)`, which may hit the
network. This standalone path is explicitly offline: it passes a **local**
torch.hub source and never downloads DINOv2.

- `--dinov2-source` must point at the local DINOv2 torch.hub checkout (the
  directory must contain `hubconf.py`). It defaults to `$DINOV2_SOURCE`, then to
  the standard cache `~/.cache/torch/hub/facebookresearch_dinov2_main` when that
  directory already exists. If neither resolves to a directory containing
  `hubconf.py`, the script fails **before** building the model with a message
  explaining `--dinov2-source`; there is no network fallback.
- The composed encoder config always sets `model.encoder.dinov2_pretrained=false`
  for inference. The trained DINOv2 weights are part of the DepthSplat
  checkpoint under `encoder.depth_predictor.pretrained.*` (174 tensors), so the
  architecture is built with random weights and then strict-loaded from the
  checkpoint. Setting `pretrained=false` is therefore intentional: it avoids
  downloading weights that would be overwritten immediately, and the strict load
  still guarantees the final weights are exactly the checkpoint's.
- Existing training commands (`python -m src.main ...`) are unchanged: with no
  `dinov2_source` set, `dinov2_pretrained=true` and the default
  `torch.hub.load("facebookresearch/dinov2", ...)` behaviour is preserved.

To populate the cache once, run an ordinary DINOv2 `torch.hub.load` while online,
or clone `facebookresearch/dinov2` and point `--dinov2-source` at the clone.

### Resize, intrinsics and patch sizes

- Context images are resized with aspect preservation (`scale = max(dst_h/src_h,
  dst_w/src_w)`) and then centre-cropped to the model input resolution. Each
  camera's pixel K is adjusted through the same resize/crop and converted to the
  normalized K used by DepthSplat
  (`[[fx/W, 0, cx/W], [0, fy/H, cy/H], [0, 0, 1]]`).
- **14 vs effective patch size.** DINOv2 operates on 14-pixel patches and the
  encoder internally floors the image to a multiple of 14. That is a *different*
  constraint from the data-shim crop, which aligns the CNN/UNet/cost-volume to a
  multiple of the *effective* patch size
  `shim_patch_size * downscale_factor` (the `dl3dv` experiment uses 16 * 4 = 64,
  the default/re10k settings use 4 * 4 = 16). This script rounds the requested
  input size down to a multiple of the effective patch size read from the
  composed config, and leaves the encoder's internal 14-pixel rounding to the
  model. Both `448x768` and `256x448` are already multiples of 64, so no extra
  crop is applied for the defaults.
- The wide render keeps camera 5's resized pixel focal length (`fx`, `fy`) and
  `cy` and sets `cx = wide_W / 2`, so the per-pixel angular scale and the output
  height are unchanged while the horizontal field of view widens by
  `--width-factor`. The wide pixel K is normalized before being passed to the
  decoder with `image_shape=(H, wide_W)`.
- The decoder builds its projection from the normalized K using an **asymmetric
  frustum**: for pixel intrinsics `fx, fy, cx, cy` it realizes exactly
  `u = fx * X/Z + cx`, `v = fy * Y/Z + cy` (rasterizer convention
  `ndc = 2 * (u/W, v/H) - 1`). Principal-point offsets are therefore honoured
  rather than collapsed into a symmetric field of view, which matters for the
  preserved (possibly off-centre) `cy` of the wide view and for any context K
  that is not exactly centred. For a centred K (`cx = W/2`, `cy = H/2`) the
  projection is identical to the previous symmetric behaviour.

### Notes

- The Gaussian splatting decoder requires the CUDA extension
  `diff-gaussian-rasterization-modified` (see `requirements.txt`); the script
  reports a clear error if it is missing.
- DepthSplat's decoder exposes rendered color and depth only; it does not expose
  an alpha mask, so no mask is written (rather than inventing a misleading one).
- The released checkpoints were not trained on nuScenes, and a 3x-wide camera-5
  frustum is outside the training distribution, so results are for research
  exploration rather than a trained-model benchmark.

## nuScenes Multi-Frame Wide-View Inference

`scripts/inference_nuscenes_wide_multiframes.py` is the nuScenes entry point for
the temporal-context counterpart of the single-frame path: it feeds
`--num-frames` consecutive frames (default 3) x cameras `5,4,3` to the DepthSplat
encoder at once and renders a widened image from the **newest** frame's camera
`5`, analogous to `dggt_infer/inference_nuscenes_multiframes.py`.

The implementation is shared: `scripts/wide_inference_multiframes_core.py` holds
all the logic, and the per-dataset entry points are thin wrappers that pin a
preset:

| Multi-frame entry point | Pinned `--dataset` |
| --- | --- |
| `scripts/inference_nuscenes_wide_multiframes.py` | `nuscenes` |
| `scripts/inference_lyft1920_wide_multiframes.py` | `lyft1920` |
| `scripts/inference_lyft1224_wide_multiframes.py` | `lyft1224` |
| `scripts/inference_ddad_wide_multiframes.py` | `ddad` |

`--dataset` remains an explicit override on each wrapper. The examples below use
the nuScenes wrapper.

### Usage

```bash
# Defaults: 256x448 model, 3 consecutive frames, every scene in the scene list,
# every complete window, no input images saved.
python scripts/inference_nuscenes_wide_multiframes.py \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_multiframes

# Explicit defaults (shown for clarity):
python scripts/inference_nuscenes_wide_multiframes.py \
  --model 256x448 --input-size 256x448 --num-frames 3 --cameras 5,4,3 \
  --render-camera 5 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_multiframes

# A single output frame (the window whose newest frame is 002):
python scripts/inference_nuscenes_wide_multiframes.py \
  --scene 037 --frame 002 --num-frames 3 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_multiframes

# Ego-car masking is on by default (every view except the current render view);
# point at a different mask root, mask the render view too for a diagnostic, or
# disable masking entirely for a controlled comparison (keeps every Gaussian):
python scripts/inference_nuscenes_wide_multiframes.py \
  --car-mask-root datasets/nuscenes/processed_10Hz/nuscenes_mask \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_multiframes
python scripts/inference_nuscenes_wide_multiframes.py \
  --mask-render-view \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_multiframe_all_masks_check
python scripts/inference_nuscenes_wide_multiframes.py \
  --disable-car-mask \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_multiframes
```

### Causal windows

For valid frames `[000,001,002,003]` and `--num-frames 3`:

| Window | Output (newest) frame |
| --- | --- |
| `[000,001,002]` | `002` |
| `[001,002,003]` | `003` |

- A window is always ordered oldest -> newest and ends on the frame that is
  rendered, so output frame ids start at `num_frames - 1` (`002` for three
  frames). **Incomplete** leading windows are skipped: a scene with fewer than
  `--num-frames` valid frames produces no output.
- `--frame` selects the window whose **newest/output** frame is the given id
  (e.g. `--frame 002`); a frame with fewer than `num_frames - 1` valid
  predecessors has no complete window and is skipped.
- `--max-frames N` limits the **source** frame enumeration per scene *before*
  windowing (so `--num-frames 3 --max-frames 3` yields only the `[000,001,002]`
  window). The default `-1` enumerates every valid frame.
- `--scene`, `--frame` and `--max-frames` otherwise behave like the single-frame
  script; `--cameras` (default `5,4,3`) and `--render-camera` (default `5`) set
  the context and rendered views.

### Per-frame extrinsics

Multi-frame inference **always** uses the already-computed per-frame global
camera-to-world matrices `extrinsics/{frame}_{cam}.txt` directly; it never reads
the static `cam2ego_extrinsics/{cam}.txt` and never recomputes `ego_pose`. Every
view in a window keeps its own global C2W matrix, so all frames live in one
consistent world frame. A missing per-frame file is a hard error. (The
single-frame script's `--extrinsics-source` A/B switch does not exist here.)

Views are flattened **frame-major** into a
`[1, num_frames*len(cameras), 3, H, W]` tensor, each with its normalized K and
global C2W:

```
[oldest cam5, oldest cam4, oldest cam3, ..., newest cam5, newest cam4, newest cam3]
```

All frames of a window share one source shape and one resize/crop plan; a shape
mismatch fails loudly.

### Ego-car masking

Multi-frame inference masks the ego-car pixels by default. Each selected camera
has a static mask under `--car-mask-root` (default
`datasets/nuscenes/processed_10Hz/nuscenes_mask`), mapped by nuScenes camera id:

| Camera id | Camera | Mask file |
| --- | --- | --- |
| 0 | `CAM_FRONT` | `CAM_FRONT_mask.png` |
| 1 | `CAM_FRONT_LEFT` | `CAM_FRONT_LEFT_mask.png` |
| 2 | `CAM_FRONT_RIGHT` | `CAM_FRONT_RIGHT_mask.png` |
| 3 | `CAM_BACK_LEFT` | `CAM_BACK_LEFT_mask.png` |
| 4 | `CAM_BACK_RIGHT` | `CAM_BACK_RIGHT_mask.png` |
| 5 | `CAM_BACK` | `CAM_BACK_mask.png` |

- **Polarity**: black (`< 128`) pixels are removed; white (`>= 128`) pixels are
  kept.
- **Transform**: each mask is loaded as PIL `L` and transformed with *exactly*
  the images' resize/crop plan — a NEAREST resize to the scaled size, then the
  same centre crop. It is **not** plain-resized straight to the destination, so
  mask pixels stay aligned with the resized/cropped images. The mask's source
  resolution must equal the source images' resolution.
- **Which views (default, `all_except_render_view`)**: each camera's mask is
  applied to **every view except the single current/newest render view**. For a
  3-frame `[5, 4, 3]` window with render camera 5 (`t3_cam5`), that masks the
  historical `t1/t2` cams 5/4/3 **and** the current `t3` cams 4/3, leaving only
  `t3_cam5` fully preserved.
- **Diagnostic (`--mask-render-view`, `all_views`)**: additionally applies the
  masks to the current render view itself, so the preset ego-car region (e.g. a
  rear ego-car hood) is removed there too. Use this to render an image expected
  to have no rear ego-car hood.
- **Disable (`--disable-car-mask`)**: highest precedence; keeps every Gaussian
  from every view and overrides both policies above.
- **Precedence**: `--disable-car-mask` > `--mask-render-view` > default.
- **Failure mode**: a missing required mask for any selected camera is a hard
  error (there is no silent all-ones fallback).
- Pruning drops the affected Gaussians from `means`, `covariances`, `harmonics`
  and `opacities` identically (rather than only zeroing opacity) after the
  encoder and before the decoder.

The multi-frame and single-frame paths share one mask implementation and the
same policy (`--car-mask-root` / `--mask-render-view` / `--disable-car-mask`);
see the single-frame "Ego-car masking" subsection for the mapping/polarity/
transform details.

This overrides the DGGT nuScenes behaviour (which only masked `CAM_BACK` on
historical frames): here every camera's mask is applied to its own views under
the policy above, except the preserved current render view by default.

### local-mv-match

DepthSplat's multi-view transformer matches each reference view against only
`local_mv_match + 1` nearest views (by camera-centre distance) when `V > 3`. With
the trained default `local_mv_match=2`, a nine-view window would mostly select
the *temporal same-camera* views (their centres are close) and drop the lateral
cameras. This script therefore composes
`model.encoder.local_mv_match = num_frames*len(cameras) - 1` by default (`8` for
`V = 9`), so every flattened view participates. Pass `--local-mv-match N` (e.g.
`2` for the trained config default) to override it; negative values are
rejected. Because the encoder keeps only `local_mv_match + 1` views when
`V > 3`, an explicit `--local-mv-match 0` would keep each view only against
itself and leave an empty cross-view cost volume that can produce NaNs, so `0`
is rejected for `V > 3`. It remains valid for `V <= 3` (including single-frame
windows), where the encoder matches all views and ignores the setting.

### Output

- Only the newest frame's render camera is rendered and written to
  `<output>/<scene>/rgb/{newest_frame}_{render_cam}_wide.jpg` (JPEG quality 95),
  e.g. `002_5_wide.jpg` — the same file naming as the single-frame script. Use a
  **different `--output-dir`** from the single-frame run to avoid overwriting its
  results; the default is `outputs/<dataset>_wide_multiframes`
  (`outputs/nuscenes_wide_multiframes` for nuscenes).
- The default width factor is `3` (same as the single-frame path): an input of
  `256x448` renders `256x1344` and `448x768` renders `448x2304` (aspect-derived
  inputs, e.g. lyft1224 `384x448`, render `384x1344`).
- Input images are not saved unless `--save-inputs` is passed, and no mask/alpha
  image is invented.

Model presets, `--input-size` semantics, offline DINOv2 handling, resize/crop and
patch sizes, strict encoder loading and `--dry-run` are shared with the
single-frame script and documented above.

## Lyft and DDAD (dataset presets)

The wide-view inference scripts share a `--dataset` preset layer so the same
pipeline runs on nuScenes, Lyft (`lyft1920`, `lyft1224`) and DDAD. The preset
supplies the data root, scene list, context/render cameras, ego-car mask naming
and the multi-frame extrinsics convention. Any explicitly passed data flag
(`--data-root`, `--scene-list`, `--cameras`, `--render-camera`,
`--car-mask-root`, `--output-dir`) still wins; unset flags fall back to the
preset. `nuscenes` is the default and is byte-identical to the previous
behaviour.

| `--dataset` | Data root | Scene list | Cameras (render) | Ego-car masks | Single extr. | Multi extr. | Default input |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nuscenes` (default) | `datasets/nuscenes/processed_10Hz/trainval2` | `nuScenes_Val2.txt` | `5,4,3` (`5`) | `datasets/nuscenes/processed_10Hz/nuscenes_mask/CAM_*_mask.png` | `cam2ego` | `per_frame` | `256x448` (model) |
| `lyft1920` | `datasets/lyft/lyft_val1920_3cams` | `lyft_val1920.txt` | `5,4,3` (`5`) | `<root>/ego_car_masks/<cam>.jpg` | `cam2ego` | `per_frame` | `256x448` (model) |
| `lyft1224` | `datasets/lyft/lyft_val1224_3cams` | `lyft_val1224.txt` | `5,4,3` (`5`) | `<root>/ego_car_masks/<cam>.jpg` | `cam2ego` | `per_frame` | `384x448` (aspect) |
| `ddad` | `datasets/ddad/valid` | `valid.txt` | `5,4,3` (`5`) | `<root>/<scene>/ego_car_masks/<cam>.jpg` | `cam2ego` | `compose` | `256x448` (aspect) |

Default output dirs become `outputs/<dataset>_wide` (single-frame) and
`outputs/<dataset>_wide_multiframes` (multi-frame), so nuScenes stays
`outputs/nuscenes_wide` / `outputs/nuscenes_wide_multiframes`.

### Layout differences

- **Lyft** (`lyft_val1920_3cams`, `lyft_val1224_3cams`): cameras `3,4,5`; per
  scene `images/{frame}_{cam}.jpg`, `intrinsics/{cam}.txt`,
  `cam2ego_extrinsics/{cam}.txt`, `extrinsics/{frame}_{cam}.txt`,
  `ego_pose/{frame}.txt`. Ego-car masks live at the **split root** in
  `ego_car_masks/<cam>.jpg` (`0.jpg`..`5.jpg`); no sky masks are used.
- **DDAD** (`datasets/ddad/valid`): cameras `0..5` (we use `5,4,3`); per scene
  `images/{frame}_{cam}.jpg`, `intrinsics/{cam}.txt`,
  `cam2ego_extrinsics/{cam}.txt`, `ego_pose/{frame}.txt`, `sky_masks/` and a
  **per-scene** `ego_car_masks/<cam>.jpg`. There is **no** `extrinsics/`
  directory.

### DDAD extrinsics must be composed

Because DDAD has no `extrinsics/{frame}_{cam}.txt`, multi-frame inference composes
the global OpenCV camera-to-world matrix:

```
C2W = ego_pose/{frame}.txt @ cam2ego_extrinsics/{cam}.txt
```

`--extrinsics-source` (multi-frame) defaults to `auto`: it resolves to
`per_frame` on nuScenes/Lyft and to `compose` on DDAD. Pass `per_frame` or
`compose` explicitly to force a mode; a missing input file is a hard error (on
DDAD, forcing `per_frame` names the missing `extrinsics/...` path and points back
at `compose`). The single-frame path always uses the static `cam2ego` rig
(`--extrinsics-source {cam2ego,per_frame}`), unchanged.

### Ego-car masks

Mask polarity, resize/crop transform and the masking policy are the **same** as
nuScenes (documented in the two "Ego-car masking" subsections): the mask is
loaded as PIL `L`, NEAREST-resized to the scaled size, centre-cropped with the
images' plan, and black (`<128`) pixels are removed while white (`>=128`) is
kept. Each camera's mask is applied to its own view except the render camera,
which is preserved by default; `--mask-render-view` masks it too and
`--disable-car-mask` disables masking. The only difference is the file location
(Lyft split-level `<cam>.jpg`, DDAD per-scene `<scene>/ego_car_masks/<cam>.jpg`).
The masks' source resolution must match the images'.

### Commands

Each dataset has its own wrapper (the pinned dataset is built in; `--dataset`
can still override it).

```bash
# Single-frame, Lyft 1920x1080 (defaults: model 256x448, width factor 3x,
# cameras 5,4,3, render camera 5; cams 4/3 masked, cam 5 preserved).
python scripts/inference_lyft1920_wide.py \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/lyft1920_wide

# Multi-frame, Lyft 1920x1080 (3 consecutive frames, newest rendered).
python scripts/inference_lyft1920_wide_multiframes.py \
  --num-frames 3 --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/lyft1920_wide_multiframes

# Single-frame, Lyft 1224x1024.
python scripts/inference_lyft1224_wide.py \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/lyft1224_wide

# Single-frame, DDAD (composed extrinsics are only needed by multi-frame).
python scripts/inference_ddad_wide.py \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/ddad_wide

# Multi-frame, DDAD: --extrinsics-source auto resolves to compose
# (ego_pose @ cam2ego) because DDAD has no extrinsics/ directory.
python scripts/inference_ddad_wide_multiframes.py \
  --num-frames 3 --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/ddad_wide_multiframes
```

The equivalent core invocations (explicit preset) are e.g.
`python scripts/wide_inference_core.py --dataset lyft1920 ...` and
`python scripts/wide_inference_multiframes_core.py --dataset ddad ...`.

Add `--dry-run` to any command to validate paths, masks and extrinsics without a
GPU.

### Native sizes and derived input height

Native image sizes are Lyft 1920x1080 (`lyft1920`), Lyft 1224x1024
(`lyft1224`) and DDAD 1936x1216; the generic aspect-preserving resize + centre
crop plan handles them without special cases.

The model input size follows the `--model` preset (`256x448` by default). The
width is resolved exactly as before (preset width, then `--input-size` width,
then `--width`) and floored to the effective patch size. The **height** depends
on the dataset's `input_height_policy`:

- `nuscenes` and `lyft1920` use `model`: the height is the model preset height,
  so the default input stays `256x448` (or `448x768` with `--model 448x768`)
  regardless of the width factor.
- `lyft1224` and `ddad` use `aspect`: when no explicit height is given, the
  height is derived at the resolved width from the native aspect ratio,
  `h = nearest_multiple_of_patch(native_h * width / native_w)`, so the native
  aspect is preserved at the chosen width while picking the closest
  model-usable height.

Explicit heights always win: `--height` (or the height of an explicit
`--input-size`) is honored and floored to the patch size, skipping the
derivation.

Example derived inputs at the effective patch size 64:

| Dataset | Width | Input (HxW) | Wide output (factor 3) |
| --- | --- | --- | --- |
| `lyft1224` | 448 (default) | `384x448` | `384x1344` |
| `ddad` | 448 (default) | `256x448` | `256x1344` |
| `lyft1224` | 768 (`--model 448x768` / `--width 768`) | `640x768` | `640x2304` |
| `ddad` | 768 (`--model 448x768` / `--width 768`) | `512x768` | `512x2304` |
| any | explicit `--input-size 448x768` | `448x768` | `448x2304` |

The output width factor stays the nuScenes-consistent default of `3x`
(`--width-factor`; an input `256x448` renders `256x1344`).

## Speed benchmark

`scripts/benchmark_nuscenes_wide.py` is the nuScenes entry point for the wide
speed benchmark. It times the real inference pipeline of either path with
`torch.cuda.Event` and saves no images. It builds the model once, loads one
sample once (outside the timing loop), runs `--warmup` iterations (excluded),
then times `--measure` iterations and reports mean / median / min / max / stdev
latency in milliseconds and throughput as `1000 / mean` FPS.

The implementation is shared: `scripts/benchmark_wide_core.py` holds all the
logic, and the per-dataset entry points are thin wrappers that pin a preset
(`--dataset` can still override):

| Benchmark entry point | Pinned `--dataset` |
| --- | --- |
| `scripts/benchmark_nuscenes_wide.py` | `nuscenes` |
| `scripts/benchmark_lyft1920_wide.py` | `lyft1920` |
| `scripts/benchmark_lyft1224_wide.py` | `lyft1224` |
| `scripts/benchmark_ddad_wide.py` | `ddad` |

### Commands

The four combinations (default model `256x448`; the `448x768` preset renders at
its native input size). The default width factor is `3`, so `256x448` renders
`256x1344` and `448x768` renders `448x2304`; aspect-derived datasets
(`lyft1224`, `ddad`) derive their input height at the chosen width (see the
[Lyft and DDAD](#lyft-and-ddad-dataset-presets) section).

```bash
# single frame, 256x448
python scripts/benchmark_nuscenes_wide.py \
  --mode single --model 256x448 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main

# single frame, 448x768
python scripts/benchmark_nuscenes_wide.py \
  --mode single --model 448x768 --input-size 448x768 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main

# multi frame (3 consecutive frames), 256x448
python scripts/benchmark_nuscenes_wide.py \
  --mode multi --model 256x448 --num-frames 3 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main

# multi frame (3 consecutive frames), 448x768
python scripts/benchmark_nuscenes_wide.py \
  --mode multi --model 448x768 --input-size 448x768 --num-frames 3 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main

# each dataset has its own wrapper (the preset is pinned; masks and extrinsics
# are resolved from it):
python scripts/benchmark_lyft1920_wide.py --mode multi \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main
python scripts/benchmark_ddad_wide.py --mode multi \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main
```

Override the sampling / timing and write JSON stats for scripted comparisons:

```bash
# first complete window of a specific scene, 20 / 100 iterations, JSON output
python scripts/benchmark_nuscenes_wide.py \
  --mode multi --model 256x448 --scene 037 --frame 002 \
  --warmup 20 --measure 100 \
  --json outputs/bench_multi_256x448.json \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main
```

### What is measured

Each timed iteration calls exactly the inference scripts' pipeline:

```
images → wide._render_wide_impl(...) → encoder → (ego-car Gaussian filter) → decoder → RGB
```

- `--mode single` loads a frame with `wide.load_frame_inputs` (static `cam2ego`
  extrinsics, as in `wide_inference_core.py`).
- `--mode multi` loads a causal window with `mf.load_window_inputs` (per-frame
  global extrinsics, as in `wide_inference_multiframes_core.py`).
- The model is built once with `wide.build_model`; the ego-car `gaussian_filter`
  is built with the same policy resolution as the inference scripts (masking is
  **on** by default, `--disable-car-mask` / `--mask-render-view` behave
  identically).

### Metrics

| Metric | Definition |
| --- | --- |
| `mean_ms` / `median_ms` / `min_ms` / `max_ms` | per-iteration CUDA-event wall time |
| `stdev_ms` | sample standard deviation (`0` for a single sample) |
| `fps` | `1000 / mean_ms` |

### Notes

- **Warmup**: the first `--warmup` iterations (default `10`) are run before
  timing and excluded from the statistics, so lazy CUDA kernel loading / cuDNN
  autotuning does not inflate the result.
- **Timing excludes data loading and model loading**: the sample is loaded once
  and the model built once, both *before* the timing loop; their one-time cost
  is reported separately (`Data load` / `Model build` in the output).
- **No images are saved** and there is no `--output-dir`.
- **CUDA is required** for the Gaussian-splatting decoder; when
  `torch.cuda.is_available()` is false the script exits with a clear error.
- **Offline DINOv2**: `--dinov2-source` must resolve to a local source (no
  network fallback), exactly like the inference scripts.
- **JSON**: pass `--json PATH` to also write the metadata and stats as JSON.
- Sample selection defaults to the first scene in the scene list and the first
  valid frame (single) / first complete window (multi); `--scene` / `--frame`
  narrow it. `--local-mv-match` is multi-only and defaults to `V - 1`; passing
  it in `--mode single` is rejected.

## Citation

```
@inproceedings{xu2024depthsplat,
      title   = {DepthSplat: Connecting Gaussian Splatting and Depth},
      author  = {Xu, Haofei and Peng, Songyou and Wang, Fangjinhua and Blum, Hermann and Barath, Daniel and Geiger, Andreas and Pollefeys, Marc},
      booktitle={CVPR},
      year={2025}
    }
```



## Acknowledgements

This project is developed with several fantastic repos: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [MVSplat360](https://github.com/donydchen/mvsplat360), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank the original authors for their excellent work.


