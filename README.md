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

`scripts/inference_nuscenes_wide.py` is a standalone nuScenes inference path. It
reconstructs Gaussians from cameras `5, 4, 3` of a frame and renders a
horizontally widened image from camera `5`, without going through the Lightning
`ModelWrapper.test_step` (which assumes a ground-truth target view). By default
it processes every scene in the scene list and every valid frame; extrinsics
come from the static camera-to-ego rig (see
[Extrinsics source](#extrinsics-source) below).

### Usage

```bash
# Defaults: 448x768 model, every scene in the scene list, every valid frame,
# no input images saved.
python scripts/inference_nuscenes_wide.py \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide

# Select the model preset and the input resize size explicitly (these are the
# same as the defaults, shown for clarity):
python scripts/inference_nuscenes_wide.py \
  --model 448x768 \
  --input-size 448x768 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide

# The other local checkpoint, rendered at its native 256x448 input:
python scripts/inference_nuscenes_wide.py \
  --model 256x448 \
  --input-size 256x448 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir outputs/nuscenes_wide_256x448

# Render a single scene/frame instead:
python scripts/inference_nuscenes_wide.py \
  --model 448x768 \
  --input-size 448x768 \
  --scene 037 \
  --frame 0 \
  --dinov2-source ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --width-factor 2 \
  --output-dir outputs/nuscenes_wide
```

- The model defaults to the `448x768` preset; context cameras default to
  `5,4,3`, the render camera to `5`, and the output width factor to `2`.
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

### Checkpoints and input resolutions

The `--model` preset selects the matching locally shipped base checkpoint and
its architecture; `--input-size` chooses the input resize independently:

| `--model` (alias `--resolution`) | Default checkpoint | Default input | Output (factor 2) | `gaussian_scale_max` |
| --- | --- | --- | --- | --- |
| `448x768` (default) | `pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth` | 448x768 | 448x1536 | 0.1 |
| `256x448` | `pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth` | 256x448 | 256x896 | 3.0 |

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
- The released checkpoints were not trained on nuScenes, and a 2x-wide camera-5
  frustum is outside the training distribution, so results are for research
  exploration rather than a trained-model benchmark.

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


