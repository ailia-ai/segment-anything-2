# SAM 2.1 Export to ONNX and TFLITE

## Download model


```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## Requirements

onnx

```
torch 2.2.1
onnx 1.16.2
```

tflite

```
torch 2.4.0
ai-edge-torch 0.2.0
tensorflow 2.18.0
```

## Export and Inference

onnx

```
python3 export_image_predictor.py --framework onnx
python3 export_video_predictor.py --framework onnx
```

tflite (float)

```
export PJRT_DEVICE=CPU
python3 export_image_predictor.py --framework tflite
python3 export_video_predictor.py --framework tflite
```

generate calibration data

```
export PJRT_DEVICE=CPU
python3 export_image_predictor.py --accuracy int8 --image_size 512 --mode calibration
python3 export_video_predictor.py --accuracy int8 --image_size 512 --mode calibration
```

tflite (int8)

```
export PJRT_DEVICE=CPU
python3 export_image_predictor.py --framework tflite --accuracy int8 --image_size 512
python3 export_video_predictor.py --framework tflite --accuracy int8 --image_size 512
```

tflite (mixed)

```
export PJRT_DEVICE=CPU
python3 export_image_predictor.py --framework tflite --accuracy mixed --image_size 512
python3 export_video_predictor.py --framework tflite --accuracy mixed --image_size 512
```

## Inference only

onnx

```
download_onnx_models.sh
python3 export_image_predictor.py --framework onnx --mode import
python3 export_video_predictor.py --framework onnx --mode import
```

tflite (float)

```
download_tflite_models.sh
python3 export_image_predictor.py --framework tflite --mode import
python3 export_video_predictor.py --framework tflite --mode import
```

tflite (int8)

```
download_tflite_models.sh
python3 export_image_predictor.py --framework tflite --mode import --accuracy int8 --image_size 512
python3 export_video_predictor.py --framework tflite --mode import --accuracy int8 --image_size 512
```

tflite (mixed)

```
download_tflite_models.sh
python3 export_image_predictor.py --framework tflite --mode import --accuracy mixed --image_size 512
python3 export_video_predictor.py --framework tflite --mode import --accuracy mixed --image_size 512
```

ailia_tflite

```
download_tflite_models.sh
python3 export_image_predictor.py --framework ailia_tflite --mode import
python3 export_video_predictor.py --framework ailia_tflite --mode import
```

## Options

- `--image_size 512` : Use 512x512 resolution (default 1024x1024)
- `--version 2` : Use SAM2 (default 2.1)

## Test

Replacing the complex tensor of RotaryEnc with matmul. To test this behavior, you can also run it with torch.

```
python3 export_video_predictor.py --framework torch
```

## Artifacts

The deliverables will be stored below.

```
output/*
model/*
```

You can also download it from the following.

### ONNX

- https://storage.googleapis.com/ailia-models/segment-anything-2.1/image_encoder_hiera_t_2.1.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2.1/prompt_encoder_hiera_t_2.1.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2.1/mask_decoder_hiera_t_2.1.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_encoder_hiera_t_2.1.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2.1/mlp_hiera_t_2.1.onnx
- https://storage.googleapis.com/ailia-models/segment-anything-2.1/memory_attention_hiera_t_2.1.onnx (4dim matmul, batch = 1)
- https://storage.googleapis.com/ailia-models/segment-anything-2.1/obj_ptr_tpos_proj_hiera_t_2.1.onnx

### TFLITE

#### Float

- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/image_encoder_hiera_t_2.1.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/prompt_encoder_hiera_t_2.1.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/mask_decoder_hiera_t_2.1.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/mlp_hiera_t_2.1.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/memory_encoder_hiera_t_2.1.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/memory_attention_hiera_t_2.1.tflite (4dim matmul, batch = 1, num_maskmem = 8)
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/obj_ptr_tpos_proj_hiera_t_2.1.tflite

#### Int8 (tensorflow quantization)

- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/image_encoder_hiera_t_2.1_512.int8.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/prompt_encoder_hiera_t_2.1_512.int8.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/mask_decoder_hiera_t_2.1_512.int8.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/mlp_hiera_t_2.1_512.int8.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/memory_encoder_hiera_t_2.1_512.int8.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/memory_attention_hiera_t_2.1_512.int8.tflite (4dim matmul, batch = 1, num_maskmem = 8)
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/obj_ptr_tpos_proj_hiera_t_2.1_512.int8.tflite

#### Mixed (torch quantization)

- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/image_encoder_hiera_t_2.1_512.mixed.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/mask_decoder_hiera_t_2.1_512.mixed.tflite
- https://storage.googleapis.com/ailia-models-tflite/segment-anything-2.1/memory_attention_hiera_t_2.1_512.mixed.tflite (4dim matmul, batch = 1, num_maskmem = 8)

## Inference Example

- [ailia-models](https://github.com/axinc-ai/ailia-models/tree/master/image_segmentation/segment-anything-2)
- [ailia-models-tflite](https://github.com/axinc-ai/ailia-models-tflite/pull/90)

## Original document

- [README_ORIGINAL.md](README_ORIGINAL.md)
