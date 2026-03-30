#!/usr/bin/env python3
"""Export prompt_encoder and memory_attention ONNX for all model sizes."""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default="hiera_t", choices=["hiera_l", "hiera_b+", "hiera_s", "hiera_t"])
parser.add_argument('--component', default="both", choices=["both", "prompt_encoder", "memory_attention"])
parser.add_argument('--image_size', default=1024, type=int, choices=[512, 1024])
args = parser.parse_args()

import os
import numpy as np
import torch

os.makedirs("model", exist_ok=True)

model_id = args.model_id
if model_id == "hiera_l":
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
elif model_id == "hiera_b+":
    sam2_checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
elif model_id == "hiera_s":
    sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
elif model_id == "hiera_t":
    sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

if args.image_size == 512:
    model_id = model_id + "_512"

device = torch.device("cpu")

# Export prompt_encoder
if args.component in ("both", "prompt_encoder"):
    print(f"=== Exporting prompt_encoder for {model_id} ===")
    from sam2.build_sam import build_sam2
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, image_size=args.image_size)
    sam2_model.eval()

    # Dummy inputs matching the export in sam2_image_predictor.py
    coords = torch.rand(1, 2, 2, dtype=torch.float32)  # batch=1, n=2 points, xy
    labels = torch.ones(1, 2, dtype=torch.int32)         # batch=1, n=2 labels
    mask_input = torch.zeros(1, 1, sam2_model.image_size // 4, sam2_model.image_size // 4, dtype=torch.float32)
    masks_enable = torch.tensor([0], dtype=torch.int32)

    output_path = f'model/prompt_encoder_{model_id}.onnx'
    torch.onnx.export(
        sam2_model.sam_prompt_encoder,
        (coords, labels, mask_input, masks_enable),
        output_path,
        input_names=["coords", "labels", "masks", "masks_enable"],
        output_names=["sparse_embeddings", "dense_embeddings", "dense_pe"],
        dynamic_axes={
            'coords': {0: 'b', 1: 'n'},
            'labels': {0: 'b', 1: 'n'},
            'masks': {0: 'b', 1: 'c', 2: 'h', 3: 'w'},
        },
        verbose=False, opset_version=17
    )
    print(f"  Saved: {output_path}")
    del sam2_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

# Export memory_attention
if args.component in ("both", "memory_attention"):
    print(f"=== Exporting memory_attention for {model_id} ===")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device, image_size=args.image_size)
    predictor.eval()

    image_size = args.image_size
    mem_dim = predictor.mem_dim  # typically 64
    hidden_dim = predictor.hidden_dim  # typically 256

    # Dummy inputs matching the shapes from sam2_base.py memory attention export
    # curr: (seq_len, batch, hidden_dim) where seq_len = (image_size//16)^2 = 4096 for 1024
    seq_len = (image_size // 16) ** 2
    B = 1
    curr = torch.randn(seq_len, B, hidden_dim, dtype=torch.float32)
    curr_pos = torch.randn(seq_len, B, hidden_dim, dtype=torch.float32)

    # memory_1: RoPE-applied memory (n_1, batch, mem_dim) - memory from encoder has mem_dim
    # memory_2: non-RoPE memory / obj pointers (n_2, batch, mem_dim)
    n_1 = seq_len  # one frame of memory
    n_2 = 1  # one obj pointer token
    memory_1 = torch.randn(n_1, B, mem_dim, dtype=torch.float32)
    memory_2 = torch.randn(n_2, B, mem_dim, dtype=torch.float32)
    memory_pos_1 = torch.randn(n_1, B, mem_dim, dtype=torch.float32)
    memory_pos_2 = torch.randn(n_2, B, mem_dim, dtype=torch.float32)

    # Pre-allocate RoPE weights
    predictor.memory_attention.allocate_rope_attention_weight(
        curr=[curr],
        curr_pos=[curr_pos],
        image_size=image_size,
    )

    output_path = f'model/memory_attention_{model_id}.opt.onnx'
    torch.onnx.export(
        predictor.memory_attention,
        (curr, memory_1, memory_2, curr_pos, memory_pos_1, memory_pos_2),
        output_path,
        input_names=["curr", "memory_1", "memory_2", "curr_pos", "memory_pos_1", "memory_pos_2"],
        output_names=["pix_feat"],
        dynamic_axes={
            'curr': {1: 'b'},
            'memory_1': {0: 'n_1', 1: 'b'},
            'memory_2': {0: 'n_2', 1: 'b'},
            'curr_pos': {1: 'b'},
            'memory_pos_1': {0: 'n_1', 1: 'b'},
            'memory_pos_2': {0: 'n_2', 1: 'b'},
            'pix_feat': {1: 'b'}
        },
        verbose=False, opset_version=17
    )
    print(f"  Saved: {output_path}")

print("Done!")
