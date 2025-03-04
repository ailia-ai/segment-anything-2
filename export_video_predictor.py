# Export memory attention and memory encoder
# Implemented by ax Inc. 2024

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default="hiera_t", choices=["hiera_l", "hiera_b+", "hiera_s", "hiera_t"])
parser.add_argument('--framework', default="onnx", choices=["onnx", "tflite", "torch", "ailia_tflite"])
parser.add_argument('--accuracy', default="float", choices=["float", "int8", "mixed"])
parser.add_argument('--mode', default="both", choices=["both", "import", "export", "calibration"])
parser.add_argument('--image_size', default=1024, type=int, choices=[512, 1024])
parser.add_argument('--version', default="2.1", choices=["2", "2.1"])
args = parser.parse_args()

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# output
os.makedirs("output", exist_ok=True)
os.makedirs("model", exist_ok=True)

# export settings
export_to_onnx = args.framework=="onnx" and (args.mode=="export" or args.mode=="both")
import_from_onnx = args.framework=="onnx" and (args.mode=="import" or args.mode=="both")
export_to_tflite = args.framework=="tflite" and (args.mode=="export" or args.mode=="both")
import_from_tflite = (args.framework=="tflite" or args.framework=="ailia_tflite") and (args.mode=="import" or args.mode=="both")
tflite_int8 = False if args.accuracy == "float" else args.accuracy
calibration = args.mode == "calibration"

if args.framework=="ailia_tflite" and import_from_tflite:
    import_from_tflite = "ailia_tflite"

# import
model_id = args.model_id
if model_id == "hiera_l":
     sam2_checkpoint = f"./checkpoints/sam{args.version}_hiera_large.pt"
     model_cfg = f"configs/sam{args.version}/sam{args.version}_hiera_l.yaml"
elif model_id == "hiera_b+":
    sam2_checkpoint = f"./checkpoints/sam{args.version}_hiera_base_plus.pt"
    model_cfg = f"configs/sam{args.version}/sam{args.version}_hiera_b+.yaml"
elif model_id == "hiera_s":
    sam2_checkpoint = f"./checkpoints/sam{args.version}_hiera_small.pt"
    model_cfg = f"configs/sam{args.version}/sam{args.version}_hiera_s.yaml"
elif model_id == "hiera_t":
    sam2_checkpoint = f"./checkpoints/sam{args.version}_hiera_tiny.pt"
    model_cfg = f"configs/sam{args.version}/sam{args.version}_hiera_t.yaml"
else:
    print("unknown model id")
    exit()

# resolution settings
if args.version == "2.1":
    model_id = model_id + "_2.1"
if args.image_size == 512:
    model_id = model_id + "_512"

device = torch.device("cpu")
print(f"using device: {device}")

from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device, image_size=args.image_size, calibration=calibration)

#if export_to_tflite or import_from_tflite:
#    predictor.set_num_maskmem(num_maskmem=1, max_obj_ptrs_in_encoder=1)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

video_dir = "./notebooks/videos/bedroom_short"

if calibration:
    video_dir = "./calibration/bedroom"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir, import_from_onnx=import_from_onnx, import_from_tflite=import_from_tflite, tflite_int8=tflite_int8, model_id=model_id)
predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
# for labels, `1` means positive click and `0` means negative click
if False:#args.framework == "tflite" or args.framework == "ailia_tflite":
    points = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)
else:
    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    labels = np.array([1, 1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    import_from_onnx=import_from_onnx,
    export_to_onnx=export_to_onnx,
    import_from_tflite=import_from_tflite,
    export_to_tflite=export_to_tflite,
    tflite_int8=tflite_int8,
    model_id=model_id
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
#plt.show()
plt.savefig(f'output/video_'+model_id+'.'+args.accuracy+'.png')

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, import_from_onnx=import_from_onnx, export_to_onnx=export_to_onnx, import_from_tflite=import_from_tflite, export_to_tflite=export_to_tflite, tflite_int8=tflite_int8, model_id=model_id):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    #plt.show()
    plt.savefig(f'output/video{out_frame_idx+1}_'+model_id+'.'+args.accuracy+'.png')
