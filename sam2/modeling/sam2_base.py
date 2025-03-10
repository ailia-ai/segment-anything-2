# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

import numpy as np

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    add_all_frames_to_correct_as_cond = False # For compatibility

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.memory_attention_onnx_exported = False
        self.memory_attention_tflite_exported = False
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.memory_encoder_onnx_exported = False
        self.memory_encoder_tflite_exported = False
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        self.mlp_onnx_exported = False
        self.mlp_tflite_exported = False

        self.obj_ptr_tpos_proj_onnx_exported = False
        self.obj_ptr_tpos_proj_tflite_exported = False

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

        # debug
        self.debug = False
         
        # onnx
        self.image_encoder_onnx = None
        self.prompt_encoder_onnx = None
        self.mask_decoder_onnx = None
        self.mlp_onnx = None
        self.memory_attention_onnx = None
        self.memory_encoder_onnx = None
        self.obj_ptr_tpos_proj_onnx = None

        # tflite
        self.image_encoder_tflite = None
        self.prompt_encoder_tflite = None
        self.mask_decoder_tflite = None
        self.mlp_tflite = None
        self.memory_attention_tflite = None
        self.memory_encoder_tflite = None
        self.obj_ptr_tpos_proj_tflite = None

        # calibration
        self.calibration = False
        self.calibration_cnt_memory_attention = 0
        self.calibration_cnt_memory_encoder = 0
        self.calibration_cnt_obj_ptr_tpos_proj = 0
        self.calibration_cnt_mlp = 0

        # Check decoder sample parameter
        assert(self.image_size == 512 or self.image_size == 1024)
        assert(self.num_feature_levels == 3)
        assert(self.hidden_dim == 256)
        assert(self.num_maskmem == 7)
        assert(self.directly_add_no_mem_embed == True)
        #assert(self.training == False)
        assert(self.mem_dim == 64)
        #assert(self.add_tpos_enc_to_obj_ptrs == False) # SAM2ではFalse、SAM2.1ではTrue
        assert(self.use_obj_ptrs_in_encoder == True)
        assert(self.add_all_frames_to_correct_as_cond == False)
        assert(self.multimask_output_in_sam == True)
        assert(self.multimask_min_pt_num == 0)
        assert(self.multimask_max_pt_num == 1)
        assert(self.sam_prompt_embed_dim == self.hidden_dim)
        assert(self.backbone_stride == 16)
        assert(self.sam_image_embedding_size == self.image_size // self.backbone_stride)
        assert(self.pred_obj_scores == True)
        assert(self.use_obj_ptrs_in_encoder == True)
        assert(self.use_mlp_for_obj_ptr_proj == True)
        assert(self.soft_no_obj_ptr == False)
        assert(self.fixed_no_obj_ptr == True)
        assert(self.non_overlap_masks_for_mem_enc == False)
        assert(self.binarize_mask_from_pts_for_mem_enc == False or self.binarize_mask_from_pts_for_mem_enc == True) # True for video
        assert(self.sigmoid_scale_for_mem_enc == 20)
        assert(self.sigmoid_bias_for_mem_enc == -10.0)
        assert(self.sam_mask_decoder.dynamic_multimask_via_stability == True)
        assert(self.sam_mask_decoder.dynamic_multimask_stability_delta == 0.05)
        assert(self.sam_mask_decoder.dynamic_multimask_stability_thresh == 0.98)
        assert(self.max_cond_frames_in_attn == -1)
        assert(self.memory_temporal_stride_for_eval == 1)
        assert(self.max_obj_ptrs_in_encoder == 16)
        assert(self.only_obj_ptrs_in_the_past_for_eval == True)
        assert(self.multimask_output_for_tracking == True)
        assert(self.use_multimask_token_for_obj_ptr == True)

    def set_num_maskmem(self, num_maskmem, max_obj_ptrs_in_encoder):
        self.num_maskmem = num_maskmem
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        assert(self.num_maskmem == 1 or self.num_maskmem == 7)
        assert(self.max_obj_ptrs_in_encoder == 1 or self.max_obj_ptrs_in_encoder == 16)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        export_to_onnx=False,
        import_from_onnx=False,
        export_to_tflite=False,
        import_from_tflite=False,
        tflite_int8=False,
        model_id=None
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        if sam_mask_prompt is None:
            import numpy as np
            mask_input_dummy = torch.Tensor(np.zeros((1, self.image_size // 4, self.image_size // 4)))
            masks_enable = torch.tensor([0], dtype=torch.int)
        else:
            mask_input_dummy = sam_mask_prompt
            masks_enable = torch.tensor([1], dtype=torch.int)

        # padding
        convert_to_static_shape = export_to_tflite or import_from_tflite or self.calibration
        original_sparse_embedding_length = sam_point_coords.shape[1] + 1
        max_sparse_embedding_length = 32

        if convert_to_static_shape:
            padding_length= max_sparse_embedding_length - 1
            sam_point_coords_pad = torch.zeros(sam_point_coords.shape[0], padding_length, sam_point_coords.shape[2])
            sam_point_labels_pad = -torch.ones(sam_point_labels.shape[0], padding_length)
            sam_point_coords_pad[:, 0:sam_point_coords.shape[1], :] = sam_point_coords
            sam_point_labels_pad[:, 0:sam_point_labels.shape[1]] = sam_point_labels
            sam_point_coords = sam_point_coords_pad
            sam_point_labels = sam_point_labels_pad

        if import_from_onnx:
            if self.debug:
                print("begin prompt encoder onnx")
            import onnxruntime
            if self.prompt_encoder_onnx == None:
                self.prompt_encoder_onnx = onnxruntime.InferenceSession("model/prompt_encoder_"+model_id+".onnx")
            sparse_embeddings, dense_embeddings, dense_pe = self.prompt_encoder_onnx.run(None, {"coords":sam_point_coords.numpy(), "labels":sam_point_labels.numpy(), "masks":mask_input_dummy.numpy(), "masks_enable":masks_enable.numpy()})
            sparse_embeddings = torch.Tensor(sparse_embeddings)
            dense_embeddings = torch.Tensor(dense_embeddings)
            dense_pe = torch.Tensor(dense_pe)

            if self.mask_decoder_onnx == None:
                self.mask_decoder_onnx  = onnxruntime.InferenceSession("model/mask_decoder_"+model_id+".onnx")
            if self.debug:
                print("backbone_features", backbone_features.shape)
                print("begin mask decoder onnx")
                print("begin mask decoder onnx")
                print("backbone_features", np.sum(backbone_features.numpy()))
                print("image_pe", np.sum(dense_pe.numpy()))
                print("sparse_embeddings", np.sum(sparse_embeddings.numpy()))
                print("dense_embeddings", np.sum(dense_embeddings.numpy()))
                print("high_res_features", np.sum(high_res_features[0].numpy()))
                print("high_res_features", np.sum(high_res_features[1].numpy()))
            masks, iou_pred, sam_tokens_out, object_score_logits  = self.mask_decoder_onnx.run(None, {
                "image_embeddings":backbone_features.numpy(),
                "image_pe": dense_pe.numpy(),
                "sparse_prompt_embeddings": sparse_embeddings.numpy(),
                "dense_prompt_embeddings": dense_embeddings.numpy(),
                #repeat_image=False,  # the image is already batched
                "high_res_features1":high_res_features[0].numpy(),
                "high_res_features2":high_res_features[1].numpy(),
                "attn_masks":attn_masks.numpy()})
            masks = torch.Tensor(masks)
            iou_pred = torch.Tensor(iou_pred)
            sam_tokens_out = torch.Tensor(sam_tokens_out)
            object_score_logits = torch.Tensor(object_score_logits)
            low_res_multimasks, ious, sam_output_tokens, object_score_logits  = self.sam_mask_decoder.forward_postprocess(masks, iou_pred, sam_tokens_out, object_score_logits, multimask_output)
            #print(low_res_multimasks.shape)
            #print(ious.shape)
            #print(sam_output_tokens.shape)
            #print(object_score_logits.shape)

        tflite_int8_prompt_encoder = tflite_int8

        if import_from_tflite:
            if self.debug:
                print("begin prompt encoder tflite")

            if tflite_int8:
                tflite_int8_prompt_encoder = False # 精度不足なのでFloatモデルで推論する
                print("Warning : prompt encoder will use float model for better accuracy.")

            if self.prompt_encoder_tflite == None:
                int8_id = ""
                if tflite_int8_prompt_encoder:
                    int8_id = "." + tflite_int8_prompt_encoder

                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.prompt_encoder_tflite = ailia_tflite.Interpreter(model_path="model/prompt_encoder_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.prompt_encoder_tflite = tf.lite.Interpreter(model_path="model/prompt_encoder_"+model_id+int8_id+".tflite")
                input_details = self.prompt_encoder_tflite.get_input_details()
                self.prompt_encoder_tflite.resize_tensor_input(
                    input_details[2]["index"], 
                    [1, sam_point_coords.shape[1], 2]
                )
                self.prompt_encoder_tflite.allocate_tensors()

            if tflite_int8_prompt_encoder:
                self.prompt_encoder_tflite.set_tensor(input_details[2]["index"], self.format_input_tensor(concat_points[0].numpy(), input_details, 2))
                self.prompt_encoder_tflite.set_tensor(input_details[3]["index"], self.format_input_tensor(concat_points[1].numpy(), input_details, 3))
                self.prompt_encoder_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(mask_input_dummy.numpy(), input_details, 0))
                self.prompt_encoder_tflite.set_tensor(input_details[1]["index"], self.format_input_tensor(masks_enable.numpy(), input_details, 1))
                self.prompt_encoder_tflite.invoke()

                sparse_embeddings = self.get_output_tensor(self.prompt_encoder_tflite, output_details, 1)
                dense_embeddings = self.get_output_tensor(self.prompt_encoder_tflite, output_details, 0)
                dense_pe = self.get_output_tensor(self.prompt_encoder_tflite, output_details, 2)
            else:
                input_details = self.prompt_encoder_tflite.get_input_details()
                output_details = self.prompt_encoder_tflite.get_output_details()

                self.prompt_encoder_tflite.set_tensor(input_details[2]["index"], sam_point_coords.numpy())
                self.prompt_encoder_tflite.set_tensor(input_details[3]["index"], sam_point_labels.numpy())
                self.prompt_encoder_tflite.set_tensor(input_details[0]["index"], mask_input_dummy.numpy())
                self.prompt_encoder_tflite.set_tensor(input_details[1]["index"], masks_enable.numpy())
                self.prompt_encoder_tflite.invoke()

                sparse_embeddings = self.prompt_encoder_tflite.get_tensor(output_details[1]["index"])
                dense_embeddings = self.prompt_encoder_tflite.get_tensor(output_details[0]["index"])
                dense_pe = self.prompt_encoder_tflite.get_tensor(output_details[2]["index"])

            if convert_to_static_shape:
                sparse_embeddings = sparse_embeddings[:, :original_sparse_embedding_length, :]

            # Prepare attn_mask for mask decoder
            if convert_to_static_shape:
                max_num_sparse_embeddings = max_sparse_embedding_length
                sparse_embeddings_pad = np.zeros((sparse_embeddings.shape[0], max_num_sparse_embeddings, sparse_embeddings.shape[2]))
                sparse_embeddings_pad[:,:sparse_embeddings.shape[1],:] = sparse_embeddings
                attn_masks = np.zeros((sparse_embeddings_pad.shape[0], sparse_embeddings_pad.shape[1]), dtype=np.bool8)
                attn_masks[:,:sparse_embeddings.shape[1]] = True
                sparse_embeddings = sparse_embeddings_pad
            else:
                attn_masks = np.ones((sparse_embeddings.shape[0], sparse_embeddings.shape[1]), dtype=np.bool8)

            if self.mask_decoder_tflite == None:
                int8_id = ""
                if tflite_int8:
                    int8_id = "." + tflite_int8

                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.mask_decoder_tflite = ailia_tflite.Interpreter(model_path="model/mask_decoder_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.mask_decoder_tflite = tf.lite.Interpreter(model_path="model/mask_decoder_"+model_id+int8_id+".tflite")

                input_details = self.mask_decoder_tflite.get_input_details()
                self.mask_decoder_tflite.resize_tensor_input(
                    input_details[1]["index"], 
                    [1, sparse_embeddings.shape[1], 256]
                )
                self.mask_decoder_tflite.allocate_tensors()

            input_details = self.mask_decoder_tflite.get_input_details()
            output_details = self.mask_decoder_tflite.get_output_details()

            batched_mode = False

            if tflite_int8:
                self.mask_decoder_tflite.set_tensor(input_details[3]["index"], self.format_input_tensor(backbone_features.numpy(), input_details, 3))
                self.mask_decoder_tflite.set_tensor(input_details[6]["index"], self.format_input_tensor(dense_pe, input_details, 6))
                self.mask_decoder_tflite.set_tensor(input_details[1]["index"], self.format_input_tensor(sparse_embeddings, input_details, 1))
                self.mask_decoder_tflite.set_tensor(input_details[2]["index"], self.format_input_tensor(dense_embeddings, input_details, 2))
                self.mask_decoder_tflite.set_tensor(input_details[5]["index"], batched_mode)
                self.mask_decoder_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(high_res_features[0].numpy(), input_details, 0))
                self.mask_decoder_tflite.set_tensor(input_details[4]["index"], self.format_input_tensor(high_res_features[1].numpy(), input_details, 4))
                self.mask_decoder_tflite.set_tensor(input_details[7]["index"], self.format_input_tensor(attn_masks, input_details, 7))
                self.mask_decoder_tflite.invoke()

                masks = self.get_output_tensor(self.mask_decoder_tflite, output_details, 2)
                iou_pred = self.get_output_tensor(self.mask_decoder_tflite, output_details, 0)
                sam_tokens_out = self.get_output_tensor(self.mask_decoder_tflite, output_details, 3)
                object_score_logits = self.get_output_tensor(self.mask_decoder_tflite, output_details, 1)
            else:
                self.mask_decoder_tflite.set_tensor(input_details[3]["index"], backbone_features.numpy())
                self.mask_decoder_tflite.set_tensor(input_details[6]["index"], dense_pe)
                self.mask_decoder_tflite.set_tensor(input_details[1]["index"], sparse_embeddings)
                self.mask_decoder_tflite.set_tensor(input_details[2]["index"], dense_embeddings)
                self.mask_decoder_tflite.set_tensor(input_details[5]["index"], batched_mode)
                self.mask_decoder_tflite.set_tensor(input_details[0]["index"], high_res_features[0].numpy())
                self.mask_decoder_tflite.set_tensor(input_details[4]["index"], high_res_features[1].numpy())
                self.mask_decoder_tflite.set_tensor(input_details[7]["index"], attn_masks)
                self.mask_decoder_tflite.invoke()

                masks = self.mask_decoder_tflite.get_tensor(output_details[2]["index"])
                iou_pred = self.mask_decoder_tflite.get_tensor(output_details[0]["index"])
                sam_tokens_out = self.mask_decoder_tflite.get_tensor(output_details[3]["index"])
                object_score_logits = self.mask_decoder_tflite.get_tensor(output_details[1]["index"])

            masks = torch.Tensor(masks)
            iou_pred = torch.Tensor(iou_pred)
            sam_tokens_out = torch.Tensor(sam_tokens_out)
            object_score_logits = torch.Tensor(object_score_logits)
            low_res_multimasks, ious, sam_output_tokens, object_score_logits  = self.sam_mask_decoder.forward_postprocess(masks, iou_pred, sam_tokens_out, object_score_logits, multimask_output)
            #print(low_res_multimasks.shape)
            #print(ious.shape)
            #print(sam_output_tokens.shape)
            #print(object_score_logits.shape)

        if not import_from_onnx and not import_from_tflite:
            if self.debug:
                print("begin mask decoder torch")
                print("backbone_features", backbone_features.shape)
            if sam_mask_prompt is None:
                import numpy as np
                mask_input_dummy = torch.Tensor(np.zeros((1, self.image_size // 4, self.image_size // 4)))
                masks_enable = torch.tensor([0], dtype=torch.int)
            else:
                mask_input_dummy = sam_mask_prompt
                masks_enable = torch.tensor([1], dtype=torch.int)
            self.sam_prompt_encoder.generate_dense_pe()
            sparse_embeddings, dense_embeddings, dense_pe = self.sam_prompt_encoder.forward(
                coords=sam_point_coords,
                labels=sam_point_labels,
                masks=mask_input_dummy,
                masks_enable=masks_enable
            )

            attn_masks = torch.ones((sparse_embeddings.shape[0], sparse_embeddings.shape[1]), dtype=torch.bool)

            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = self.sam_mask_decoder.forward_normal(
                image_embeddings=backbone_features,
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,  # the image is already batched
                high_res_features1=high_res_features[0],
                high_res_features2=high_res_features[1],
                attn_masks=attn_masks
            )
            if self.debug:
                print(low_res_multimasks.shape)
                print(ious.shape)
                print(sam_output_tokens.shape)
                print(object_score_logits.shape)

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        if export_to_onnx and not self.mlp_onnx_exported:
            print("x", sam_output_token.shape)
            self.mlp_onnx_exported = True
            torch.onnx.export(
                self.obj_ptr_proj, (sam_output_token), 'model/mlp_'+model_id+'.onnx',
                input_names=["x"],
                output_names=["x_out"],
                dynamic_axes={
                    'x': {0: 'n'},
                    'obj_ptr': {0: 'n'}
                },
                verbose=False, opset_version=17
            )

        if import_from_onnx:
            import onnxruntime
            if self.mlp_onnx == None:
                self.mlp_onnx  = onnxruntime.InferenceSession("model/mlp_"+model_id+".onnx")
            import numpy as np
            obj_ptr = self.mlp_onnx.run(None, {"x":sam_output_token.numpy()})[0]
            obj_ptr = torch.Tensor(obj_ptr)
        
        if export_to_tflite and not self.mlp_tflite_exported:
            self.mlp_tflite_exported = True
            import ai_edge_torch
            import tensorflow as tf
            sample_inputs = (sam_output_token,)
            if not tflite_int8:
                tfl_converter_flags = {'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS]}}
                edge_model = ai_edge_torch.convert(self.obj_ptr_proj, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
                edge_model.export("model/mlp_"+model_id+".tflite")
            else:
                # tensorflow
                import glob
                images = glob.glob("./calibration/mlp/*.npz")
                if len(images) == 0:
                    raise Exception("Calibration data not found. Please run --mode calibration first.")

                def representative_dataset():
                    import numpy as np
                    for i in range(len(images)):
                        #data_0 = sam_output_token.numpy()

                        npz = np.load(images[i])
                        data_0 = npz["arr_0"]

                        yield [data_0]

                tfl_converter_flags = {
                    'optimizations': [tf.lite.Optimize.DEFAULT],
                    'representative_dataset': representative_dataset,
                    'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                    'inference_input_type': tf.int8,
                    'inference_output_type': tf.int8,
                }

                tfl_fullint_model = ai_edge_torch.convert(
                    self.obj_ptr_proj, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
                )

                tfl_fullint_model.export("model/mlp_"+model_id+".int8.tflite")

        if import_from_tflite:
            if self.mlp_tflite == None:
                int8_id = ""
                if tflite_int8:
                    int8_id = ".int8"
                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.mlp_tflite = ailia_tflite.Interpreter(model_path="model/mlp_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.mlp_tflite = tf.lite.Interpreter(model_path="model/mlp_"+model_id+int8_id+".tflite")
                self.mlp_tflite.allocate_tensors()

            input_details = self.mlp_tflite.get_input_details()
            output_details = self.mlp_tflite.get_output_details()

            if tflite_int8:
                self.mlp_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(sam_output_token.numpy(), input_details, 0))
                self.mlp_tflite.invoke()
                obj_ptr = self.get_output_tensor(self.mlp_tflite, output_details, 0)
            else:
                self.mlp_tflite.set_tensor(input_details[0]["index"], sam_output_token.numpy())
                self.mlp_tflite.invoke()
                obj_ptr = self.mlp_tflite.get_tensor(output_details[0]["index"])

            obj_ptr = torch.Tensor(obj_ptr)

        if not import_from_onnx and not import_from_tflite:
            obj_ptr = self.obj_ptr_proj(sam_output_token)

            if self.calibration:
                import os
                os.makedirs("./calibration/mlp/", exist_ok=True)
                import numpy as np
                np.savez("./calibration/mlp/"+str(self.calibration_cnt_mlp)+".npz",
                    sam_output_token.numpy()
                )
                self.calibration_cnt_mlp = self.calibration_cnt_mlp + 1

        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs, export_to_onnx, import_from_onnx, export_to_tflite, import_from_tflite, tflite_int8, model_id):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
                export_to_onnx=export_to_onnx,
                import_from_onnx=import_from_onnx,
                export_to_tflite=export_to_tflite,
                import_from_tflite=import_from_tflite,
                tflite_int8=tflite_int8,
                model_id=model_id
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out["vision_features"], backbone_out["vision_pos_enc"][0], backbone_out["vision_pos_enc"][1], backbone_out["vision_pos_enc"][2], backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1], backbone_out["backbone_fpn"][2]

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def call_obj_ptr_tpos_proj(self, obj_pos,
        export_to_onnx=False,
        import_from_onnx=False,
        export_to_tflite=False,
        import_from_tflite=False,
        tflite_int8=False,
        model_id=None):
        if export_to_onnx and not self.obj_ptr_tpos_proj_onnx_exported:
            print("x", obj_pos.shape)
            self.obj_ptr_tpos_proj_onnx_exported = True
            torch.onnx.export(
                self.obj_ptr_tpos_proj, (obj_pos), 'model/obj_ptr_tpos_proj_'+model_id+'.onnx',
                input_names=["x"],
                output_names=["x_out"],
                dynamic_axes={
                    'x': {0: 'n'},
                    'obj_ptr': {0: 'n'}
                },
                verbose=False, opset_version=17
            )

        if import_from_onnx:
            import onnxruntime
            if self.obj_ptr_tpos_proj_onnx == None:
                self.obj_ptr_tpos_proj_onnx  = onnxruntime.InferenceSession("model/obj_ptr_tpos_proj_"+model_id+".onnx")
            import numpy as np
            tpos = self.obj_ptr_tpos_proj_onnx.run(None, {"x":obj_pos.numpy()})[0]
            tpos = torch.Tensor(tpos)
        
        if export_to_tflite and not self.obj_ptr_tpos_proj_tflite_exported:
            self.obj_ptr_tpos_proj_tflite_exported = True
            import ai_edge_torch
            import tensorflow as tf
            sample_inputs = (obj_pos,)
            if not tflite_int8:
                tfl_converter_flags = {'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS]}}
                edge_model = ai_edge_torch.convert(self.obj_ptr_tpos_proj, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
                edge_model.export("model/obj_ptr_tpos_proj_"+model_id+".tflite")
            else:
                # tensorflow
                import glob
                images = glob.glob("./calibration/obj_ptr_tpos_proj/*.npz")
                if len(images) == 0:
                    raise Exception("Calibration data not found. Please run --mode calibration first.")

                def representative_dataset():
                    import numpy as np
                    for i in range(len(images)):
                        #data_0 = obj_pos.numpy()

                        npz = np.load(images[i])
                        data_0 = npz["arr_0"]

                        yield [data_0]
                
                tfl_converter_flags = {
                    'optimizations': [tf.lite.Optimize.DEFAULT],
                    'representative_dataset': representative_dataset,
                    'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                    'inference_input_type': tf.int8,
                    'inference_output_type': tf.int8,
                }

                tfl_fullint_model = ai_edge_torch.convert(
                    self.obj_ptr_tpos_proj, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
                )

                tfl_fullint_model.export("model/obj_ptr_tpos_proj_"+model_id+".int8.tflite")
            
        if import_from_tflite:
            if self.obj_ptr_tpos_proj_tflite == None:
                int8_id = ""
                if tflite_int8:
                    int8_id = ".int8"
                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.obj_ptr_tpos_proj_tflite = ailia_tflite.Interpreter(model_path="model/obj_ptr_tpos_proj_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.obj_ptr_tpos_proj_tflite = tf.lite.Interpreter(model_path="model/obj_ptr_tpos_proj_"+model_id+int8_id+".tflite")
                self.obj_ptr_tpos_proj_tflite.allocate_tensors()

            input_details = self.obj_ptr_tpos_proj_tflite.get_input_details()
            output_details = self.obj_ptr_tpos_proj_tflite.get_output_details()

            import numpy as np
            tpos = np.zeros((obj_pos.shape[0], 64))
            for i in range(obj_pos.shape[0]):
                if tflite_int8:
                    self.obj_ptr_tpos_proj_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(obj_pos[i:i+1,:].numpy(), input_details, 0))
                    self.obj_ptr_tpos_proj_tflite.invoke()
                    tpos[i:i+1,:] = self.get_output_tensor(self.obj_ptr_tpos_proj_tflite, output_details, 0)
                else:
                    self.obj_ptr_tpos_proj_tflite.set_tensor(input_details[0]["index"], obj_pos[i:i+1,:].numpy())
                    self.obj_ptr_tpos_proj_tflite.invoke()
                    tpos[i:i+1,:] = self.obj_ptr_tpos_proj_tflite.get_tensor(output_details[0]["index"])
            tpos = torch.Tensor(tpos)

        if not import_from_onnx and not import_from_tflite:
            tpos = self.obj_ptr_tpos_proj(obj_pos)

            if self.calibration:
                import os
                os.makedirs("./calibration/obj_ptr_tpos_proj/", exist_ok=True)
                import numpy as np
                for i in range(obj_pos.shape[0]):
                    np.savez("./calibration/obj_ptr_tpos_proj/"+str(self.calibration_cnt_obj_ptr_tpos_proj)+".npz",
                        obj_pos[i:i+1,:].numpy()
                    )
                    self.calibration_cnt_obj_ptr_tpos_proj = self.calibration_cnt_obj_ptr_tpos_proj + 1

        return tpos


    def format_input_tensor(self, tensor, input_details, idx):
        details = input_details[idx]
        dtype = details['dtype']
        if dtype == np.uint8 or dtype == np.int8:
            quant_params = details['quantization_parameters']
            input_tensor = tensor / quant_params['scales'] + quant_params['zero_points']
            if dtype == np.int8:
                input_tensor = input_tensor.clip(-128, 127)
            else:
                input_tensor = input_tensor.clip(0, 255)
            return input_tensor.astype(dtype)
        else:
            return tensor

    def get_output_tensor(self, interpreter, output_details, idx):
        details = output_details[idx]
        if details['dtype'] == np.uint8 or details['dtype'] == np.int8:
            quant_params = details['quantization_parameters']
            int_tensor = interpreter.get_tensor(details['index']).astype(np.int32)
            real_tensor = int_tensor - quant_params['zero_points']
            real_tensor = real_tensor.astype(np.float32) * quant_params['scales']
        else:
            real_tensor = interpreter.get_tensor(details['index'])
        return real_tensor


    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        export_to_onnx=False,
        import_from_onnx=False,
        export_to_tflite=False,
        import_from_tflite=False,
        tflite_int8=False,
        model_id=None
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        #obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = self.call_obj_ptr_tpos_proj(obj_pos, export_to_onnx=export_to_onnx, import_from_onnx=import_from_onnx, export_to_tflite=export_to_tflite, import_from_tflite=import_from_tflite, tflite_int8=tflite_int8, model_id=model_id)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        # 標準の実装ではforwardの中でweightが確保されるが、エクスポート時に固定するために先に確保する
        self.memory_attention.allocate_rope_attention_weight(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            image_size=self.image_size,
        )

        # 4096の倍数のRoPEAttentionが適用される部分と手協されない部分を事前に分割する
        # 動的なsliceがdynamoでエラーになるため
        memory_1 = memory[:-num_obj_ptr_tokens,:,:]
        memory_2 = memory[-num_obj_ptr_tokens:,:,:]
        memory_pos_embed_1 = memory_pos_embed[:-num_obj_ptr_tokens,:,:]
        memory_pos_embed_2 = memory_pos_embed[-num_obj_ptr_tokens:,:,:]

        # Static Shapeに変換するために、定数フレームに変換する
        convert_to_static_shape = export_to_tflite or import_from_tflite or self.calibration
        if convert_to_static_shape:
            max_num_frames = 8
            memory_1_pad = torch.zeros(max_num_frames * self.sam_image_embedding_size * self.sam_image_embedding_size, memory_1.shape[1], memory_1.shape[2])
            memory_1_pad[:memory_1.shape[0],:,:] = memory_1
            memory_pos_embed_1_pad = torch.zeros(max_num_frames * self.sam_image_embedding_size * self.sam_image_embedding_size, memory_pos_embed_1.shape[1], memory_pos_embed_1.shape[2])
            memory_pos_embed_1_pad[:memory_pos_embed_1.shape[0],:,:] = memory_pos_embed_1
            memory_2_pad = torch.zeros(max_num_frames * 4, memory_2.shape[1], memory_2.shape[2])
            memory_2_pad[:memory_2.shape[0],:,:] = memory_2
            memory_pos_embed_2_pad = torch.zeros(max_num_frames * 4, memory_pos_embed_2.shape[1], memory_pos_embed_2.shape[2])
            memory_pos_embed_2_pad[:memory_pos_embed_2.shape[0],:,:] = memory_pos_embed_2
            memory_2_pad = torch.zeros(max_num_frames * 4, memory_2.shape[1], memory_2.shape[2])
            memory_2_pad[:memory_2.shape[0],:,:] = memory_2
            attention_mask_1 = torch.zeros(max_num_frames * self.sam_image_embedding_size * self.sam_image_embedding_size, memory_1.shape[1], dtype=torch.bool)
            attention_mask_2 = torch.zeros(max_num_frames * 4, memory_2.shape[1], dtype=torch.bool)
            attention_mask_1[:memory_1.shape[0],:] = True
            attention_mask_2[:memory_2.shape[0],:] = True
            memory_1 = memory_1_pad
            memory_2 = memory_2_pad
            memory_pos_embed_1 = memory_pos_embed_1_pad
            memory_pos_embed_2 = memory_pos_embed_2_pad
        else:
            attention_mask_1 = torch.zeros(memory_1.shape[0], memory_1.shape[1], dtype=torch.bool)
            attention_mask_2 = torch.zeros(memory_2.shape[0], memory_2.shape[1], dtype=torch.bool)
            attention_mask_1[:memory_1.shape[0],:] = True
            attention_mask_2[:memory_2.shape[0],:] = True

        if self.debug:
            print("memory attention shape")
            print("curr", current_vision_feats[0].shape)
            print("memory", memory.shape)
            print("curr_pos", current_vision_pos_embeds[0].shape)
            print("memory_pos", memory_pos_embed.shape)
            print("num_obj_ptr_tokens", num_obj_ptr_tokens)

        if export_to_onnx and not self.memory_attention_onnx_exported:
            self.memory_attention_onnx_exported = True
            #print("current_vision_feats", current_vision_feats[0].shape, current_vision_feats[0].dtype)
            #print("memory", memory.shape, memory.dtype)
            #print("current_vision_pos_embeds", current_vision_pos_embeds[0].shape, current_vision_pos_embeds[0].dtype)
            #print("memory_pos_embed", memory_pos_embed.shape, memory_pos_embed.dtype)
            #print("num_obj_ptr_tokens", num_obj_ptr_tokens)
            torch.onnx.export(
                self.memory_attention, (current_vision_feats[0], memory_1, memory_2, current_vision_pos_embeds[0], memory_pos_embed_1, memory_pos_embed_2, attention_mask_1, attention_mask_2), 'model/memory_attention_'+model_id+'.onnx',
                input_names=["curr", "memory_1", "memory_2", "curr_pos", "memory_pos_1", "memory_pos_2", "attention_mask_1", "attention_mask_2"],
                output_names=["pix_feat"],
                dynamic_axes={
                    'memory_1': {0: 'n_1'},
                    'memory_2': {0: 'n_2'},
                    'memory_pos_1': {0: 'n_1'},
                    'memory_pos_2': {0: 'n_2'},
                    'attention_mask_1': {0: 'n_1'},
                    'attention_mask_2': {0: 'n_2'}
                },
                verbose=False, opset_version=17
            )
            #export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
            #onnx_program  =torch.onnx.dynamo_export(
            #    self.memory_attention, current_vision_feats[0], memory_1, memory_2, current_vision_pos_embeds[0], memory_pos_embed_1, memory_pos_embed_2, export_options=export_options
            #)
            #onnx_program.save('model/memory_attention_'+model_id+'.onnx')

        if import_from_onnx:
            if self.debug:
                print("begin memory attention onnx")
            import onnxruntime
            if self.memory_attention_onnx == None:
                self.memory_attention_onnx = onnxruntime.InferenceSession("model/memory_attention_"+model_id+".onnx")
            import numpy as np
            #num_obj_ptr_tokens_numpy = np.array((num_obj_ptr_tokens)).astype(np.int64)
            #print("curr", np.sum(current_vision_feats[0].numpy()))
            #print("memory", np.sum(memory.numpy()))
            #print("curr_pos", np.sum(current_vision_pos_embeds[0].numpy()))
            #print("memory_pos", np.sum(memory_pos_embed.numpy()))
            #print("num_obj_ptr_tokens", np.sum(num_obj_ptr_tokens_numpy))

            pix_feat_with_mem = self.memory_attention_onnx.run(None, {"curr":current_vision_feats[0].numpy(), "memory_1":memory_1.numpy(), "memory_2":memory_2.numpy(), "curr_pos":current_vision_pos_embeds[0].numpy(), "memory_pos_1":memory_pos_embed_1.numpy(), "memory_pos_2":memory_pos_embed_2.numpy(), "attention_mask_1":attention_mask_1.numpy(), "attention_mask_2":attention_mask_2.numpy()})
            pix_feat_with_mem = torch.Tensor(pix_feat_with_mem[0])
        
        tflite_int8_memory_attention = tflite_int8
        #tflite_int8_memory_attention = False
        
        if export_to_tflite and not self.memory_attention_tflite_exported:
            self.memory_attention_tflite_exported = True
            import ai_edge_torch
            import tensorflow as tf
            sample_inputs = (current_vision_feats[0], memory_1, memory_2, current_vision_pos_embeds[0], memory_pos_embed_1, memory_pos_embed_2, attention_mask_1, attention_mask_2)
            tfl_converter_flags = {'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS]}}
            if not tflite_int8_memory_attention:
                edge_model = ai_edge_torch.convert(self.memory_attention, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
                edge_model.export("model/memory_attention_"+model_id+".tflite")
            else:
                if tflite_int8 == "mixed": # Memory Attentionはattn_maskをint8に量子化すると精度が出ないため、torchで量子化する
                    # torch
                    from ai_edge_torch.quantize import pt2e_quantizer
                    from ai_edge_torch.quantize import quant_config
                    from torch.ao.quantization import quantize_pt2e

                    # torch quantization
                    from ai_edge_torch.quantize import pt2e_quantizer
                    from ai_edge_torch.quantize import quant_config
                    from torch.ao.quantization import quantize_pt2e

                    # quantize transpose_conv
                    from ai_edge_torch.quantize.pt2e_quantizer_utils import get_input_act_qspec, get_weight_qspec, get_bias_qspec, get_output_act_qspec
                    from torch.ao.quantization.quantizer import QuantizationAnnotation, QuantizationSpec
                    from torch.ao.quantization.observer import PlaceholderObserver, HistogramObserver

                    quantization_config = pt2e_quantizer.get_symmetric_quantization_config(is_dynamic=False, is_per_channel=True)

                    class PT2EQuantizer2(pt2e_quantizer.PT2EQuantizer):
                        def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                            super().annotate(model)

                            # 出力の量子化
                            target_node = None
                            for n in model.graph.nodes:
                                if str(n.target) == "output":
                                    target_node = n.args[0][0]

                            for n in model.graph.nodes:
                                if n == target_node:
                                    input_qspec_map = {}
                                    input_qspec_map[n.args[0]] = get_input_act_qspec(quantization_config)
                                    n.meta["quantization_annotation"] = QuantizationAnnotation(
                                        input_qspec_map=input_qspec_map,
                                        output_qspec=get_output_act_qspec(quantization_config),
                                        _annotated=True,
                                    )

                            # scaled dot product attentionのsymmetric quantization
                            for n in model.graph.nodes:
                                print(n.name, n.target, n.args)
                                if n.target in [torch.ops.aten.scaled_dot_product_attention.default]:
                                    print(n.name)
                                    print(n.target)
                                    print(n.meta)
                                    print(n.args)
                                    input_qspec_map = {}

                                    spec = QuantizationSpec(
                                        dtype=torch.int8,
                                        quant_min=-128,
                                        quant_max=127,
                                        qscheme=torch.per_tensor_symmetric, # per_tensor_affine
                                        observer_or_fake_quant_ctr=HistogramObserver.with_args(
                                            reduce_range=True,
                                            quant_min=-128,
                                            quant_max=127
                                        )
                                    )

                                    for i in range(3):  # query(0), key(1), value(2)
                                        input_qspec_map[n.args[i]] = spec
                                    
                                    if len(n.args) >= 4:
                                        float_spec = QuantizationSpec(
                                            dtype=torch.float32,
                                            observer_or_fake_quant_ctr=PlaceholderObserver.with_args(eps=2**-12)
                                        )
                                        input_qspec_map[n.args[3]] = float_spec
                                        
                                    n.meta["quantization_annotation"] = QuantizationAnnotation(
                                        input_qspec_map=input_qspec_map,
                                        output_qspec=get_output_act_qspec(quantization_config),
                                        _annotated=True,
                                    )

                            return model

                    if False:
                        quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
                            quantization_config
                        )
                    else:
                        quantizer = PT2EQuantizer2().set_global(
                            quantization_config
                        )
                    model = torch._export.capture_pre_autograd_graph(self.memory_attention, sample_inputs)
                    model = quantize_pt2e.prepare_pt2e(model, quantizer)

                    import glob
                    images = glob.glob("./calibration/memory_attention/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    import numpy as np
                    for i in range(len(images)):
                        npz = np.load(images[i])
                        data_0 = torch.tensor(npz["arr_0"])
                        data_1 = torch.tensor(npz["arr_1"])
                        data_2 = torch.tensor(npz["arr_2"])
                        data_3 = torch.tensor(npz["arr_3"])
                        data_4 = torch.tensor(npz["arr_4"])
                        data_5 = torch.tensor(npz["arr_5"])
                        data_6 = torch.tensor(npz["arr_6"])
                        data_7 = torch.tensor(npz["arr_7"])
                        model(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7)

                    #model(current_vision_feats[0], memory_1, memory_2, current_vision_pos_embeds[0], memory_pos_embed_1, memory_pos_embed_2, attention_mask_1, attention_mask_2)
                    
                    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

                    with_quantizer = ai_edge_torch.convert(
                        model,
                        sample_inputs,
                        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
                    )
                    with_quantizer.export("model/memory_attention_"+model_id+".mixed.tflite")
                else:
                    # tensorflow
                    import glob
                    images = glob.glob("./calibration/memory_attention/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    def representative_dataset():
                        import numpy as np
                        for i in range(len(images)):
                            #data_3 = current_vision_feats[0].numpy()
                            #data_6 = memory_1.numpy()
                            #data_1 = memory_2.numpy()
                            #data_2 = current_vision_pos_embeds[0].numpy()
                            #data_5 = memory_pos_embed_1.numpy()
                            #data_0 = memory_pos_embed_2.numpy()
                            #data_4 = attention_mask_1.numpy()
                            #data_7 = attention_mask_2.numpy()

                            npz = np.load(images[i])
                            data_3 = npz["arr_0"]
                            data_6 = npz["arr_1"]
                            data_1 = npz["arr_2"]
                            data_2 = npz["arr_3"]
                            data_5 = npz["arr_4"]
                            data_0 = npz["arr_5"]
                            data_4 = npz["arr_6"]
                            data_7 = npz["arr_7"]

                            yield [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7]
                    
                    tfl_converter_flags = {
                        'optimizations': [tf.lite.Optimize.DEFAULT],
                        'representative_dataset': representative_dataset,
                        'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                        'inference_input_type': tf.int8,
                        'inference_output_type': tf.int8,
                    }

                    tfl_fullint_model = ai_edge_torch.convert(
                        self.memory_attention, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
                    )

                    tfl_fullint_model.export("model/memory_attention_"+model_id+".int8.tflite")

            if False: # Dynamic
                n_1 = torch.export.Dim("n_1", min=1, max=256)
                n_4096 = n_1 * 4096
                n_2 = torch.export.Dim("n_2", min=1, max=256)
                n_4 = n_2 * 4
                dynamic_shapes={
                    'curr': None,
                    'memory_1': {0: n_4096},
                    'memory_2': {0: n_4},
                    'curr_pos': None,
                    'memory_pos_1': {0: n_4096},
                    'memory_pos_2': {0: n_4},
                    'attention_mask_1': {0: n_4096},
                    'attention_mask_2': {0: n_4}
                }
                edge_model = ai_edge_torch.convert(self.memory_attention, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags, dynamic_shapes=dynamic_shapes)
                edge_model.export("model/memory_attention_"+model_id+".tflite")

        if import_from_tflite:
            if self.debug:
                print("begin memory attention tflite")
            if self.memory_attention_tflite == None:
                int8_id = ""
                if tflite_int8_memory_attention:
                    int8_id = "." + tflite_int8_memory_attention

                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.memory_attention_tflite = ailia_tflite.Interpreter(model_path="model/memory_attention_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.memory_attention_tflite = tf.lite.Interpreter(model_path="model/memory_attention_"+model_id+int8_id+".tflite")
                self.memory_attention_tflite.allocate_tensors()
                input_details = self.memory_attention_tflite.get_input_details()
                #self.memory_attention_tflite.resize_tensor_input(
                #    input_details[5]["index"], 
                #    [memory_1.shape[0], 1, 64]
                #)
                #self.memory_attention_tflite.resize_tensor_input(
                #    input_details[1]["index"], 
                #    [memory_2.shape[0], 1, 64]
                #)
                #self.memory_attention_tflite.resize_tensor_input(
                #    input_details[4]["index"], 
                #    [memory_pos_embed_1.shape[0], 1, 64]
                #)
                #self.memory_attention_tflite.resize_tensor_input(
                #    input_details[0]["index"], 
                #    [memory_pos_embed_2.shape[0], 1, 64]
                #)
                self.memory_attention_tflite.allocate_tensors()

            input_details = self.memory_attention_tflite.get_input_details()
            output_details = self.memory_attention_tflite.get_output_details()

            if tflite_int8_memory_attention:
                self.memory_attention_tflite.set_tensor(input_details[3]["index"], self.format_input_tensor(current_vision_feats[0].numpy(), input_details, 3))
                self.memory_attention_tflite.set_tensor(input_details[6]["index"], self.format_input_tensor(memory_1.numpy(), input_details, 6))
                self.memory_attention_tflite.set_tensor(input_details[1]["index"], self.format_input_tensor(memory_2.numpy(), input_details, 1))
                self.memory_attention_tflite.set_tensor(input_details[2]["index"], self.format_input_tensor(current_vision_pos_embeds[0].numpy(), input_details, 2))
                self.memory_attention_tflite.set_tensor(input_details[5]["index"], self.format_input_tensor(memory_pos_embed_1.numpy(), input_details, 5))
                self.memory_attention_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(memory_pos_embed_2.numpy(), input_details, 0))
                self.memory_attention_tflite.set_tensor(input_details[4]["index"], self.format_input_tensor(attention_mask_1.numpy(), input_details, 4))
                self.memory_attention_tflite.set_tensor(input_details[7]["index"], self.format_input_tensor(attention_mask_2.numpy(), input_details, 7))

                self.memory_attention_tflite.invoke()

                pix_feat_with_mem = self.get_output_tensor(self.memory_attention_tflite, output_details, 0)
            else:
                self.memory_attention_tflite.set_tensor(input_details[3]["index"], current_vision_feats[0].numpy())
                self.memory_attention_tflite.set_tensor(input_details[6]["index"], memory_1.numpy())
                self.memory_attention_tflite.set_tensor(input_details[1]["index"], memory_2.numpy())
                self.memory_attention_tflite.set_tensor(input_details[2]["index"], current_vision_pos_embeds[0].numpy())
                self.memory_attention_tflite.set_tensor(input_details[5]["index"], memory_pos_embed_1.numpy())
                self.memory_attention_tflite.set_tensor(input_details[0]["index"], memory_pos_embed_2.numpy())
                self.memory_attention_tflite.set_tensor(input_details[4]["index"], attention_mask_1.numpy())
                self.memory_attention_tflite.set_tensor(input_details[7]["index"], attention_mask_2.numpy())

                self.memory_attention_tflite.invoke()

                pix_feat_with_mem = self.memory_attention_tflite.get_tensor(output_details[0]["index"])
            pix_feat_with_mem = torch.Tensor(pix_feat_with_mem)

        if not import_from_onnx and not import_from_tflite:
            #print("begin memory attention torch")
            #print("current_vision_feats", current_vision_feats[0].shape)
            #print("current_vision_pos_embeds", current_vision_pos_embeds[0].shape)
            #print("memory", memory.shape)
            #print("memory_pos_embed", memory_pos_embed.shape)
            pix_feat_with_mem = self.memory_attention(
                curr=current_vision_feats,
                memory_1=memory_1,
                memory_2=memory_2,
                curr_pos=current_vision_pos_embeds,
                memory_pos_1=memory_pos_embed_1,
                memory_pos_2=memory_pos_embed_2,
                attention_mask_1=attention_mask_1,
                attention_mask_2=attention_mask_2,
            )

            if self.calibration:
                import os
                os.makedirs("./calibration/memory_attention/", exist_ok=True)
                import numpy as np
                np.savez("./calibration/memory_attention/"+str(self.calibration_cnt_memory_attention)+".npz",
                    current_vision_feats[0].numpy(),
                    memory_1.numpy(),
                    memory_2.numpy(),
                    current_vision_pos_embeds[0].numpy(),
                    memory_pos_embed_1.numpy(),
                    memory_pos_embed_2.numpy(),
                    attention_mask_1.numpy(),
                    attention_mask_2.numpy(),
                )
                self.calibration_cnt_memory_attention = self.calibration_cnt_memory_attention + 1


        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        export_to_onnx = False,
        import_from_onnx = False,
        export_to_tflite = False,
        import_from_tflite = False,
        tflite_int8 = False,
        model_id = None
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        if export_to_onnx and not self.memory_encoder_onnx_exported:
            self.memory_encoder_onnx_exported = True
            torch.onnx.export(
                self.memory_encoder, (pix_feat, mask_for_mem), 'model/memory_encoder_'+model_id+'.onnx',
                input_names=["pix_feat", "masks"],
                output_names=["vision_features", "vision_pos_enc"],
                verbose=False, opset_version=17
            )

        if import_from_onnx:
            if self.debug:
                print("begin memory encoder onnx")
            import onnxruntime
            if self.memory_encoder_onnx == None:
                self.memory_encoder_onnx = onnxruntime.InferenceSession("model/memory_encoder_"+model_id+".onnx")
            vision_features, vision_pos_enc = self.memory_encoder_onnx.run(None, {"pix_feat":pix_feat.numpy(), "masks":mask_for_mem.numpy()})
            vision_features = torch.Tensor(vision_features)
            vision_pos_enc = torch.Tensor(vision_pos_enc)

        tflite_int8_memory_encoder = tflite_int8
        #if tflite_int8_memory_encoder == "mixed":
        #    tflite_int8_memory_encoder = "int8" # 暫定

        self.memory_encoder.prepare_position_encoding(pix_feat)

        if export_to_tflite and not self.memory_encoder_tflite_exported:
            self.memory_encoder_tflite_exported = True

            import ai_edge_torch
            import tensorflow as tf

            sample_inputs = (pix_feat, mask_for_mem)

            if not tflite_int8_memory_encoder:
                tfl_converter_flags = {'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS]}}
                edge_model = ai_edge_torch.convert(self.memory_encoder, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
                edge_model.export("model/memory_encoder_"+model_id+".tflite")
            else:
                if tflite_int8_memory_encoder == "mixed":
                    # torch
                    from ai_edge_torch.quantize import pt2e_quantizer
                    from ai_edge_torch.quantize import quant_config
                    from torch.ao.quantization import quantize_pt2e

                    quantization_config = pt2e_quantizer.get_symmetric_quantization_config(is_dynamic=False, is_per_channel=True)

                    if False:
                        quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
                            pt2e_quantizer.get_symmetric_quantization_config(is_dynamic=False, is_per_channel=True)
                        )
                    else:
                        import ai_edge_torch

                        # torch quantization
                        from ai_edge_torch.quantize import pt2e_quantizer
                        from ai_edge_torch.quantize import quant_config
                        from torch.ao.quantization import quantize_pt2e

                        # quantize transpose_conv
                        from ai_edge_torch.quantize.pt2e_quantizer_utils import get_input_act_qspec, get_output_act_qspec
                        from torch.ao.quantization.quantizer import QuantizationAnnotation

                        class PT2EQuantizer2(pt2e_quantizer.PT2EQuantizer):
                            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                                super().annotate(model)

                                for n in model.graph.nodes:
                                    if n.target in [torch.ops.aten.gelu.default]:
                                        input_qspec_map = {}

                                        input_qspec_map[n.args[0]] = get_input_act_qspec(quantization_config)

                                        n.meta["quantization_annotation"] = QuantizationAnnotation(
                                            input_qspec_map=input_qspec_map,
                                            output_qspec=get_output_act_qspec(quantization_config),
                                            _annotated=True,
                                        )

                                return model

                        quantizer = PT2EQuantizer2().set_global(
                            quantization_config
                        )

                    model = torch._export.capture_pre_autograd_graph(self.memory_encoder, sample_inputs)
                    model = quantize_pt2e.prepare_pt2e(model, quantizer)

                    import glob
                    images = glob.glob("./calibration/memory_encoder/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    import numpy as np
                    for i in range(len(images)):
                        npz = np.load(images[i])
                        data_0 = torch.tensor(npz["arr_0"])
                        data_1 = torch.tensor(npz["arr_1"])
                        model(data_0, data_1)

                    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)
                    
                    with_quantizer = ai_edge_torch.convert(
                        model,
                        sample_inputs,
                        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
                    )
                    with_quantizer.export("model/memory_encoder_"+model_id+".mixed.tflite")

                    #exit()
                else:
                    # tensorflow
                    import glob
                    images = glob.glob("./calibration/memory_encoder/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    def representative_dataset():
                        import numpy as np
                        for i in range(len(images)):
                            npz = np.load(images[i])
                            data_0 = npz["arr_0"]
                            data_1 = npz["arr_1"]
                            #data_0 = pix_feat.numpy()
                            #data_1 = mask_for_mem.numpy()
                            yield [data_0, data_1]

                    tfl_converter_flags = {
                        'optimizations': [tf.lite.Optimize.DEFAULT],
                        'representative_dataset': representative_dataset,
                        'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                        'inference_input_type': tf.int8,
                        'inference_output_type': tf.int8,
                    }

                    tfl_fullint_model = ai_edge_torch.convert(
                        self.memory_encoder, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
                    )

                    tfl_fullint_model.export("model/memory_encoder_"+model_id+".int8.tflite")
            
        if import_from_tflite:
            if self.debug:
                print("begin memory encoder tflite")
            if self.memory_encoder_tflite == None:
                int8_id = ""
                if tflite_int8_memory_encoder:
                    int8_id = "." + tflite_int8_memory_encoder
                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.memory_encoder_tflite = ailia_tflite.Interpreter(model_path="model/memory_encoder_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.memory_encoder_tflite = tf.lite.Interpreter(model_path="model/memory_encoder_"+model_id+int8_id+".tflite")
                self.memory_encoder_tflite.allocate_tensors()

            input_details = self.memory_encoder_tflite.get_input_details()
            output_details = self.memory_encoder_tflite.get_output_details()

            if tflite_int8_memory_encoder:
                self.memory_encoder_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(pix_feat.numpy(), input_details, 0))
                self.memory_encoder_tflite.set_tensor(input_details[1]["index"], self.format_input_tensor(mask_for_mem.numpy(), input_details, 1))
                self.memory_encoder_tflite.invoke()

                vision_features = self.get_output_tensor(self.memory_encoder_tflite, output_details, 1)
                vision_pos_enc = self.get_output_tensor(self.memory_encoder_tflite, output_details, 0)
            else:
                self.memory_encoder_tflite.set_tensor(input_details[0]["index"], pix_feat.numpy())
                self.memory_encoder_tflite.set_tensor(input_details[1]["index"], mask_for_mem.numpy())
                self.memory_encoder_tflite.invoke()

                vision_features = self.memory_encoder_tflite.get_tensor(output_details[1]["index"])
                vision_pos_enc = self.memory_encoder_tflite.get_tensor(output_details[0]["index"])
            vision_features = torch.Tensor(vision_features)
            vision_pos_enc = torch.Tensor(vision_pos_enc)

        if not import_from_onnx and not import_from_tflite:
            if self.debug:
                print("begin memory encoder torch")
            vision_features, vision_pos_enc = self.memory_encoder(
                pix_feat, mask_for_mem#, skip_mask_sigmoid=True  # sigmoid already applied (fixed to constant)
            )

            if self.calibration:
                import os
                os.makedirs("./calibration/memory_encoder/", exist_ok=True)
                import numpy as np
                np.savez("./calibration/memory_encoder/"+str(self.calibration_cnt_memory_encoder)+".npz",
                    pix_feat.numpy(),
                    mask_for_mem.numpy()
                )
                self.calibration_cnt_memory_encoder = self.calibration_cnt_memory_encoder + 1

        maskmem_features = vision_features
        maskmem_pos_enc = [vision_pos_enc]

        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
        import_from_onnx=False,
        export_to_onnx=False,
        import_from_tflite=False,
        export_to_tflite=False,
        tflite_int8=False,
        model_id=None
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs,
                export_to_onnx=export_to_onnx,
                import_from_onnx=import_from_onnx,
                export_to_tflite=export_to_tflite,
                import_from_tflite=import_from_tflite,
                tflite_int8=tflite_int8,
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                export_to_onnx=export_to_onnx,
                import_from_onnx=import_from_onnx,
                export_to_tflite=export_to_tflite,
                import_from_tflite=import_from_tflite,
                tflite_int8=tflite_int8,
                model_id=model_id
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                export_to_onnx=export_to_onnx,
                import_from_onnx=import_from_onnx,
                export_to_tflite=export_to_tflite,
                import_from_tflite=import_from_tflite,
                tflite_int8=tflite_int8,
                model_id=model_id
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
        export_to_onnx,
        import_from_onnx,
        export_to_tflite,
        import_from_tflite,
        tflite_int8,
        model_id
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                export_to_onnx=export_to_onnx,
                import_from_onnx=import_from_onnx,
                export_to_tflite=export_to_tflite,
                import_from_tflite=import_from_tflite,
                tflite_int8=tflite_int8,
                model_id=model_id
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        # ONNX Export
        export_to_onnx=False,
        import_from_onnx=False,
        export_to_tflite=False,
        import_from_tflite=False,
        tflite_int8=False,
        model_id=None
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            export_to_onnx=export_to_onnx,
            import_from_onnx=import_from_onnx,
            export_to_tflite=export_to_tflite,
            import_from_tflite=import_from_tflite,
            tflite_int8=tflite_int8,
            model_id=model_id
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
            export_to_onnx=export_to_onnx,
            import_from_onnx=import_from_onnx,
            export_to_tflite=export_to_tflite,
            import_from_tflite=import_from_tflite,
            tflite_int8=tflite_int8,
            model_id=model_id
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
