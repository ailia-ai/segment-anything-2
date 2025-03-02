# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

import onnxruntime

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor:
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        image_size=1024,
        calibration=False,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=sam_model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (sam_model.image_size // 4, sam_model.image_size // 4),
            (sam_model.image_size // 8, sam_model.image_size // 8),
            (sam_model.image_size // 16, sam_model.image_size // 16),
        ]

        # debug
        self.debug = False
         
        # onnx
        self.image_encoder_onnx = None
        self.prompt_encoder_onnx = None
        self.mask_decoder_onnx = None

        # tflite
        self.image_encoder_tflite = None
        self.prompt_encoder_tflite = None
        self.mask_decoder_tflite = None

        # calibration
        self.calibration = calibration
        self.calibration_cnt_image_encoder = 0
        self.calibration_cnt_mask_decoder = 0
        self.calibration_cnt_prompt_encoder = 0

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

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

    @torch.no_grad()
    def set_image(
        self,
        image: Union[np.ndarray, Image],
        export_to_onnx = False,
        export_to_tflite = False,
        import_from_onnx = False,
        import_from_tflite = False,
        tflite_int8=False,
        model_id=None
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")

        if export_to_onnx:
            #print("input_image", input_image.shape)
            self.model.forward = self.model.forward_image
            torch.onnx.export(
                self.model, (input_image), 'model/image_encoder_'+model_id+'.onnx',
                input_names=["input_image"],
                output_names=["vision_features", "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
                verbose=False, opset_version=17
            )
        
        if import_from_onnx:
            if self.image_encoder_onnx == None:
                self.image_encoder_onnx = onnxruntime.InferenceSession("model/image_encoder_"+model_id+".onnx")
            vision_features, vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = self.image_encoder_onnx.run(None, {"input_image":input_image.numpy()})
            if self.debug:
                print("vision_features", vision_features.shape)
                print("vision_pos_enc_0", vision_pos_enc_0.shape)
                print("vision_pos_enc_1", vision_pos_enc_1.shape)
                print("vision_pos_enc_2", vision_pos_enc_2.shape)
                print("backbone_fpn_0", backbone_fpn_0.shape)
                print("backbone_fpn_1", backbone_fpn_1.shape)
                print("backbone_fpn_2", backbone_fpn_2.shape)

        if export_to_tflite:
            import ai_edge_torch
            import tensorflow as tf
            sample_inputs = (input_image,)
            self.model.forward = self.model.forward_image

            if not tflite_int8:
                tfl_converter_flags = {'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS]}}
                edge_model = ai_edge_torch.convert(self.model, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
                edge_model.export("model/image_encoder_"+model_id+".tflite")
            
            if tflite_int8:
                if tflite_int8 == "mixed":
                    # torch quantization
                    from ai_edge_torch.quantize import pt2e_quantizer
                    from ai_edge_torch.quantize import quant_config
                    from torch.ao.quantization import quantize_pt2e

                    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
                        pt2e_quantizer.get_symmetric_quantization_config(is_dynamic=False, is_per_channel=True)
                    )
                    model = torch._export.capture_pre_autograd_graph(self.model, sample_inputs)
                    model = quantize_pt2e.prepare_pt2e(model, quantizer)

                    import glob
                    images = glob.glob("./calibration/image_encoder/*.npy")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    for i in range(len(images)):
                        data = torch.tensor(np.load(images[i]))
                        model(data) # calibration (you need to edit reset_histogram function)

                    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

                    with_quantizer = ai_edge_torch.convert(
                        model,
                        sample_inputs,
                        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
                    )
                    with_quantizer.export("model/image_encoder_"+model_id+".mixed.tflite")
                else:
                    # tensorflow quantization
                    import glob
                    images = glob.glob("./calibration/image_encoder/*.npy")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    def representative_dataset():
                        for i in range(len(images)):
                            #data = input_image.numpy()
                            data = np.load(images[i])
                            yield [data.astype(np.float32)]
                    
                    tfl_converter_flags = {
                        'optimizations': [tf.lite.Optimize.DEFAULT],
                        'representative_dataset': representative_dataset,
                        'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                        'inference_input_type': tf.int8,
                        'inference_output_type': tf.int8,
                    }

                    tfl_fullint_model = ai_edge_torch.convert(
                        self.model, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
                    )

                    tfl_fullint_model.export("model/image_encoder_"+model_id+".int8.tflite")

        if import_from_tflite:
            if self.image_encoder_tflite == None:
                int8_id = ""
                if tflite_int8:
                    int8_id = "." + tflite_int8
                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.image_encoder_tflite = ailia_tflite.Interpreter(model_path="model/image_encoder_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.image_encoder_tflite = tf.lite.Interpreter(model_path="model/image_encoder_"+model_id+int8_id+".tflite")
                self.image_encoder_tflite.allocate_tensors()

            input_details = self.image_encoder_tflite.get_input_details()
            output_details = self.image_encoder_tflite.get_output_details()

            if tflite_int8:
                self.image_encoder_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(input_image.numpy(), input_details, 0))

                self.image_encoder_tflite.invoke()

                vision_features = self.get_output_tensor(self.image_encoder_tflite, output_details, 4)
                vision_pos_enc_0 = self.get_output_tensor(self.image_encoder_tflite, output_details, 1)
                vision_pos_enc_1 = self.get_output_tensor(self.image_encoder_tflite, output_details, 5)
                vision_pos_enc_2 = self.get_output_tensor(self.image_encoder_tflite, output_details, 3)
                backbone_fpn_0 = self.get_output_tensor(self.image_encoder_tflite, output_details, 0)
                backbone_fpn_1 = self.get_output_tensor(self.image_encoder_tflite, output_details, 2)
                backbone_fpn_2 = self.get_output_tensor(self.image_encoder_tflite, output_details, 6)
            else:
                self.image_encoder_tflite.set_tensor(input_details[0]["index"], input_image.numpy())
                self.image_encoder_tflite.invoke()

                vision_features = self.image_encoder_tflite.get_tensor(output_details[4]["index"])
                vision_pos_enc_0 = self.image_encoder_tflite.get_tensor(output_details[1]["index"])
                vision_pos_enc_1 = self.image_encoder_tflite.get_tensor(output_details[5]["index"])
                vision_pos_enc_2 = self.image_encoder_tflite.get_tensor(output_details[3]["index"])
                backbone_fpn_0 = self.image_encoder_tflite.get_tensor(output_details[0]["index"])
                backbone_fpn_1 = self.image_encoder_tflite.get_tensor(output_details[2]["index"])
                backbone_fpn_2 = self.image_encoder_tflite.get_tensor(output_details[6]["index"])

        if not import_from_onnx and not import_from_tflite:
            vision_features, vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = self.model.forward_image(input_image)
            if self.calibration:
                import os
                os.makedirs("./calibration/image_encoder/", exist_ok=True)
                np.save("./calibration/image_encoder/"+str(self.calibration_cnt_image_encoder)+".npy", input_image)
                self.calibration_cnt_image_encoder = self.calibration_cnt_image_encoder + 1


        backbone_out = {"vision_features":torch.Tensor(vision_features),
                        "vision_pos_enc":[torch.Tensor(vision_pos_enc_0), torch.Tensor(vision_pos_enc_1), torch.Tensor(vision_pos_enc_2)],
                        "backbone_fpn":[torch.Tensor(backbone_fpn_0), torch.Tensor(backbone_fpn_1), torch.Tensor(backbone_fpn_2)]}

        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        export_to_onnx=False,
        export_to_tflite=False,
        import_from_onnx = False,
        import_from_tflite = False,
        tflite_int8=False,
        model_id=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
            export_to_onnx=export_to_onnx,
            export_to_tflite=export_to_tflite,
            import_from_onnx=import_from_onnx,
            import_from_tflite=import_from_tflite,
            tflite_int8=tflite_int8,
            model_id=model_id
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
        export_to_onnx = False,
        export_to_tflite = False,
        tflite_int8 = False,
        import_from_onnx = False,
        import_from_tflite = False,
        model_id = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        # New data for onnx
        if concat_points is None:
            raise ("concat points must be exists") # Noneの場合はtensorサイズが0のテンソルを返さないといけないためwhereで組めない
        if mask_input is None:
            mask_input_dummy = torch.Tensor(np.zeros((1, self.model.image_size // 4, self.model.image_size // 4)))
            masks_enable = torch.tensor([0], dtype=torch.int) # boolだとonnxへのエクスポートのwhereでエラーになる
        else:
            mask_input_dummy = mask_input
            masks_enable = torch.tensor([1], dtype=torch.int)
        
        self.model.sam_prompt_encoder.generate_dense_pe()

        if export_to_onnx:
            #print("concat_points", concat_points.shape)
            #print("mask_input", mask_input.shape)
            torch.onnx.export(
                self.model.sam_prompt_encoder, (concat_points[0], concat_points[1], mask_input_dummy, masks_enable), 'model/prompt_encoder_'+model_id+'.onnx',
                input_names=["coords", "labels", "masks", "masks_enable"],
                output_names=["sparse_embeddings", "dense_embeddings", "dense_pe"],
                dynamic_axes={
                    'coords': {0: 'b', 1: 'n'},
                    'labels': {0: 'b', 1: 'n'},
                    'masks': {0: 'b', 1: 'h', 2: 'w'},
                },
                verbose=False, opset_version=17
            )

        if import_from_onnx:
            if self.prompt_encoder_onnx == None:
                self.prompt_encoder_onnx = onnxruntime.InferenceSession("model/prompt_encoder_"+model_id+".onnx")
            sparse_embeddings, dense_embeddings, dense_pe = self.prompt_encoder_onnx.run(None, {"coords":concat_points[0].numpy(), "labels":concat_points[1].numpy(), "masks": mask_input_dummy.numpy(), "masks_enable":masks_enable.numpy()})
            sparse_embeddings = torch.Tensor(sparse_embeddings)
            dense_embeddings = torch.Tensor(dense_embeddings)
            dense_pe = torch.Tensor(dense_pe)

        tflite_int8_prompt_encoder = tflite_int8

        if export_to_tflite:
            import ai_edge_torch
            import tensorflow as tf
            sample_inputs = (concat_points[0], concat_points[1], mask_input_dummy, masks_enable)

            if not tflite_int8_prompt_encoder:
                edge_model = ai_edge_torch.convert(self.model.sam_prompt_encoder, sample_inputs)
                edge_model.export("model/prompt_encoder_"+model_id+".tflite")

            if tflite_int8_prompt_encoder:
                if tflite_int8 == "mixed":
                    # torch quantization
                    from ai_edge_torch.quantize import pt2e_quantizer
                    from ai_edge_torch.quantize import quant_config
                    from torch.ao.quantization import quantize_pt2e

                    # torch quantization
                    from ai_edge_torch.quantize import pt2e_quantizer
                    from ai_edge_torch.quantize import quant_config
                    from torch.ao.quantization import quantize_pt2e
                    from ai_edge_torch.quantize.pt2e_quantizer_utils import get_input_act_qspec, get_output_act_qspec
                    from torch.ao.quantization.quantizer import QuantizationAnnotation

                    quantization_config = pt2e_quantizer.get_symmetric_quantization_config(is_dynamic=False, is_per_channel=True)

                    class PT2EQuantizerPromptEncoder(pt2e_quantizer.PT2EQuantizer):
                        def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                            super().annotate(model)

                            for n in model.graph.nodes:
                                #print(n.target, n.name)
                                #if "quantization_annotation" in n.meta:
                                #    print(n.meta["quantization_annotation"])

                                if n.target in [torch.ops.aten.cat.default]:
                                    # labelがint64で量子化できないため、floatにannotateして扱う
                                    if "quantization_annotation" in n.meta:
                                        #print(str(n.args[0][0]))
                                        if str(n.args[0][0]) == "arg1_1":
                                            from torch.ao.quantization.quantizer import QuantizationSpec
                                            from torch.ao.quantization.observer import PlaceholderObserver
                                            act_observer_or_fake_quant_ctr = PlaceholderObserver

                                            float_spec = QuantizationSpec(dtype=torch.float32,
                                                observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
                                                eps=2**-12
                                            ),)

                                            for key in n.meta["quantization_annotation"].input_qspec_map:
                                                n.meta["quantization_annotation"].input_qspec_map[key] = float_spec

                                            #n.meta["quantization_annotation"].output_qspec = int_spec

                                if n.target in [torch.ops.aten.gelu.default, torch.ops.aten.sin.default, torch.ops.aten.cos.default, torch.ops.aten.div_.Tensor]:
                                    # geluとdivをint8で実行
                                    input_qspec_map = {}
                                    for i in range(len(n.args)):
                                        input_qspec_map[n.args[i]] = get_input_act_qspec(quantization_config)
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
                        quantizer = PT2EQuantizerPromptEncoder().set_global(
                            quantization_config
                        )

                    model = torch._export.capture_pre_autograd_graph(self.model.sam_prompt_encoder, sample_inputs)
                    model = quantize_pt2e.prepare_pt2e(model, quantizer)

                    import glob
                    images = glob.glob("./calibration/prompt_encoder/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    for i in range(len(images)):
                        npz = np.load(images[i])
                        data0 = torch.tensor(npz["arr_0"])
                        data1 = torch.tensor(npz["arr_1"])
                        data2 = torch.tensor(npz["arr_2"])
                        data3 = torch.tensor(npz["arr_3"])
                        model(data0, data1, data2, data3) # calibration
    
                    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

                    with_quantizer = ai_edge_torch.convert(
                        model,
                        sample_inputs,
                        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
                    )
                    with_quantizer.export("model/prompt_encoder_"+model_id+".mixed.tflite")
                else:
                    # tensorflow quantization
                    import glob
                    images = glob.glob("./calibration/prompt_encoder/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    def representative_dataset():
                        for i in range(len(images)):
                            npz = np.load(images[i])
                            data2 = npz["arr_0"]
                            data3 = npz["arr_1"]
                            data0 = npz["arr_2"]
                            data1 = npz["arr_3"]

                            yield [data0, data1, data2, data3]

                    
                    tfl_converter_flags = {
                        'optimizations': [tf.lite.Optimize.DEFAULT],
                        'representative_dataset': representative_dataset,
                        'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                        'inference_input_type': tf.int8,
                        'inference_output_type': tf.int8,
                    }

                    tfl_fullint_model = ai_edge_torch.convert(
                        self.model.sam_prompt_encoder, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
                    )

                    tfl_fullint_model.export("model/prompt_encoder_"+model_id+".int8.tflite")

        #if tflite_int8:
        #    tflite_int8_prompt_encoder = False # 精度不足なのでFloatモデルで推論する
        #    print("Warning : prompt encoder will use float model for better accuracy.")

        if import_from_tflite:
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
                self.prompt_encoder_tflite.allocate_tensors()
                input_details = self.prompt_encoder_tflite.get_input_details()
                #self.prompt_encoder_tflite.resize_tensor_input(
                #    input_details[2]["index"], 
                #    [1, concat_points[0].shape[1], 2]
                #)
                self.prompt_encoder_tflite.allocate_tensors()

            input_details = self.prompt_encoder_tflite.get_input_details()
            output_details = self.prompt_encoder_tflite.get_output_details()

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
                self.prompt_encoder_tflite.set_tensor(input_details[2]["index"], concat_points[0].numpy())
                self.prompt_encoder_tflite.set_tensor(input_details[3]["index"], concat_points[1].numpy())
                self.prompt_encoder_tflite.set_tensor(input_details[0]["index"], mask_input_dummy.numpy())
                self.prompt_encoder_tflite.set_tensor(input_details[1]["index"], masks_enable.numpy())
                self.prompt_encoder_tflite.invoke()

                sparse_embeddings = self.prompt_encoder_tflite.get_tensor(output_details[1]["index"])
                dense_embeddings = self.prompt_encoder_tflite.get_tensor(output_details[0]["index"])
                dense_pe = self.prompt_encoder_tflite.get_tensor(output_details[2]["index"])

            sparse_embeddings = torch.Tensor(sparse_embeddings)
            dense_embeddings = torch.Tensor(dense_embeddings)
            dense_pe = torch.Tensor(dense_pe)

        if not import_from_onnx and not import_from_tflite:
            sparse_embeddings, dense_embeddings, dense_pe = self.model.sam_prompt_encoder.forward(
                coords=concat_points[0],
                labels=concat_points[1],
                #boxes=None,
                masks=mask_input_dummy,
                masks_enable=masks_enable
            )

            if self.calibration:
                import os
                os.makedirs("./calibration/prompt_encoder/", exist_ok=True)
                np.savez("./calibration/prompt_encoder/"+str(self.calibration_cnt_prompt_encoder)+".npz",
                    concat_points[0].numpy(),
                    concat_points[1].numpy(),
                    mask_input_dummy.numpy(),
                    masks_enable.numpy()
                )
                self.calibration_cnt_prompt_encoder = self.calibration_cnt_prompt_encoder + 1

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        #print("sparse_embeddings", sparse_embeddings.shape)
        #print("dense_embeddings", dense_embeddings.shape)

        if export_to_onnx:
            self.model.sam_mask_decoder.forward = self.model.sam_mask_decoder.forward_masks # multimask_outputが定数になってしまうので分離
            torch.onnx.export(
                self.model.sam_mask_decoder, (self._features["image_embed"][img_idx].unsqueeze(0), dense_pe, sparse_embeddings, dense_embeddings, batched_mode, high_res_features[0], high_res_features[1]),
                'model/mask_decoder_'+model_id+'.onnx',
                input_names=["image_embeddings", "image_pe", "sparse_prompt_embeddings", "dense_prompt_embeddings", "repeat_image", "high_res_features1", "high_res_features2"],
                output_names=["masks", "iou_pred", "sam_tokens_out", "object_score_logits"],
                dynamic_axes={
                    'sparse_prompt_embeddings': {1: 'n'},
                },
                verbose=False, opset_version=17
            )
        
        if import_from_onnx:
            if self.mask_decoder_onnx == None:
                self.mask_decoder_onnx  = onnxruntime.InferenceSession("model/mask_decoder_"+model_id+".onnx")
            masks, iou_pred, sam_tokens_out, object_score_logits = self.mask_decoder_onnx.run(None, {
                "image_embeddings":self._features["image_embed"][img_idx].unsqueeze(0).numpy(),
                "image_pe": dense_pe.numpy(),
                "sparse_prompt_embeddings": sparse_embeddings.numpy(),
                "dense_prompt_embeddings": dense_embeddings.numpy(),
                "high_res_features1":high_res_features[0].numpy(),
                "high_res_features2":high_res_features[1].numpy()})
            masks = torch.Tensor(masks)
            iou_pred = torch.Tensor(iou_pred)
            sam_tokens_out = torch.Tensor(sam_tokens_out)
            object_score_logits = torch.Tensor(object_score_logits)
            low_res_masks, iou_predictions, _, _  = self.model.sam_mask_decoder.forward_postprocess(masks, iou_pred, sam_tokens_out, object_score_logits, multimask_output)

        if export_to_tflite:
            self.model.sam_mask_decoder.forward = self.model.sam_mask_decoder.forward_masks
            sample_inputs = (self._features["image_embed"][img_idx].unsqueeze(0), dense_pe, sparse_embeddings, dense_embeddings, batched_mode, high_res_features[0], high_res_features[1])

            if not tflite_int8:
                import ai_edge_torch
                edge_model = ai_edge_torch.convert(self.model.sam_mask_decoder, sample_inputs)
                edge_model.export("model/mask_decoder_"+model_id+".tflite")

            if tflite_int8:
                if tflite_int8 == "mixed":
                    import ai_edge_torch

                    # torch quantization
                    from ai_edge_torch.quantize import pt2e_quantizer
                    from ai_edge_torch.quantize import quant_config
                    from torch.ao.quantization import quantize_pt2e

                    # quantize transpose_conv
                    from ai_edge_torch.quantize.pt2e_quantizer_utils import get_input_act_qspec, get_weight_qspec, get_bias_qspec, get_output_act_qspec
                    from torch.ao.quantization.quantizer import QuantizationAnnotation

                    quantization_config = pt2e_quantizer.get_symmetric_quantization_config(is_dynamic=False, is_per_channel=False)

                    class PT2EQuantizer2(pt2e_quantizer.PT2EQuantizer):
                        def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                            super().annotate(model)

                            for n in model.graph.nodes:
                                if n.target in [torch.ops.aten.conv_transpose2d.input]:
                                    input_qspec_map = {}

                                    input_qspec_map[n.args[0]] = get_input_act_qspec(quantization_config)

                                    # weightにも制約をかけるとうまくint8に変換できなくなるため、inputにだけ制約をかける
                                    #input_qspec_map[n.args[1]] = get_weight_qspec(quantization_config)
                                    #input_qspec_map[n.args[2]] = get_bias_qspec(quantization_config)

                                    n.meta["quantization_annotation"] = QuantizationAnnotation(
                                        input_qspec_map=input_qspec_map,
                                        output_qspec=get_output_act_qspec(quantization_config),
                                        _annotated=True,
                                    )

                                if n.target in [torch.ops.aten.gelu.default]:
                                    input_qspec_map = {}
                                    input_qspec_map[n.args[0]] = get_input_act_qspec(quantization_config)
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
                    model = torch._export.capture_pre_autograd_graph(self.model.sam_mask_decoder, sample_inputs)
                    model = quantize_pt2e.prepare_pt2e(model, quantizer)

                    import glob
                    images = glob.glob("./calibration/mask_decoder/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    for i in range(len(images)):
                        npz = np.load(images[i])
                        data0 = torch.tensor(npz["arr_0"])
                        data1 = torch.tensor(npz["arr_1"])
                        data2 = torch.tensor(npz["arr_2"])
                        data3 = torch.tensor(npz["arr_3"])
                        data4 = batched_mode
                        data5 = torch.tensor(npz["arr_4"])
                        data6 = torch.tensor(npz["arr_5"])
                        model(data0, data1, data2, data3, data4, data5, data6)

                    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

                    with_quantizer = ai_edge_torch.convert(
                        model,
                        sample_inputs,
                        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
                    )
                    with_quantizer.export("model/mask_decoder_"+model_id+".mixed.tflite")
                else:
                    # tensorflow quantization
                    import glob
                    images = glob.glob("./calibration/mask_decoder/*.npz")
                    if len(images) == 0:
                        raise Exception("Calibration data not found. Please run --mode calibration first.")

                    def representative_dataset():
                        for i in range(len(images)):
                            #data3 = self._features["image_embed"][img_idx].unsqueeze(0).numpy().astype(np.float32)
                            #data6 = dense_pe.numpy().astype(np.float32)
                            #data1 = sparse_embeddings.numpy().astype(np.float32)
                            #data2 = dense_embeddings.numpy().astype(np.float32)
                            #data5 = np.zeros((1), dtype=bool)
                            #data5[0] = batched_mode
                            #data0 = high_res_features[0].numpy().astype(np.float32)
                            #data4 = high_res_features[1].numpy().astype(np.float32)

                            npz = np.load(images[i])
                            data3 = npz["arr_0"]
                            data6 =npz["arr_1"]
                            data1 = npz["arr_2"]
                            data2 = npz["arr_3"]
                            data5 = np.zeros((1), dtype=bool)
                            data5[0] = batched_mode
                            data0 = npz["arr_4"]
                            data4 = npz["arr_5"]

                            yield [data0, data1, data2, data3, data4, data5, data6]
                    
                    tfl_converter_flags = {
                        'optimizations': [tf.lite.Optimize.DEFAULT],
                        'representative_dataset': representative_dataset,
                        'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                        'inference_input_type': tf.int8,
                        'inference_output_type': tf.int8,
                    }

                    tfl_fullint_model = ai_edge_torch.convert(
                        self.model.sam_mask_decoder, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
                    )

                    tfl_fullint_model.export("model/mask_decoder_"+model_id+".int8.tflite")
                
        if import_from_tflite:
            batched_mode_np = np.zeros((1), dtype=bool)
            if batched_mode:
                batched_mode_np[0] = True

            if self.mask_decoder_tflite == None:
                int8_id = ""
                if tflite_int8:
                    int8_id = "." + tflite_int8
                if import_from_tflite == "ailia_tflite":
                    import ailia_tflite
                    self.mask_decoder_tflite  = ailia_tflite.Interpreter(model_path="model/mask_decoder_"+model_id+int8_id+".tflite", memory_mode=ailia_tflite.AILIA_TFLITE_MEMORY_MODE_REDUCE_INTERSTAGE, flags=ailia_tflite.AILIA_TFLITE_FLAG_DYNAMIC_QUANT)
                else:
                    import tensorflow as tf
                    self.mask_decoder_tflite  = tf.lite.Interpreter(model_path="model/mask_decoder_"+model_id+int8_id+".tflite")
                self.mask_decoder_tflite.allocate_tensors()
                input_details = self.mask_decoder_tflite.get_input_details()
                #self.mask_decoder_tflite.resize_tensor_input(
                #    input_details[1]["index"], 
                #    [1, sparse_embeddings.shape[1], 256]
                #)
                self.mask_decoder_tflite.allocate_tensors()

            input_details = self.mask_decoder_tflite.get_input_details()
            output_details = self.mask_decoder_tflite.get_output_details()

            batched_mode = False

            if tflite_int8:
                self.mask_decoder_tflite.set_tensor(input_details[3]["index"], self.format_input_tensor(self._features["image_embed"][img_idx].unsqueeze(0).numpy(), input_details, 3))
                self.mask_decoder_tflite.set_tensor(input_details[6]["index"], self.format_input_tensor(dense_pe.numpy(), input_details, 6))
                self.mask_decoder_tflite.set_tensor(input_details[1]["index"], self.format_input_tensor(sparse_embeddings.numpy(), input_details, 1))
                self.mask_decoder_tflite.set_tensor(input_details[2]["index"], self.format_input_tensor(dense_embeddings.numpy(), input_details, 2))
                self.mask_decoder_tflite.set_tensor(input_details[5]["index"], batched_mode)
                self.mask_decoder_tflite.set_tensor(input_details[0]["index"], self.format_input_tensor(high_res_features[0].numpy(), input_details, 0))
                self.mask_decoder_tflite.set_tensor(input_details[4]["index"], self.format_input_tensor(high_res_features[1].numpy(), input_details, 4))
                self.mask_decoder_tflite.invoke()

                masks = self.get_output_tensor(self.mask_decoder_tflite, output_details, 2)
                iou_pred = self.get_output_tensor(self.mask_decoder_tflite, output_details, 0)
                sam_tokens_out = self.get_output_tensor(self.mask_decoder_tflite, output_details, 3)
                object_score_logits = self.get_output_tensor(self.mask_decoder_tflite, output_details, 1)
            else:
                self.mask_decoder_tflite.set_tensor(input_details[3]["index"], self._features["image_embed"][img_idx].unsqueeze(0).numpy())
                self.mask_decoder_tflite.set_tensor(input_details[6]["index"], dense_pe.numpy())
                self.mask_decoder_tflite.set_tensor(input_details[1]["index"], sparse_embeddings.numpy())
                self.mask_decoder_tflite.set_tensor(input_details[2]["index"], dense_embeddings.numpy())
                self.mask_decoder_tflite.set_tensor(input_details[5]["index"], batched_mode)
                self.mask_decoder_tflite.set_tensor(input_details[0]["index"], high_res_features[0].numpy())
                self.mask_decoder_tflite.set_tensor(input_details[4]["index"], high_res_features[1].numpy())
                self.mask_decoder_tflite.invoke()

                masks = self.mask_decoder_tflite.get_tensor(output_details[2]["index"])
                iou_pred = self.mask_decoder_tflite.get_tensor(output_details[0]["index"])
                sam_tokens_out = self.mask_decoder_tflite.get_tensor(output_details[3]["index"])
                object_score_logits = self.mask_decoder_tflite.get_tensor(output_details[1]["index"])

            masks = torch.Tensor(masks)
            iou_pred = torch.Tensor(iou_pred)
            sam_tokens_out = torch.Tensor(sam_tokens_out)
            object_score_logits = torch.Tensor(object_score_logits)

            low_res_masks, iou_predictions, _, _  = self.model.sam_mask_decoder.forward_postprocess(masks, iou_pred, sam_tokens_out, object_score_logits, multimask_output)

        if not import_from_onnx and not import_from_tflite:
            convert_to_static_shape = True #export_to_tflite or import_from_tflite or self.calibration
            if convert_to_static_shape:
                max_num_sparse_embeddings = 32 #sparse_embeddings.shape[1]
                sparse_embeddings_pad = torch.zeros(sparse_embeddings.shape[0], max_num_sparse_embeddings, sparse_embeddings.shape[2])
                sparse_embeddings_pad[:,:sparse_embeddings.shape[1],:] = sparse_embeddings
                attn_masks = torch.zeros((sparse_embeddings_pad.shape[0], sparse_embeddings_pad.shape[1]), dtype=torch.bool)
                attn_masks[:,:sparse_embeddings.shape[1]] = True
                print(sparse_embeddings.shape)
                print(attn_masks)
                sparse_embeddings = sparse_embeddings_pad
            else:
                attn_masks = torch.ones((sparse_embeddings.shape[0], sparse_embeddings.shape[1]), dtype=torch.bool)

            self.model.sam_mask_decoder.forward = self.model.sam_mask_decoder.forward_normal
            low_res_masks, iou_predictions, _, _  = self.model.sam_mask_decoder(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=batched_mode,
                high_res_features1=high_res_features[0],
                high_res_features2=high_res_features[1],
                attn_masks=attn_masks
            )

            if self.calibration:
                import os
                os.makedirs("./calibration/mask_decoder/", exist_ok=True)
                np.savez("./calibration/mask_decoder/"+str(self.calibration_cnt_mask_decoder)+".npz",
                    self._features["image_embed"][img_idx].unsqueeze(0).numpy(),
                    dense_pe.numpy(),
                    sparse_embeddings.numpy(),
                    dense_embeddings.numpy(),
                    high_res_features[0].numpy(),
                    high_res_features[1].numpy()
                )
                self.calibration_cnt_mask_decoder = self.calibration_cnt_mask_decoder + 1

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
