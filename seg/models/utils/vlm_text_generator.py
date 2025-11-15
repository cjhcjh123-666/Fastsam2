# Copyright (c) OpenMMLab. All rights reserved.
"""VLM-based Text Generator for Instance Captioning.

This module uses Vision-Language Models (VLM) to generate text descriptions
for instances, which can be used for training or inference.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType
from mmengine.structures import InstanceData


@MODELS.register_module()
class VLMTextGenerator(nn.Module):
    """Generate text descriptions for instances using VLM.
    
    This module can be used in two scenarios:
    1. Training: Generate text descriptions for COCO instances that don't have text
    2. Inference: Generate text descriptions for detected instances
    
    Args:
        vlm_cfg (OptConfigType): Configuration for VLM model (e.g., CoCa, BLIP).
            If None, uses a simple class-name-based generator. Default: None.
        use_class_names (bool): Whether to use class names as fallback text.
            Default: True.
        class_name_mapping (Optional[Dict[int, str]]): Mapping from class ID to class name.
            If None, uses generic names. Default: None.
        max_text_length (int): Maximum length of generated text. Default: 77.
        temperature (float): Temperature for text generation. Default: 1.0.
    """
    
    def __init__(self,
                 vlm_cfg: OptConfigType = None,
                 use_class_names: bool = True,
                 class_name_mapping: Optional[Dict[int, str]] = None,
                 max_text_length: int = 77,
                 temperature: float = 1.0):
        super().__init__()
        self.vlm_model = None
        self.use_class_names = use_class_names
        self.class_name_mapping = class_name_mapping
        self.max_text_length = max_text_length
        self.temperature = temperature
        
        if vlm_cfg is not None:
            # Build VLM model from config (e.g., CoCa, BLIP)
            self.vlm_model = MODELS.build(vlm_cfg)
            # Freeze VLM if needed (usually we want to freeze it)
            if hasattr(self.vlm_model, 'eval'):
                self.vlm_model.eval()
                for param in self.vlm_model.parameters():
                    param.requires_grad = False
    
    def generate_from_image_crop(self,
                                 image: torch.Tensor,
                                 bbox: torch.Tensor,
                                 image_size: Tuple[int, int]) -> str:
        """Generate text description from an image crop.
        
        Args:
            image: Full image tensor [C, H, W].
            bbox: Bounding box in format [x1, y1, x2, y2] (normalized or pixel).
            image_size: (H, W) of the image.
            
        Returns:
            Generated text description string.
        """
        if self.vlm_model is None:
            # Fallback: use generic description
            return "object"
        
        # Crop image region
        h, w = image_size
        x1, y1, x2, y2 = bbox
        
        # Convert normalized bbox to pixel coordinates if needed
        if x1 < 1.0 and y1 < 1.0:
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure valid coordinates
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Crop and resize if needed
        crop = image[:, y1:y2, x1:x2]
        
        # Generate text using VLM
        if hasattr(self.vlm_model, 'generate'):
            # Use VLM's generate method (e.g., CoCa)
            with torch.no_grad():
                text = self.vlm_model.generate(
                    crop.unsqueeze(0),
                    seq_len=self.max_text_length,
                    temperature=self.temperature
                )
            # Decode text tokens to string (implementation depends on VLM)
            # For now, return placeholder
            return "object"  # TODO: Implement proper token decoding
        else:
            return "object"
    
    def generate_from_class_id(self, class_id: int) -> str:
        """Generate text description from class ID.
        
        Args:
            class_id: Class ID.
            
        Returns:
            Text description string.
        """
        if self.class_name_mapping is not None and class_id in self.class_name_mapping:
            return self.class_name_mapping[class_id]
        elif self.use_class_names:
            # Generic class name
            return f"class_{class_id}"
        else:
            return "object"
    
    def generate_for_instances(self,
                              instances: InstanceData,
                              image: Optional[torch.Tensor] = None,
                              image_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """Generate text descriptions for multiple instances.
        
        Args:
            instances: InstanceData containing bboxes, labels, etc.
            image: Full image tensor [C, H, W]. Optional if using class names.
            image_size: (H, W) of the image. Optional if using class names.
            
        Returns:
            List of text description strings, one per instance.
        """
        num_instances = len(instances)
        texts = []
        
        has_bboxes = hasattr(instances, 'bboxes') and instances.bboxes is not None
        has_labels = hasattr(instances, 'labels') and instances.labels is not None
        
        for i in range(num_instances):
            # Try to generate from image crop if VLM is available
            if self.vlm_model is not None and image is not None and image_size is not None and has_bboxes:
                bbox = instances.bboxes[i]
                text = self.generate_from_image_crop(image, bbox, image_size)
            elif has_labels:
                # Fallback to class name
                class_id = instances.labels[i].item() if torch.is_tensor(instances.labels[i]) else instances.labels[i]
                text = self.generate_from_class_id(class_id)
            else:
                # Default text
                text = "object"
            
            texts.append(text)
        
        return texts
    
    def forward(self,
                instances: Union[InstanceData, List[InstanceData]],
                image: Optional[torch.Tensor] = None,
                image_size: Optional[Tuple[int, int]] = None) -> Union[List[str], List[List[str]]]:
        """Forward pass to generate text descriptions.
        
        Args:
            instances: InstanceData or list of InstanceData.
            image: Full image tensor [C, H, W] or batch of images [B, C, H, W].
            image_size: (H, W) of the image or list of (H, W) for batch.
            
        Returns:
            List of text strings or list of lists for batch.
        """
        if isinstance(instances, list):
            # Batch processing
            texts_list = []
            for i, inst in enumerate(instances):
                img = image[i] if image is not None and image.dim() == 4 else image
                img_size = image_size[i] if isinstance(image_size, list) else image_size
                texts = self.generate_for_instances(inst, img, img_size)
                texts_list.append(texts)
            return texts_list
        else:
            # Single instance
            return self.generate_for_instances(instances, image, image_size)

