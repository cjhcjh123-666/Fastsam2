import torch
import random
from mmdet.models import DetDataPreprocessor
from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from kornia.contrib import distance_transform
import torch.nn.functional as F

@MODELS.register_module()
class SAMDataPreprocessor(DetDataPreprocessor):
    def __init__(self, *args, num_mask_tokens=6, repeat=False, num_proposals=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_mask_tokens = num_mask_tokens
        self.repeat = repeat
        self.num_proposals = num_proposals

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']
        if training:
            return dict(inputs=inputs, data_samples=data_samples)
        for data_sample in data_samples:
            gt_instances = data_sample.gt_instances
            
            device = gt_instances.labels.device
            gt_collected = []
            
            ori_num_instances = len(gt_instances)
            ori_indices = torch.randperm(ori_num_instances)
            if self.repeat and ori_num_instances < self.num_proposals:
                repeat_cnt = (self.num_proposals // ori_num_instances) + 1
                ori_indices = ori_indices.repeat(repeat_cnt)
                ori_indices = ori_indices[:self.num_proposals]
            gt_instances.masks = gt_instances.masks.to_tensor(torch.bool, device)
            h, w = data_sample.metainfo['img_shape']
            
            # Check if text exists in metainfo or gt_instances
            has_text_in_metainfo = 'text' in data_sample.metainfo
            has_text_in_instances = hasattr(gt_instances, 'text') and gt_instances.text is not None
            
            texts_list = []
            for instance_idx in ori_indices:
                mask = gt_instances.masks[instance_idx]
                mask_clone = mask[:h, :w][None, None, :]
                n, _, h, w = mask_clone.shape
                mask_dt = (distance_transform((~F.pad(mask_clone, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:, :, 1:-1, 1:-1])
                selected_point = torch.tensor([mask_dt.argmax()/w, mask_dt.argmax()%w]).long().flip(0).to(device)
                selected_point = torch.cat([selected_point - 3, selected_point + 3], 0)
                # selected_point = gt_instances.bboxes[instance_idx]
                gt_collected.append({
                    'point_coords': selected_point,
                    'instances': None,
                    'masks': mask,
                })
                
                # Collect text if available
                if has_text_in_instances:
                    # If text is a list, get the text for this instance
                    if isinstance(gt_instances.text, list) and instance_idx < len(gt_instances.text):
                        texts_list.append(gt_instances.text[instance_idx])
                    elif not isinstance(gt_instances.text, list):
                        texts_list.append(gt_instances.text)
                    else:
                        texts_list.append(None)
                elif has_text_in_metainfo:
                    # Use text from metainfo (same text for all instances)
                    text = data_sample.metainfo.get('text')
                    if isinstance(text, list):
                        texts_list.append(text[0] if text else None)
                    else:
                        texts_list.append(text)
                else:
                    texts_list.append(None)
            
            # Only create gt_instances_collected if we have collected instances
            if len(gt_collected) > 0:
                data_sample.gt_instances_collected = InstanceData(
                    point_coords=torch.stack([itm['point_coords'] for itm in gt_collected]),
                    sub_instances=[itm['instances'] for itm in gt_collected],
                    masks=torch.stack([itm['masks'] for itm in gt_collected]),
                )
                pb_labels = torch.ones(len(data_sample.gt_instances_collected), dtype=torch.long, device=device)
                data_sample.gt_instances_collected.pb_labels = pb_labels
                
                # Add text to gt_instances_collected if available
                if any(t is not None for t in texts_list):
                    data_sample.gt_instances_collected.text = texts_list
            else:
                # No instances, create empty gt_instances_collected
                data_sample.gt_instances_collected = InstanceData()
                data_sample.gt_instances_collected.point_coords = torch.empty((0, 4), dtype=torch.long, device=device)
                data_sample.gt_instances_collected.masks = torch.empty((0, h, w), dtype=torch.bool, device=device)
                data_sample.gt_instances_collected.pb_labels = torch.empty((0,), dtype=torch.long, device=device)
        return dict(inputs=inputs, data_samples=data_samples)