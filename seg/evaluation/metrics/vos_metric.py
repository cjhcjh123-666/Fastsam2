# Copyright (c) OpenMMLab. All rights reserved.
import os.path
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dist import barrier
from mmdet.registry import METRICS
from mmdet.evaluation.metrics.base_video_metric import BaseVideoMetric

PALETTE = {
    'davis': b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0',
    'mose': b'\x00\x00\x00\xe4\x1a\x1c7~\xb8M\xafJ\x98N\xa3\xff\x7f\x00\xff\xff3\xa6V(\xf7\x81\xbf\x99\x99\x99f\xc2\xa5\xfc\x8db\x8d\xa0\xcb\xe7\x8a\xc3\xa6\xd8T\xff\xd9/\xe5\xc4\x94\xb3\xb3\xb3\x8d\xd3\xc7\xff\xff\xb3\xbe\xba\xda\xfb\x80r\x80\xb1\xd3\xfd\xb4b\xb3\xdei\xfc\xcd\xe5\xd9\xd9\xd9\xbc\x80\xbd\xcc\xeb\xc5\xff\xedo',
}


@METRICS.register_module()
class VOSMetric(BaseVideoMetric):
    """mAP evaluation metrics for the VIS task.

    Args:
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `youtube_vis_ap`..
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonyms metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
    """

    default_prefix: Optional[str] = 'vip_seg'

    def __init__(self,
                 metric: Optional[Union[str, List[str]]] = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 palette: Optional[str] = None,
                 results_path: str = 'DAVIS'
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metric = metric
        self.outfile_prefix = outfile_prefix
        self.format_only = format_only
        if palette is not None:
            self.palette = PALETTE[palette]
        else:
            self.palette = None
        self.results_path = results_path

        self.per_video_res = []
        self.categories = {}
        self._vis_meta_info = defaultdict(list)  # record video and image infos
        
        # Store predictions and GT for metric computation
        self.pred_videos = []  # List of dicts: {video_name: {frame_name: pred_mask}}
        self.gt_videos = []   # List of dicts: {video_name: {frame_name: gt_mask}}

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for track_data_sample in data_samples:
            video_data_samples = track_data_sample['video_data_samples']
            
            # Check for pred_track_proposal or pred_track_instances
            has_proposal = 'pred_track_proposal' in video_data_samples[0]
            has_instances = 'pred_track_instances' in video_data_samples[0] and \
                           hasattr(video_data_samples[0].pred_track_instances, 'masks') and \
                           len(video_data_samples[0].pred_track_instances.masks) > 0
            
            if not has_proposal and not has_instances:
                continue
            
            # Convert pred_track_instances to pred_track_proposal if needed
            if not has_proposal and has_instances:
                for frame_sample in video_data_samples:
                    if 'pred_track_instances' in frame_sample:
                        pred_instances = frame_sample.pred_track_instances
                        if hasattr(pred_instances, 'masks') and len(pred_instances.masks) > 0:
                            # Convert masks to proposal format [H, W] with instance IDs
                            masks = pred_instances.masks
                            if hasattr(pred_instances, 'instances_id'):
                                instance_ids = pred_instances.instances_id.cpu().numpy()
                            else:
                                instance_ids = np.arange(1, len(masks) + 1)
                            
                            # Get image shape
                            if isinstance(masks, torch.Tensor):
                                h, w = masks.shape[-2:]
                                masks_np = masks.cpu().numpy()
                            else:
                                h, w = masks[0].shape
                                masks_np = np.array([m.cpu().numpy() if isinstance(m, torch.Tensor) else m for m in masks])
                            
                            # Create proposal map
                            proposal_map = np.zeros((h, w), dtype=np.int32)
                            for idx, (mask, ins_id) in enumerate(zip(masks_np, instance_ids)):
                                if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
                                    mask_bool = mask > 0.5
                                else:
                                    mask_bool = mask[0] > 0.5 if len(mask.shape) == 3 else mask > 0.5
                                proposal_map[mask_bool] = int(ins_id)
                            
                            frame_sample.pred_track_proposal = proposal_map
            
            ori_video_len = video_data_samples[0].ori_video_length
            if ori_video_len == len(video_data_samples):
                # video process
                self.process_video(video_data_samples)
            else:
                # image process
                raise NotImplementedError

    def process_video(self, data_samples):
        video_length = len(data_samples)
        mkdir_or_exist(self.results_path)
        
        # Get video name
        ori_img_path = data_samples[0].img_path
        folder_name = os.path.basename(os.path.dirname(ori_img_path))
        
        # Store predictions and GT for this video
        pred_video = {}
        gt_video = {}
        
        for frame_id in range(video_length):
            img_data_sample = data_samples[frame_id].to_dict()
            pred = img_data_sample['pred_track_proposal']
            h, w = pred.shape
            
            # Store prediction
            file_name = os.path.basename(ori_img_path) if frame_id == 0 else os.path.basename(data_samples[frame_id].img_path)
            file_name = file_name.replace('.jpg', '.png').replace('.jpeg', '.png')
            pred_video[file_name] = pred.copy()
            
            # Store GT if available
            if 'gt_instances' in data_samples[frame_id] and hasattr(data_samples[frame_id].gt_instances, 'masks'):
                gt_masks = data_samples[frame_id].gt_instances.masks
                if hasattr(data_samples[frame_id].gt_instances, 'instances_ids'):
                    gt_ids = data_samples[frame_id].gt_instances.instances_ids.cpu().numpy()
                else:
                    gt_ids = np.arange(1, len(gt_masks) + 1)
                
                # Convert GT masks to proposal format
                gt_proposal = np.zeros((h, w), dtype=np.int32)
                if isinstance(gt_masks, torch.Tensor):
                    gt_masks_np = gt_masks.cpu().numpy()
                else:
                    gt_masks_np = np.array([m.cpu().numpy() if isinstance(m, torch.Tensor) else m for m in gt_masks])
                
                for mask, ins_id in zip(gt_masks_np, gt_ids):
                    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
                        mask_bool = mask > 0.5
                    else:
                        mask_bool = mask[0] > 0.5 if len(mask.shape) == 3 else mask > 0.5
                    gt_proposal[mask_bool] = int(ins_id)
                
                gt_video[file_name] = gt_proposal
            
            # Save visualization
            pred_map = np.zeros((h, w, 3), dtype=np.uint8)
            for ins_id in np.unique(pred):
                if ins_id == 0:
                    continue
                r = ins_id // 1000000
                g = (ins_id % 1000000) // 1000
                b = ins_id % 1000
                pred_map[pred == ins_id] = np.array([r, g, b], dtype=np.uint8)
            
            out_path = os.path.join(self.results_path, folder_name, file_name)
            mkdir_or_exist(os.path.dirname(out_path))
            if self.palette is not None:
                from PIL import Image
                pred_map_rgb = mmcv.bgr2rgb(pred_map)
                pil_image = Image.fromarray(pred_map_rgb)
                pil_image = pil_image.convert('P', palette=self.palette)
                pil_image.save(out_path)
            else:
                mmcv.imwrite(pred_map, out_path)
        
        # Store for metric computation
        if len(pred_video) > 0:
            self.pred_videos.append({folder_name: pred_video})
        if len(gt_video) > 0:
            self.gt_videos.append({folder_name: gt_video})

    def _compute_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute IoU between prediction and GT mask."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection) / float(union)
    
    def _compute_f_measure(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute F-measure (contour accuracy) between prediction and GT mask."""
        from scipy import ndimage
        
        # Get contours using morphological operations
        def get_contour(mask):
            eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
            contour = mask.astype(bool) & (~eroded)
            return contour
        
        pred_contour = get_contour(pred_mask)
        gt_contour = get_contour(gt_mask)
        
        # Compute precision and recall
        if pred_contour.sum() == 0:
            precision = 1.0 if gt_contour.sum() == 0 else 0.0
        else:
            precision = np.logical_and(pred_contour, gt_contour).sum() / pred_contour.sum()
        
        if gt_contour.sum() == 0:
            recall = 1.0 if pred_contour.sum() == 0 else 0.0
        else:
            recall = np.logical_and(pred_contour, gt_contour).sum() / gt_contour.sum()
        
        # Compute F-measure
        if precision + recall == 0:
            return 0.0
        f_measure = 2 * precision * recall / (precision + recall)
        return float(f_measure)
    
    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute VOS metrics (J, F, J&F).
        
        J (Region similarity): Average IoU across all frames and instances
        F (Contour accuracy): Average F-measure across all frames and instances
        J&F: Average of J and F
        """
        from mmengine.logging import print_log
        from mmengine.dist import is_main_process
        
        if not is_main_process():
            return {}
        
        metrics = {}
        
        if self.metric is None:
            metric_list = ['J', 'F', 'J&F']
        elif isinstance(self.metric, str):
            metric_list = [self.metric]
        else:
            metric_list = self.metric
        
        # Collect all IoU and F-measure values
        all_j_values = []
        all_f_values = []
        
        # Process each video
        for pred_video_dict, gt_video_dict in zip(self.pred_videos, self.gt_videos):
            pred_video_name = list(pred_video_dict.keys())[0]
            gt_video_name = list(gt_video_dict.keys())[0]
            
            if pred_video_name != gt_video_name:
                continue
            
            pred_frames = pred_video_dict[pred_video_name]
            gt_frames = gt_video_dict[gt_video_name]
            
            # Process each frame
            for frame_name in pred_frames.keys():
                if frame_name not in gt_frames:
                    continue
                
                pred_proposal = pred_frames[frame_name]
                gt_proposal = gt_frames[frame_name]
                
                # Get unique instance IDs (excluding background 0)
                pred_ids = np.unique(pred_proposal)
                pred_ids = pred_ids[pred_ids != 0]
                gt_ids = np.unique(gt_proposal)
                gt_ids = gt_ids[gt_ids != 0]
                
                # Match instances and compute metrics
                matched_gt_ids = set()
                for pred_id in pred_ids:
                    pred_mask = (pred_proposal == pred_id)
                    
                    # Find best matching GT instance
                    best_iou = 0.0
                    best_gt_id = None
                    best_f = 0.0
                    
                    for gt_id in gt_ids:
                        if gt_id in matched_gt_ids:
                            continue
                        gt_mask = (gt_proposal == gt_id)
                        
                        iou = self._compute_iou(pred_mask, gt_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_id = gt_id
                            best_f = self._compute_f_measure(pred_mask, gt_mask)
                    
                    if best_gt_id is not None and best_iou > 0.5:  # IoU threshold
                        all_j_values.append(best_iou)
                        all_f_values.append(best_f)
                        matched_gt_ids.add(best_gt_id)
                
                # Handle unmatched GT instances (count as 0 IoU and F)
                for gt_id in gt_ids:
                    if gt_id not in matched_gt_ids:
                        all_j_values.append(0.0)
                        all_f_values.append(0.0)
        
        # Compute average metrics
        if len(all_j_values) > 0:
            j_mean = np.mean(all_j_values)
            f_mean = np.mean(all_f_values)
            jf_mean = (j_mean + f_mean) / 2.0
        else:
            j_mean = f_mean = jf_mean = 0.0
            print_log(
                'Warning: No valid predictions found for metric computation.',
                logger='current',
                level='WARNING'
            )
        
        # Store metrics
        for metric_name in metric_list:
            if metric_name == 'J':
                metrics['J'] = float(j_mean)
            elif metric_name == 'F':
                metrics['F'] = float(f_mean)
            elif metric_name == 'J&F':
                metrics['J&F'] = float(jf_mean)
        
        # Log results
        print_log(
            f'VOS Evaluation Results:\n'
            f'  J (Region similarity): {j_mean:.4f}\n'
            f'  F (Contour accuracy): {f_mean:.4f}\n'
            f'  J&F: {jf_mean:.4f}\n'
            f'Results saved to: {self.results_path}',
            logger='current'
        )
        
        return metrics

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        # wait for all processes to complete prediction.
        barrier()
        metrics = self.compute_metrics([])
        return metrics
