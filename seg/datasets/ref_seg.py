"""
author: chenjiahui
date: 2025-11-11
description: 新增文本交互数据集，兼容 jsonl/json/目录结构，按 mmengine 格式返回 img_path/instances/text 等字段。
支持 COCO 格式 JSON + refs.p 的组合格式，以及 REFER API 加载方式。
"""

import os
import os.path as osp
import json
import pickle
import re
import numpy as np
from glob import glob
from collections import defaultdict
from mmdet.registry import DATASETS
from mmengine.fileio import list_from_file, load, get_local_path
from mmdet.datasets.base_det_dataset import BaseDetDataset
from seg.datasets.utils.refcoco_refer import REFER

@DATASETS.register_module()
class RefSegDataset(BaseDetDataset):
    """Referring Expression Segmentation Dataset.
    
    Supports both REFER API loading and direct COCO format + refs.p loading.
    
    Args:
        ann_file: Annotation file path or dataset directory
        data_root: Root directory of the dataset
        dataset_name: Dataset name ('refcoco', 'refcoco+', 'refcocog', 'refclef') 
                     for REFER API loading. If None, uses direct loading.
        split_by: Split type ('unc' for refcoco/refcoco+/refclef, 'umd' or 'google' for refcocog)
        split: Data split ('train', 'val', 'test')
        use_refer_api: Whether to use REFER API for loading (default: False, auto-detect)
        max_sentences_per_image: Maximum number of sentences to sample per image (default: 3)
        debug: Debug mode, only load first 1000 samples (default: False)
        **kwargs: Other arguments for BaseDetDataset
    """
    METAINFO = dict(classes=[], palette=[])

    def __init__(self, 
                 ann_file, 
                 data_root='', 
                 dataset_name=None,
                 split_by='unc',
                 split='train',
                 use_refer_api=None,
                 max_sentences_per_image=3,
                 debug=False,
                 **kwargs):
        self.dataset_name = dataset_name
        self.split_by = split_by
        self.split = split
        self.use_refer_api = use_refer_api
        self.max_sentences_per_image = max_sentences_per_image
        self.debug = debug
        self.refer_api = None
        
        super().__init__(ann_file=ann_file, data_root=data_root, **kwargs)

    def _load_coco_format(self, json_path):
        """加载 COCO 格式的 JSON 文件，返回 images 和 annotations 的映射"""
        with get_local_path(json_path) as p:
            content = json.load(open(p, 'r'))
        
        if not isinstance(content, dict):
            return None, None
        
        # 检查是否是 COCO 格式
        if 'images' not in content or 'annotations' not in content:
            return None, None
        
        images = {img['id']: img for img in content['images']}
        # 按 image_id 组织 annotations
        anns_by_img = defaultdict(list)
        for ann in content['annotations']:
            anns_by_img[ann['image_id']].append(ann)
        
        return images, anns_by_img

    def _load_refs_pickle(self, refs_path):
        """加载 refs.p 文件，返回按 ann_id 组织的引用信息"""
        if not osp.exists(refs_path):
            return {}
        
        try:
            with open(refs_path, 'rb') as f:
                refs = pickle.load(f)
        except Exception:
            return {}
        
        # 按 ann_id 组织，每个 ann_id 可能有多个引用（多个文本描述）
        refs_by_ann = defaultdict(list)
        for ref in refs:
            ann_id = ref.get('ann_id')
            if ann_id is not None:
                # 提取所有句子作为文本
                sentences = ref.get('sentences', [])
                for sent in sentences:
                    text = sent.get('sent') or sent.get('raw', '')
                    if text:
                        refs_by_ann[ann_id].append({
                            'text': text,
                            'ann_id': ann_id,
                            'image_id': ref.get('image_id'),
                            'file_name': ref.get('file_name')
                        })
        
        return refs_by_ann

    def _load_with_refer_api(self):
        """Load data using REFER API."""
        if self.refer_api is None:
            # Determine if we should use REFER API
            if self.use_refer_api is None:
                # Auto-detect: check if dataset_name is provided and directory exists
                if self.dataset_name is not None:
                    dataset_dir = osp.join(self.data_root, self.dataset_name)
                    refs_file = osp.join(dataset_dir, f'refs({self.split_by}).p')
                    instances_file = osp.join(dataset_dir, 'instances.json')
                    if osp.exists(refs_file) and osp.exists(instances_file):
                        self.use_refer_api = True
                    else:
                        self.use_refer_api = False
                else:
                    self.use_refer_api = False
            
            if self.use_refer_api:
                try:
                    self.refer_api = REFER(
                        data_root=self.data_root,
                        dataset=self.dataset_name,
                        splitBy=self.split_by
                    )
                except Exception as e:
                    print(f"Failed to load REFER API: {e}")
                    print("Falling back to direct loading...")
                    self.use_refer_api = False
        
        if not self.use_refer_api:
            return []
        
        # Load data using REFER API
        ref_ids = self.refer_api.getRefIds(split=self.split)
        image_ids = self.refer_api.getImgIds(ref_ids=ref_ids)
        refs = self.refer_api.loadRefs(ref_ids=ref_ids)
        
        # Create image to refs mapping
        img_to_refs = defaultdict(list)
        for ref in refs:
            img_to_refs[ref['image_id']].append(ref)
        
        # Load images
        images = self.refer_api.loadImgs(image_ids=image_ids)
        
        data_list = []
        for image_info in images:
            img_id = image_info['id']
            refs_for_img = img_to_refs.get(img_id, [])
            
            if len(refs_for_img) == 0:
                continue
            
            # Collect all sentences and ann_ids
            sents = []
            ann_ids = []
            for ref in refs_for_img:
                for sent in ref.get('sentences', []):
                    text = sent.get('sent') or sent.get('raw', '')
                    if text:
                        sents.append(text)
                        ann_ids.append(ref['ann_id'])
            
            if len(sents) == 0:
                continue
            
            # Sample sentences (similar to OMG-Seg-main)
            if len(sents) >= self.max_sentences_per_image:
                sampled_inds = np.random.choice(
                    list(range(len(sents))), 
                    size=self.max_sentences_per_image, 
                    replace=False
                )
            else:
                sampled_inds = list(range(len(sents)))
            
            sampled_sents = [sents[i] for i in sampled_inds]
            sampled_ann_ids = [ann_ids[i] for i in sampled_inds]
            
            # Create data items (one per sampled sentence)
            for sent_text, ann_id in zip(sampled_sents, sampled_ann_ids):
                ann = self.refer_api.Anns[ann_id]
                
                # Get image path (relative to data_root)
                # Convert COCO 2014 filename format to COCO 2017 format if needed
                # COCO 2014: COCO_train2014_000000098304.jpg -> 000000098304.jpg
                # COCO 2014: COCO_val2014_000000098304.jpg -> 000000098304.jpg
                # This allows using COCO 2017 images with RefCOCO annotations
                original_file_name = image_info.get('file_name', '')
                if 'COCO_train2014_' in original_file_name or 'COCO_val2014_' in original_file_name:
                    # Use image ID directly (more reliable than parsing filename)
                    # Format as 12-digit zero-padded number
                    img_path = f"{img_id:012d}.jpg"
                else:
                    # Keep original filename (for RefCLEF or other datasets)
                    img_path = original_file_name
                
                # Keep relative path, BaseDetDataset will handle it with data_prefix
                # The actual image loading will use data_prefix specified in config
                # Join with data_prefix if provided and path is not absolute
                img_prefix = None
                if isinstance(self.data_prefix, dict):
                    img_prefix = self.data_prefix.get('img')
                elif isinstance(self.data_prefix, str):
                    img_prefix = self.data_prefix
                if img_prefix and not osp.isabs(img_path):
                    img_path = osp.join(img_prefix, img_path)
                
                # Create instance with annotation
                bbox = ann.get('bbox', [0, 0, 0, 0])
                # COCO bbox format is [x, y, w, h], convert to [x1, y1, x2, y2]
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                
                instances = [dict(
                    bbox=bbox,
                    bbox_label=ann.get('category_id', 0),
                    ignore_flag=ann.get('iscrowd', 0),
                    mask=ann.get('segmentation')  # RLE or polygon format
                )]
                
                data_item = dict(
                    img_path=img_path,  # Keep relative path (converted to COCO 2017 format)
                    img_id=img_id,
                    height=image_info.get('height'),
                    width=image_info.get('width'),
                    text=sent_text,
                    custom_entities=True,
                    instances=instances
                )
                data_list.append(data_item)
        
        if self.debug:
            data_list = data_list[:1000]
        
        return data_list

    def load_data_list(self):
        def _normalize_item(item):
            img_path = item.get('img_path') or item.get('image') or item.get('file_name')
            height = item.get('height')
            width = item.get('width')
            text = item.get('text') or item.get('ref') or item.get('phrase') or item.get('sentence')
            instances_src = item.get('instances') or item.get('annotations') or []
            instances = []
            for ann in instances_src:
                bbox = ann.get('bbox') or ann.get('box') or [0, 0, 0, 0]
                # 处理 COCO 格式的 bbox (x, y, w, h) -> (x1, y1, x2, y2)
                if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                    if bbox[2] > 0 and bbox[3] > 0:  # w, h > 0
                        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                instances.append(dict(
                    bbox=bbox,
                    bbox_label=ann.get('bbox_label', ann.get('category_id', 0)),
                    ignore_flag=ann.get('ignore_flag', ann.get('iscrowd', 0)),
                    mask=ann.get('mask') or ann.get('segmentation')
                ))
            if not (img_path and height and width):
                return None
            if not text:
                # 如果没有文本，跳过（文本交互数据集必须有文本）
                return None
            # BaseDetDataset 会使用 data_prefix 来处理路径，所以这里保持相对路径
            # 如果已经是绝对路径，则保持不变
            if not osp.isabs(img_path):
                # 将相对路径与 data_prefix 合并为绝对路径（与其他数据集实现保持一致）
                img_prefix = None
                if isinstance(self.data_prefix, dict):
                    img_prefix = self.data_prefix.get('img')
                elif isinstance(self.data_prefix, str):
                    img_prefix = self.data_prefix
                if img_prefix:
                    img_path = osp.join(img_prefix, img_path)
            return dict(
                img_path=img_path,
                img_id=item.get('img_id', item.get('id', 0)),
                height=height,
                width=width,
                text=text,
                custom_entities=True,
                instances=instances
            )

        data_list = []
        ann_path = self.ann_file
        
        # Try REFER API loading first
        refer_data = self._load_with_refer_api()
        if refer_data:
            return refer_data

        # 1) jsonl
        if ann_path.endswith('.jsonl') and osp.isfile(ann_path):
            lines = list_from_file(ann_path)
            for line in lines:
                try:
                    item = load(line, file_format='json')
                    norm = _normalize_item(item)
                    if norm is not None:
                        data_list.append(norm)
                except Exception:
                    continue
            return data_list

        # 2) json（支持数组/字典/逐行）
        if ann_path.endswith('.json') and osp.isfile(ann_path):
            # 尝试加载 COCO 格式
            images, anns_by_img = self._load_coco_format(ann_path)
            if images is not None:
                # 查找同目录下的 refs.p 文件
                refs_dir = osp.dirname(ann_path)
                refs_files = glob(osp.join(refs_dir, 'refs*.p'))
                refs_by_ann = {}
                if refs_files:
                    # 优先使用 refs(unc).p
                    refs_path = osp.join(refs_dir, 'refs(unc).p')
                    if not osp.exists(refs_path) and refs_files:
                        refs_path = refs_files[0]
                    refs_by_ann = self._load_refs_pickle(refs_path)
                
                # 组合 images, annotations 和 refs
                for img_id, img in images.items():
                    img_anns = anns_by_img.get(img_id, [])
                    for ann in img_anns:
                        ann_id = ann.get('id')
                        refs = refs_by_ann.get(ann_id, [])
                        if not refs:
                            # 如果没有引用，跳过这个 annotation
                            continue
                        # 为每个引用创建一个数据项
                        for ref_info in refs:
                            item = {
                                'file_name': img.get('file_name', ref_info.get('file_name', '')),
                                'height': img.get('height'),
                                'width': img.get('width'),
                                'id': img_id,
                                'img_id': img_id,
                                'text': ref_info.get('text', ''),
                                'annotations': [ann]
                            }
                            norm = _normalize_item(item)
                            if norm is not None:
                                data_list.append(norm)
                return data_list
            
            # 非 COCO 格式，按原逻辑处理
            with get_local_path(ann_path) as p:
                try:
                    content = json.load(open(p, 'r'))
                    if isinstance(content, list):
                        for item in content:
                            norm = _normalize_item(item)
                            if norm is not None:
                                data_list.append(norm)
                    elif isinstance(content, dict):
                        candidates = content.get('data') or content.get('annotations') or []
                        for item in candidates:
                            norm = _normalize_item(item)
                            if norm is not None:
                                data_list.append(norm)
                except Exception:
                    # 按行解析
                    lines = list_from_file(ann_path)
                    for line in lines:
                        try:
                            itm = json.loads(line)
                            norm = _normalize_item(itm)
                            if norm is not None:
                                data_list.append(norm)
                        except Exception:
                            continue
            return data_list

        # 3) 目录：递归解析 *.jsonl / *.json
        if osp.isdir(ann_path):
            json_files = glob(osp.join(ann_path, '**', '*.jsonl'), recursive=True)
            json_files += glob(osp.join(ann_path, '**', '*.json'), recursive=True)
            for jf in sorted(json_files):
                if jf.endswith('.jsonl'):
                    lines = list_from_file(jf)
                    for line in lines:
                        try:
                            item = load(line, file_format='json')
                            norm = _normalize_item(item)
                            if norm is not None:
                                data_list.append(norm)
                        except Exception:
                            continue
                else:
                    # 尝试加载 COCO 格式
                    images, anns_by_img = self._load_coco_format(jf)
                    if images is not None:
                        # 查找同目录下的 refs.p 文件
                        refs_dir = osp.dirname(jf)
                        refs_files = glob(osp.join(refs_dir, 'refs*.p'))
                        refs_by_ann = {}
                        if refs_files:
                            # 优先使用 refs(unc).p
                            refs_path = osp.join(refs_dir, 'refs(unc).p')
                            if not osp.exists(refs_path) and refs_files:
                                refs_path = refs_files[0]
                            refs_by_ann = self._load_refs_pickle(refs_path)
                        
                        # 组合 images, annotations 和 refs
                        for img_id, img in images.items():
                            img_anns = anns_by_img.get(img_id, [])
                            for ann in img_anns:
                                ann_id = ann.get('id')
                                refs = refs_by_ann.get(ann_id, [])
                                if not refs:
                                    continue
                                # 为每个引用创建一个数据项
                                for ref_info in refs:
                                    item = {
                                        'file_name': img.get('file_name', ref_info.get('file_name', '')),
                                        'height': img.get('height'),
                                        'width': img.get('width'),
                                        'id': img_id,
                                        'img_id': img_id,
                                        'text': ref_info.get('text', ''),
                                        'annotations': [ann]
                                    }
                                    norm = _normalize_item(item)
                                    if norm is not None:
                                        data_list.append(norm)
                        continue
                    
                    # 非 COCO 格式，按原逻辑处理
                    try:
                        content = json.load(open(jf, 'r'))
                        if isinstance(content, list):
                            for item in content:
                                norm = _normalize_item(item)
                                if norm is not None:
                                    data_list.append(norm)
                        elif isinstance(content, dict):
                            candidates = content.get('data') or content.get('annotations') or []
                            for item in candidates:
                                norm = _normalize_item(item)
                                if norm is not None:
                                    data_list.append(norm)
                    except Exception:
                        lines = list_from_file(jf)
                        for line in lines:
                            try:
                                itm = json.loads(line)
                                norm = _normalize_item(itm)
                                if norm is not None:
                                    data_list.append(norm)
                            except Exception:
                                continue
            return data_list

        # 兜底：按行解析
        lines = list_from_file(ann_path)
        for line in lines:
            try:
                item = load(line, file_format='json')
                norm = _normalize_item(item)
                if norm is not None:
                    data_list.append(norm)
            except Exception:
                continue
        return data_list