"""
author: chenjiahui
date: 2025-11-11
description: 新增文本交互数据集，兼容 jsonl/json/目录结构，按 mmengine 格式返回 img_path/instances/text 等字段。
"""

import os
import os.path as osp
import json
from glob import glob
from mmdet.registry import DATASETS
from mmengine.fileio import list_from_file, load, get_local_path
from mmdet.datasets.base_det_dataset import BaseDetDataset

@DATASETS.register_module()
class RefSegDataset(BaseDetDataset):
    METAINFO = dict(classes=[], palette=[])

    def __init__(self, ann_file, data_root='', **kwargs):
        super().__init__(ann_file=ann_file, data_root=data_root, **kwargs)

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
                instances.append(dict(
                    bbox=bbox,
                    bbox_label=ann.get('bbox_label', 0),
                    ignore_flag=ann.get('ignore_flag', 0),
                    mask=ann.get('mask')
                ))
            if not (img_path and height and width and text):
                return None
            if not osp.isabs(img_path):
                img_path = osp.join(self.data_root, img_path)
            return dict(
                img_path=img_path,
                img_id=0,
                height=height,
                width=width,
                text=text,
                custom_entities=True,
                instances=instances
            )

        data_list = []
        ann_path = self.ann_file

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