"""
author: chenjiahui
date: 2025-11-11
description: 新增文本交互数据集，按 mmengine 格式返回 img_path/instances/text 等字段：
"""

from mmdet.registry import DATASETS
from mmengine.fileio import list_from_file, load
from mmdet.datasets.base_det_dataset import BaseDetDataset

@DATASETS.register_module()
class RefSegDataset(BaseDetDataset):
    METAINFO = dict(classes=[], palette=[])

    def __init__(self, ann_file, data_root='', **kwargs):
        super().__init__(ann_file=ann_file, data_root=data_root, **kwargs)

    def load_data_list(self):
        lines = list_from_file(self.ann_file)
        data_list = []
        for line in lines:
            item = load(line, file_format='json')
            data_info = dict(
                img_path=item['img_path'],
                img_id=0,
                height=item['height'],
                width=item['width'],
                text=item['text'],
                custom_entities=True,  # 标注有自定义文本实体
            )
            instances = []
            for ann in item.get('instances', []):
                instances.append(dict(
                    bbox=ann['bbox'],
                    bbox_label=0,
                    ignore_flag=ann.get('ignore_flag', 0),
                    mask=ann['mask'],
                ))
            data_info['instances'] = instances
            data_list.append(data_info)
        return data_list