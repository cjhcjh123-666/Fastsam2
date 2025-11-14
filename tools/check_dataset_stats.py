#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""快速检查数据集统计信息，验证数据完整性。

Usage:
    python tools/check_dataset_stats.py configs/rap_sam/rap_sam_r50_12e_adaptor.py
"""
import argparse
import sys
import os

import torch
from mmengine.config import Config
from mmengine.registry import DATASETS
from mmengine.logging import print_log

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dataset_stats(cfg_path, check_samples=100):
    """检查数据集统计信息。
    
    Args:
        cfg_path: 配置文件路径
        check_samples: 检查的样本数量
    """
    print("=" * 80)
    print("数据集统计信息检查")
    print("=" * 80)
    
    # 加载配置
    cfg = Config.fromfile(cfg_path)
    
    if not hasattr(cfg, 'train_dataloader') or cfg.train_dataloader is None:
        print("❌ 配置文件中没有 train_dataloader")
        return
    
    # 构建数据集
    print("\n[1/3] 正在构建数据集...")
    try:
        dataset = DATASETS.build(cfg.train_dataloader['dataset'])
        print(f"✅ 数据集构建成功")
        print(f"   类型: {type(dataset).__name__}")
        print(f"   总长度: {len(dataset):,}")
    except Exception as e:
        print(f"❌ 数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查子数据集
    print(f"\n[2/3] 检查子数据集信息...")
    if hasattr(dataset, 'datasets'):
        print(f"   包含 {len(dataset.datasets)} 个子数据集\n")
        
        total_original = 0
        total_effective = 0
        
        for i, sub_dataset in enumerate(dataset.datasets):
            # 获取实际数据集和重复次数
            actual_dataset = sub_dataset
            repeat_times = 1
            if hasattr(sub_dataset, 'dataset'):
                actual_dataset = sub_dataset.dataset
                repeat_times = getattr(sub_dataset, 'times', 1)
            
            dataset_name = type(actual_dataset).__name__
            
            # 获取数据集长度
            try:
                dataset_len = len(actual_dataset) if hasattr(actual_dataset, '__len__') else 0
            except:
                dataset_len = 0
            
            effective_len = dataset_len * repeat_times
            total_original += dataset_len
            total_effective += effective_len
            
            # 获取数据标签（如果有）
            data_tag = ''
            if hasattr(dataset, 'data_tag') and i < len(dataset.data_tag):
                data_tag = f" ({dataset.data_tag[i]})"
            
            print(f"   [{i+1}] {dataset_name}{data_tag}")
            print(f"       原始样本数: {dataset_len:,}")
            print(f"       重复次数: {repeat_times}x")
            print(f"       有效样本数: {effective_len:,}")
            
            # 检查过滤配置
            filter_info = []
            if hasattr(actual_dataset, 'filter_cfg') and actual_dataset.filter_cfg:
                filter_cfg = actual_dataset.filter_cfg
                if filter_cfg.get('filter_empty_gt', False):
                    filter_info.append("filter_empty_gt=True")
                if 'min_size' in filter_cfg:
                    filter_info.append(f"min_size={filter_cfg['min_size']}")
            
            if filter_info:
                print(f"       过滤配置: {', '.join(filter_info)}")
            
            # 检查 pipeline 中的过滤
            pipeline_filters = []
            if hasattr(actual_dataset, 'pipeline'):
                pipeline = actual_dataset.pipeline
                if hasattr(pipeline, 'transforms'):
                    transforms = pipeline.transforms
                    for transform in transforms:
                        transform_name = ''
                        if hasattr(transform, '__class__'):
                            transform_name = transform.__class__.__name__
                        elif hasattr(transform, 'type'):
                            transform_name = transform.type
                        
                        if 'Filter' in transform_name:
                            # 获取过滤参数
                            filter_params = {}
                            if hasattr(transform, 'min_gt_mask_area'):
                                filter_params['min_gt_mask_area'] = transform.min_gt_mask_area
                            if hasattr(transform, 'by_box'):
                                filter_params['by_box'] = transform.by_box
                            if hasattr(transform, 'by_mask'):
                                filter_params['by_mask'] = transform.by_mask
                            
                            if filter_params:
                                params_str = ', '.join([f"{k}={v}" for k, v in filter_params.items()])
                                pipeline_filters.append(f"{transform_name}({params_str})")
                            else:
                                pipeline_filters.append(transform_name)
            
            if pipeline_filters:
                print(f"       Pipeline过滤: {', '.join(pipeline_filters)}")
            
            print()
        
        print(f"   总计:")
        print(f"       原始样本数: {total_original:,}")
        print(f"       有效样本数: {total_effective:,}")
        print(f"       数据集长度: {len(dataset):,}")
        
        if len(dataset) != total_effective:
            print(f"   ⚠️  注意: 数据集长度 ({len(dataset):,}) != 有效样本数 ({total_effective:,})")
            print(f"       这可能是因为 ConcatOVDataset 的内部处理")
    else:
        print(f"   单一数据集，无子数据集")
    
    # 快速采样检查
    print(f"\n[3/3] 快速采样检查 (检查 {min(check_samples, len(dataset))} 个样本)...")
    success_count = 0
    error_count = 0
    empty_count = 0
    
    import random
    random.seed(42)  # 固定随机种子，便于复现
    indices = random.sample(range(len(dataset)), min(check_samples, len(dataset)))
    
    for idx in indices:
        try:
            sample = dataset[idx]
            
            # 检查样本是否为空
            if sample is None:
                empty_count += 1
                continue
            
            # 检查数据完整性
            has_data = False
            if isinstance(sample, dict):
                if 'inputs' in sample:
                    has_data = True
                if 'data_samples' in sample:
                    data_samples = sample['data_samples']
                    # 检查是否有有效的 GT
                    if hasattr(data_samples, 'gt_instances'):
                        gt = data_samples.gt_instances
                        if hasattr(gt, 'labels') and len(gt.labels) > 0:
                            has_data = True
                        elif hasattr(gt, 'masks'):
                            has_data = True
                    elif hasattr(data_samples, 'gt_sem_seg'):
                        has_data = True
            elif sample is not None:
                has_data = True
            
            if not has_data:
                empty_count += 1
            else:
                success_count += 1
                
        except Exception as e:
            error_count += 1
            if error_count <= 3:  # 只打印前3个错误
                print(f"   ⚠️  样本 {idx} 加载失败: {type(e).__name__}: {e}")
    
    print(f"\n   采样检查结果:")
    print(f"     成功: {success_count}/{check_samples} ({success_count/check_samples*100:.1f}%)")
    print(f"     空样本: {empty_count}/{check_samples} ({empty_count/check_samples*100:.1f}%)")
    print(f"     错误: {error_count}/{check_samples} ({error_count/check_samples*100:.1f}%)")
    
    # 总结和建议
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    
    print(f"\n✅ 数据集总长度: {len(dataset):,}")
    print(f"✅ 采样成功率: {success_count/check_samples*100:.1f}%")
    
    if empty_count > check_samples * 0.15:  # 空样本超过15%
        print(f"\n⚠️  警告: 空样本比例较高 ({empty_count/check_samples*100:.1f}%)")
        print(f"   建议:")
        print(f"     1. 检查过滤配置，考虑减小 min_size 或 min_gt_mask_area")
        print(f"     2. 检查数据标注是否完整")
        print(f"     3. 查看训练日志中的过滤信息")
    
    if error_count > check_samples * 0.1:  # 错误超过10%
        print(f"\n⚠️  警告: 数据加载错误较多 ({error_count/check_samples*100:.1f}%)")
        print(f"   建议:")
        print(f"     1. 检查数据路径是否正确")
        print(f"     2. 检查标注文件是否完整")
        print(f"     3. 检查图像文件是否存在")
    
    if success_count / check_samples > 0.85:  # 成功率超过85%
        print(f"\n✅ 数据质量良好，可以开始训练")
    
    print("\n" + "=" * 80)
    print("如何监控训练时的数据使用情况")
    print("=" * 80)
    
    print("""
1. 查看训练日志中的过滤信息:
   训练时会输出类似:
   "The number of samples before and after filtering: X / Y"
   
   这表明:
   - Y: 原始数据集大小
   - X: 过滤后的数据集大小
   - 过滤率 = (Y - X) / Y * 100%
   
   如果过滤率 > 20%，可能需要调整过滤配置

2. 监控训练损失:
   - 如果损失不下降，可能是数据质量有问题
   - 如果损失波动很大，可能是数据不平衡

3. 检查训练速度:
   - 如果训练很慢，可能是数据加载有问题
   - 检查 num_workers 是否合适

4. 使用 TensorBoard 或 WandB:
   - 监控每个 epoch 的数据使用情况
   - 检查数据分布是否正常
    """)


def main():
    parser = argparse.ArgumentParser(description='检查数据集统计信息')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--check-samples', type=int, default=100,
                        help='采样检查的样本数量 (默认: 100)')
    args = parser.parse_args()
    
    check_dataset_stats(args.config, args.check_samples)


if __name__ == '__main__':
    main()

