#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""检查数据集完整性，验证是否有大量数据被过滤掉。

Usage:
    python tools/check_dataset_integrity.py configs/rap_sam/rap_sam_r50_12e_adaptor.py
"""
import argparse
import sys
import os

import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from mmengine.logging import print_log

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dataset_filtering(cfg_path, num_samples=1000):
    """检查数据集过滤情况。
    
    Args:
        cfg_path: 配置文件路径
        num_samples: 采样检查的样本数量
    """
    print("=" * 80)
    print("数据集完整性检查")
    print("=" * 80)
    
    # 加载配置
    cfg = Config.fromfile(cfg_path)
    
    if not hasattr(cfg, 'train_dataloader') or cfg.train_dataloader is None:
        print("❌ 配置文件中没有 train_dataloader")
        return
    
    # 构建数据集
    print("\n[1/4] 正在构建数据集...")
    dataset = DATASETS.build(cfg.train_dataloader['dataset'])
    print(f"✅ 数据集构建成功")
    print(f"   类型: {type(dataset).__name__}")
    print(f"   总长度: {len(dataset)}")
    
    # 检查是否是 ConcatOVDataset
    if hasattr(dataset, 'datasets'):
        print(f"\n[2/4] 检查子数据集...")
        print(f"   包含 {len(dataset.datasets)} 个子数据集")
        
        # 统计每个子数据集的信息
        for i, sub_dataset in enumerate(dataset.datasets):
            # 如果是 RepeatDataset，获取实际数据集
            actual_dataset = sub_dataset
            repeat_times = 1
            if hasattr(sub_dataset, 'dataset'):
                actual_dataset = sub_dataset.dataset
                repeat_times = sub_dataset.times if hasattr(sub_dataset, 'times') else 1
            
            dataset_name = type(actual_dataset).__name__
            dataset_len = len(actual_dataset) if hasattr(actual_dataset, '__len__') else 0
            effective_len = dataset_len * repeat_times
            
            print(f"\n   子数据集 {i+1}: {dataset_name}")
            print(f"     原始长度: {dataset_len}")
            print(f"     重复次数: {repeat_times}")
            print(f"     有效长度: {effective_len}")
            
            # 检查过滤配置
            if hasattr(actual_dataset, 'filter_cfg') and actual_dataset.filter_cfg:
                print(f"     过滤配置: {actual_dataset.filter_cfg}")
            
            # 检查 pipeline 中的过滤
            if hasattr(actual_dataset, 'pipeline'):
                pipeline = actual_dataset.pipeline
                # Compose 对象有 transforms 属性
                if hasattr(pipeline, 'transforms'):
                    transforms = pipeline.transforms
                elif hasattr(pipeline, '__iter__'):
                    # 如果是可迭代的，转换为列表
                    try:
                        transforms = list(pipeline)
                    except:
                        transforms = []
                else:
                    transforms = []
                
                for j, transform in enumerate(transforms):
                    transform_type = ''
                    if isinstance(transform, dict):
                        transform_type = transform.get('type', '')
                    elif hasattr(transform, 'type'):
                        transform_type = transform.type
                    elif hasattr(transform, '__class__'):
                        transform_type = transform.__class__.__name__
                    
                    if 'Filter' in transform_type:
                        if isinstance(transform, dict):
                            print(f"     Pipeline[{j}]: {transform}")
                        else:
                            print(f"     Pipeline[{j}]: {transform_type}")
    
    # 采样检查数据加载
    print(f"\n[3/4] 采样检查数据加载 (检查 {min(num_samples, len(dataset))} 个样本)...")
    success_count = 0
    error_count = 0
    empty_count = 0
    
    # 随机采样
    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        try:
            sample = dataset[idx]
            
            # 检查样本是否为空
            if sample is None:
                empty_count += 1
                continue
            
            # 检查是否有有效数据
            if 'data_samples' in sample:
                data_samples = sample['data_samples']
                if hasattr(data_samples, 'gt_instances'):
                    gt_instances = data_samples.gt_instances
                    if hasattr(gt_instances, 'labels') and len(gt_instances.labels) == 0:
                        empty_count += 1
                        continue
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # 只打印前5个错误
                print(f"   ⚠️  样本 {idx} 加载失败: {e}")
    
    print(f"\n   检查结果:")
    print(f"     成功加载: {success_count}/{num_samples} ({success_count/num_samples*100:.1f}%)")
    print(f"     空样本: {empty_count}/{num_samples} ({empty_count/num_samples*100:.1f}%)")
    print(f"     错误: {error_count}/{num_samples} ({error_count/num_samples*100:.1f}%)")
    
    # 检查训练时的过滤
    print(f"\n[4/4] 检查训练时的数据过滤...")
    
    # 模拟一个训练迭代
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.train_dataloader.get('batch_size', 1),
            num_workers=cfg.train_dataloader.get('num_workers', 2),
            shuffle=False,
            sampler=torch.utils.data.SequentialSampler(dataset)
        )
        
        # 检查前几个 batch
        batch_count = 0
        total_samples = 0
        valid_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # 只检查前10个batch
                break
            
            batch_count += 1
            if isinstance(batch, (list, tuple)):
                batch_size = len(batch)
            elif isinstance(batch, dict):
                if 'inputs' in batch:
                    if isinstance(batch['inputs'], torch.Tensor):
                        batch_size = batch['inputs'].shape[0]
                    else:
                        batch_size = len(batch['inputs'])
                else:
                    batch_size = 1
            else:
                batch_size = 1
            
            total_samples += batch_size
            
            # 检查 batch 中的有效样本
            if isinstance(batch, dict) and 'data_samples' in batch:
                data_samples = batch['data_samples']
                if isinstance(data_samples, list):
                    for ds in data_samples:
                        if hasattr(ds, 'gt_instances'):
                            if hasattr(ds.gt_instances, 'labels'):
                                if len(ds.gt_instances.labels) > 0:
                                    valid_samples += 1
                                else:
                                    pass  # 空样本
                            else:
                                valid_samples += 1  # 没有labels属性，假设有效
                        else:
                            valid_samples += 1  # 没有gt_instances，假设有效
                else:
                    valid_samples += batch_size
            else:
                valid_samples += batch_size
        
        print(f"   检查了 {batch_count} 个 batch")
        print(f"   总样本数: {total_samples}")
        print(f"   有效样本数: {valid_samples}")
        print(f"   有效率: {valid_samples/total_samples*100:.1f}%")
        
    except Exception as e:
        print(f"   ⚠️  无法创建 DataLoader: {e}")
    
    # 总结
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    
    print(f"\n✅ 数据集总长度: {len(dataset)}")
    print(f"✅ 采样检查成功率: {success_count/num_samples*100:.1f}%")
    
    if empty_count > num_samples * 0.1:  # 如果空样本超过10%
        print(f"⚠️  警告: 空样本比例较高 ({empty_count/num_samples*100:.1f}%)")
        print(f"   建议检查过滤配置，可能需要调整 min_size 或 min_gt_mask_area")
    
    if error_count > num_samples * 0.05:  # 如果错误超过5%
        print(f"⚠️  警告: 数据加载错误较多 ({error_count/num_samples*100:.1f}%)")
        print(f"   建议检查数据路径和标注文件")
    
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    
    print("""
1. 如果空样本比例 > 10%:
   - 检查 filter_cfg 中的 min_size 是否过大
   - 检查 FilterAnnotationsHB 中的 min_gt_mask_area 是否过大
   - 考虑减小这些阈值，例如: min_size=16, min_gt_mask_area=16

2. 监控训练日志:
   - 查看训练日志中的 "The number of samples before and after filtering"
   - 确认过滤比例是否合理（通常应该 < 20%）

3. 检查数据质量:
   - 确认所有数据集路径正确
   - 确认标注文件完整
   - 检查是否有损坏的图像

4. 训练时监控:
   - 使用 tensorboard 或 wandb 监控训练
   - 确认损失正常下降
   - 检查是否有大量 batch 被跳过
    """)


def check_filtering_in_logs(log_file):
    """从训练日志中提取过滤信息。
    
    Args:
        log_file: 训练日志文件路径
    """
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print("\n" + "=" * 80)
    print("从训练日志中提取过滤信息")
    print("=" * 80)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 查找过滤信息
    filter_lines = []
    for line in lines:
        if 'filtering' in line.lower() or 'before and after' in line.lower():
            filter_lines.append(line.strip())
    
    if filter_lines:
        print(f"\n找到 {len(filter_lines)} 条过滤信息:")
        for line in filter_lines[:20]:  # 只显示前20条
            print(f"  {line}")
    else:
        print("\n⚠️  日志中没有找到过滤信息")
        print("   训练时应该会输出类似: 'The number of samples before and after filtering: X / Y'")


def main():
    parser = argparse.ArgumentParser(description='检查数据集完整性')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='采样检查的样本数量 (默认: 1000)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='训练日志文件路径（可选）')
    args = parser.parse_args()
    
    # 检查数据集
    check_dataset_filtering(args.config, args.num_samples)
    
    # 如果提供了日志文件，检查日志
    if args.log_file:
        check_filtering_in_logs(args.log_file)


if __name__ == '__main__':
    main()

