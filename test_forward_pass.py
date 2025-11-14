#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多任务模型的前向传播和loss计算
验证各个任务类型的loss masking机制是否正常工作
"""

import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import MODELS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData


def create_dummy_batch(task_type='interactive_image', batch_size=2, num_frames=1):
    """创建虚拟batch数据用于测试
    
    Args:
        task_type: 任务类型 ('interactive_image', 'interactive_video', 'vos', 'panoptic')
        batch_size: batch大小
        num_frames: 视频帧数（仅视频任务）
    
    Returns:
        batch_inputs, batch_data_samples
    """
    # 创建输入图像
    if task_type in ['interactive_video', 'vos']:
        # 视频数据: [B, T, C, H, W]
        batch_inputs = torch.randn(batch_size, num_frames, 3, 512, 512).cuda()
    else:
        # 图像数据: [B, C, H, W]
        batch_inputs = torch.randn(batch_size, 3, 512, 512).cuda()
    
    batch_data_samples = []
    
    for i in range(batch_size):
        data_sample = DetDataSample()
        
        # 设置metainfo
        data_sample.set_metainfo({
            'img_shape': (512, 512),
            'ori_shape': (512, 512),
            'pad_shape': (512, 512),
            'img_id': i,
        })
        
        # 创建GT instances
        gt_instances = InstanceData()
        num_instances = 5
        
        # 添加masks - 必须使用BitmapMasks对象
        from mmdet.structures.mask import BitmapMasks
        masks_np = np.random.randint(0, 2, (num_instances, 512, 512), dtype=np.uint8)
        gt_instances.masks = BitmapMasks(masks_np, height=512, width=512)
        
        # 添加labels
        gt_instances.labels = torch.randint(0, 80, (num_instances,)).cuda()
        
        # 添加bboxes
        bboxes = torch.rand(num_instances, 4).cuda() * 512
        bboxes[:, 2:] = bboxes[:, 2:] + bboxes[:, :2]  # 确保x2>x1, y2>y1
        gt_instances.bboxes = bboxes
        
        # 根据任务类型添加特定数据
        if task_type in ['interactive_image', 'interactive_video']:
            # 交互任务：添加prompt
            gt_instances_collected = InstanceData()
            # 添加点击坐标 - 修正：应该是 (num_instances, 2) 维度，prepare_for_dn_mo 会 stack 成 (B, N, 2)
            point_coords = torch.rand(num_instances, 2).cuda() * 512
            gt_instances_collected.point_coords = point_coords
            # 添加点击标签 (1=前景, 0=背景) - (num_instances,)
            gt_instances_collected.pb_labels = torch.ones(num_instances, dtype=torch.long).cuda()
            data_sample.gt_instances_collected = gt_instances_collected
            
            # 🔥 关键：给所有交互样本都添加文本提示，确保loss_text_visual能激活
            # 不同样本使用不同的text，模拟真实场景
            text_prompts = [
                'a person wearing red shirt',
                'a dog running in the park',
                'a car on the street',
            ]
            data_sample.set_metainfo({'text': text_prompts[i % len(text_prompts)]})
        
        elif task_type == 'vos':
            # VOS任务：添加实例ID用于跟踪
            gt_instances.instances_ids = torch.arange(num_instances).cuda()
        
        # 对于panoptic任务，不需要额外数据
        
        data_sample.gt_instances = gt_instances
        batch_data_samples.append(data_sample)
    
    # 如果是视频任务，需要特殊处理
    if task_type in ['interactive_video', 'vos']:
        from mmdet.structures import TrackDataSample
        from mmdet.structures.mask import BitmapMasks
        track_samples = []
        
        for i in range(batch_size):
            track_sample = TrackDataSample()
            # 创建多帧数据
            video_data_samples = []
            
            # 对于VOS任务，所有帧的labels必须一致
            # 提前生成固定的labels
            fixed_labels = torch.randint(0, 80, (num_instances,)).cuda()
            
            for t in range(num_frames):
                # 为每一帧创建独立的数据样本
                frame_sample = DetDataSample()
                frame_sample.set_metainfo({
                    'img_shape': (512, 512),
                    'ori_shape': (512, 512),
                    'pad_shape': (512, 512),
                    'img_id': i * num_frames + t,
                    'frame_id': t,
                })
                
                # 创建该帧的GT instances
                frame_instances = InstanceData()
                masks_np = np.random.randint(0, 2, (num_instances, 512, 512), dtype=np.uint8)
                frame_instances.masks = BitmapMasks(masks_np, height=512, width=512)
                
                # 使用固定的labels（VOS要求所有帧labels一致）
                frame_instances.labels = fixed_labels.clone()
                
                bboxes = torch.rand(num_instances, 4).cuda() * 512
                bboxes[:, 2:] = bboxes[:, 2:] + bboxes[:, :2]
                frame_instances.bboxes = bboxes
                
                # 只有VOS任务需要实例ID
                if task_type == 'vos':
                    frame_instances.instances_ids = torch.arange(num_instances).cuda()
                
                # 交互视频任务：添加prompt（仅第一帧）
                if task_type == 'interactive_video' and t == 0:
                    gt_instances_collected = InstanceData()
                    # 修正维度：(num_instances, 2)
                    point_coords = torch.rand(num_instances, 2).cuda() * 512
                    gt_instances_collected.point_coords = point_coords
                    # 修正维度：(num_instances,)
                    gt_instances_collected.pb_labels = torch.ones(num_instances, dtype=torch.long).cuda()
                    frame_sample.gt_instances_collected = gt_instances_collected
                    
                    # 🔥 给所有视频交互样本都添加文本提示
                    text_prompts = [
                        'a person wearing red shirt',
                        'a dog running in the park',
                    ]
                    frame_sample.set_metainfo({'text': text_prompts[i % len(text_prompts)]})
                
                frame_sample.gt_instances = frame_instances
                video_data_samples.append(frame_sample)
            
            track_sample.video_data_samples = video_data_samples
            track_samples.append(track_sample)
        
        batch_data_samples = track_samples
    
    return batch_inputs, batch_data_samples


def test_forward_pass(config_path='/mnt/chenjiahui/Fastsam2-main/configs/rap_sam/rap_sam_r50_12e_adaptor.py'):
    """测试前向传播和loss计算"""
    
    print("=" * 80)
    print("测试多任务模型前向传播")
    print("=" * 80)
    
    # 加载配置
    print("\n[1] 加载配置文件...")
    try:
        cfg = Config.fromfile(config_path)
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False
    
    # 构建模型
    print("\n[2] 构建模型...")
    try:
        model = MODELS.build(cfg.model)
        
        # 🔧 关键修复：将SyncBatchNorm转换为BatchNorm用于单GPU测试
        # 这样可以避免分布式初始化的要求
        from torch.nn import SyncBatchNorm
        print("   转换 SyncBatchNorm -> BatchNorm (单GPU模式)...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # 实际上我们需要反向转换，将SyncBN转为普通BN
        def convert_sync_bn_to_bn(module):
            """递归转换SyncBatchNorm为BatchNorm"""
            import torch.nn as nn
            module_output = module
            if isinstance(module, SyncBatchNorm):
                module_output = nn.BatchNorm2d(
                    module.num_features,
                    module.eps,
                    module.momentum,
                    module.affine,
                    module.track_running_stats
                )
                if module.affine:
                    with torch.no_grad():
                        module_output.weight = module.weight
                        module_output.bias = module.bias
                module_output.running_mean = module.running_mean
                module_output.running_var = module.running_var
                module_output.num_batches_tracked = module.num_batches_tracked
            for name, child in module.named_children():
                module_output.add_module(name, convert_sync_bn_to_bn(child))
            del module
            return module_output
        
        model = convert_sync_bn_to_bn(model)
        
        model = model.cuda()
        model.train()
        print("✓ 模型构建成功")
        print(f"   - 使用TaskRouter: {model.use_task_router}")
        print(f"   - 使用StreamingMemory: {model.use_streaming_memory}")
        print(f"   - 使用PromptFusion: {model.use_prompt_fusion}")
        print(f"   - Loss权重配置: {len(model.task_loss_weights)} 个任务类型")
    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试不同任务类型
    task_types = [
        ('interactive_image', '图像交互分割', 1),
        ('interactive_video', '视频交互分割', 3),
        ('vos', '视频对象分割', 3),
        ('panoptic', '全景分割', 1),
    ]
    
    all_passed = True
    
    for task_type, task_name, num_frames in task_types:
        print(f"\n[3] 测试任务类型: {task_name} ({task_type})")
        print("-" * 60)
        
        try:
            # 创建虚拟数据
            batch_inputs, batch_data_samples = create_dummy_batch(
                task_type=task_type, 
                batch_size=2, 
                num_frames=num_frames
            )
            print(f"   ✓ 创建测试数据: batch_size=2, num_frames={num_frames}")
            print(f"     输入形状: {batch_inputs.shape}")
            
            # 前向传播 - loss计算
            with torch.cuda.amp.autocast(enabled=False):  # 不使用混合精度以便调试
                losses = model.loss(batch_inputs, batch_data_samples)
            
            print(f"   ✓ 前向传播成功")
            print(f"\n   计算的Loss:")
            
            total_loss_value = 0.0
            active_losses = []
            masked_losses = []
            
            for loss_name, loss_value in losses.items():
                loss_val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
                total_loss_value += loss_val
                
                # 判断loss是否被激活（权重>0）
                if loss_val > 1e-6:  # 非零loss
                    status = "✓ 激活"
                    active_losses.append(loss_name)
                else:
                    status = "○ 屏蔽"
                    masked_losses.append(loss_name)
                
                print(f"     {status} {loss_name:25s}: {loss_val:>12.6f}")
            
            print(f"\n   总Loss值: {total_loss_value:.6f}")
            print(f"   激活的Loss ({len(active_losses)}个): {', '.join(active_losses)}")
            print(f"   屏蔽的Loss ({len(masked_losses)}个): {', '.join(masked_losses)}")
            
            # 验证loss masking是否正确
            # 提取基础loss名称（去掉decoder层前缀）
            def get_base_loss_name(loss_name):
                return loss_name.split('.')[-1] if '.' in loss_name else loss_name
            
            # 定义每个任务应该激活的loss（基础loss + 任务特定loss）
            expected_active_base = {
                'interactive_image': ['loss_mask', 'loss_dice', 'loss_iou', 'loss_prompt_align', 'loss_text_visual'],
                'interactive_video': ['loss_mask', 'loss_dice', 'loss_iou', 'loss_prompt_align', 'loss_text_visual', 'loss_temporal'],
                'vos': ['loss_cls', 'loss_mask', 'loss_dice', 'loss_dpsr', 'loss_temporal', 'loss_memory_align'],
                'panoptic': ['loss_cls', 'loss_mask', 'loss_dice', 'loss_panoptic'],
            }
            
            # 定义每个任务应该屏蔽的loss
            expected_masked_base = {
                'interactive_image': ['loss_cls', 'loss_dpsr', 'loss_temporal', 'loss_memory_align', 'loss_panoptic'],
                'interactive_video': ['loss_cls', 'loss_dpsr', 'loss_memory_align', 'loss_panoptic'],
                'vos': ['loss_iou', 'loss_prompt_align', 'loss_text_visual', 'loss_panoptic'],
                'panoptic': ['loss_iou', 'loss_dpsr', 'loss_temporal', 'loss_prompt_align', 'loss_text_visual', 'loss_memory_align'],
            }
            
            # 检查是否有应该激活但未激活的loss
            expected_active = expected_active_base.get(task_type, [])
            expected_masked = expected_masked_base.get(task_type, [])
            active_base_losses = [get_base_loss_name(l) for l in active_losses]
            masked_base_losses = [get_base_loss_name(l) for l in masked_losses]
            
            # 去重（因为d0/d1/d2会重复）
            active_base_losses_unique = list(set(active_base_losses))
            masked_base_losses_unique = list(set(masked_base_losses))
            
            # 检查缺失的激活loss
            missing_active = [l for l in expected_active if l not in active_base_losses_unique]
            # 检查不应该激活的loss
            unexpected_active = [l for l in expected_masked if l in active_base_losses_unique]
            # 检查应该屏蔽但未屏蔽的loss
            missing_masked = [l for l in expected_masked if l not in masked_base_losses_unique and l not in active_base_losses_unique]
            # 检查不应该屏蔽的loss
            unexpected_masked = [l for l in expected_active if l in masked_base_losses_unique]
            
            # 打印详细的验证结果
            validation_passed = True
            if missing_active:
                print(f"\n   ❌ 错误: 以下loss应该激活但未激活: {missing_active}")
                validation_passed = False
            if unexpected_active:
                print(f"   ❌ 错误: 以下loss不应该激活但被激活: {unexpected_active}")
                validation_passed = False
            if unexpected_masked:
                print(f"   ❌ 错误: 以下loss不应该屏蔽但被屏蔽: {unexpected_masked}")
                validation_passed = False
            
            if validation_passed:
                print(f"\n   ✅ Loss验证通过: 所有loss的激活/屏蔽状态正确")
            else:
                all_passed = False
            
            # 测试反向传播
            total_loss = sum(losses.values())
            total_loss.backward()
            print(f"\n   ✓ 反向传播成功")
            
            # 检查梯度
            has_grad = False
            no_grad_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_grad = True
                    elif param.grad is None:
                        no_grad_params.append(name)
            
            if has_grad:
                print(f"   ✓ 梯度计算正常")
            else:
                print(f"   ✗ 警告: 没有参数有梯度")
                all_passed = False
            
            if no_grad_params and len(no_grad_params) < 10:  # 只显示前几个
                print(f"   ⚠ 部分参数无梯度: {no_grad_params[:5]}...")
            
            # 清理梯度
            model.zero_grad()
            
            print(f"\n   ✅ {task_name} 测试通过")
            
        except Exception as e:
            print(f"\n   ✗ {task_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有测试通过！多任务loss masking机制工作正常。")
    else:
        print("⚠️  部分测试失败，请检查上面的错误信息。")
    print("=" * 80)
    
    return all_passed


if __name__ == '__main__':
    import sys
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    success = test_forward_pass()
    
    sys.exit(0 if success else 1)

