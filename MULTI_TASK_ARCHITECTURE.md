# 多任务架构说明文档

## 概述

本项目已重构为支持多任务分割的统一架构，一个模型可以处理以下任务：

1. **图像交互分割**：支持点、框、文本提示
2. **视频交互分割**：支持点、框、文本提示（带时序记忆）
3. **视频对象分割 (VOS)**：自动 mask 传播和时序一致性
4. **全景分割**：统一的实例和语义分割

## 架构特点

✅ **自动任务检测**：根据数据样本自动识别任务类型，无需手动指定  
✅ **模块化设计**：各组件可独立启用/禁用，灵活配置  
✅ **向后兼容**：不启用多任务时，模型行为与之前版本完全一致  
✅ **统一训练**：支持混合数据集联合训练，自动处理不同任务

## 核心组件

### 1. TaskRouter (`seg/models/utils/task_router.py`)

自动检测任务类型并路由到相应的处理路径。

**功能**：
- 自动检测任务类型（交互图像/交互视频/VOS/全景）
- 检测提示类型（点/框/文本/无）
- 根据任务类型配置 decoder stages 和 memory 使用

**使用方式**：
```python
task_router = TaskRouter(
    feat_channels=256,
    num_decoder_stages=3,
    enable_streaming_memory=True
)
routing_config = task_router(data_samples, prompts, mode='train')
```

### 2. StreamingMemoryAdapter (`seg/models/utils/memory_adapter.py`)

用于 VOS 任务的长期和短期记忆管理。

**功能**：
- 维护关键帧的长期记忆
- 维护最近帧的短期记忆
- 支持语言引导的记忆检索
- 自适应记忆更新策略

**使用方式**：
```python
memory = StreamingMemoryAdapter(
    feat_channels=256,
    long_mem_size=10,
    short_mem_size=5
)
memory.update(frame_id, instance_embed, mask, instance_id)
memory_embed, memory_mask = memory.fetch(frame_id, instance_id)
```

### 3. PromptFusion (`seg/models/utils/prompt_fusion.py`)

融合多种提示类型（点、框、文本）用于交互分割。

**功能**：
- 融合点、框、文本提示
- 支持多模态交互分割
- 文本-视觉对齐损失

**使用方式**：
```python
prompt_fusion = PromptFusion(feat_channels=256, num_heads=8)
fused_prompts = prompt_fusion(point_embed, box_embed, text_embed)
```

## 模型架构

### RapSAM Detector (`seg/models/detectors/rapsam.py`)

主检测器，集成了所有多任务组件。

**新增参数**：
- `use_task_router`: 是否使用任务路由
- `task_router`: 任务路由配置
- `use_streaming_memory`: 是否使用流式记忆
- `streaming_memory`: 流式记忆配置
- `use_prompt_fusion`: 是否使用提示融合
- `prompt_fusion`: 提示融合配置

**工作流程**：
1. 在 `loss()` 或 `predict()` 中自动检测任务类型
2. 通过 TaskRouter 获取路由配置
3. 将路由配置传递给 Head
4. Head 根据配置使用相应的组件

### RapSAMVideoHead (`seg/models/heads/rapsam_head.py`)

Head 层，集成了多任务处理逻辑。

**新增功能**：
- `set_routing_config()`: 接收路由配置
- 在 `forward()` 中根据路由配置应用 PromptFusion 和 StreamingMemory
- 支持任务特定的损失计算

## 配置文件

### 示例配置 (`configs/rap_sam/rap_sam_r50_12e_adaptor.py`)

```python
# 多任务组件配置
task_router = dict(
    type='TaskRouter',
    feat_channels=256,
    num_decoder_stages=3,
    enable_streaming_memory=True,
    interactive_stages=3,
    vos_stages=3,
    panoptic_stages=3
)

streaming_memory = dict(
    type='StreamingMemoryAdapter',
    feat_channels=256,
    long_mem_size=10,
    short_mem_size=5,
    update_strategy='adaptive'
)

prompt_fusion = dict(
    type='PromptFusion',
    feat_channels=256,
    num_heads=8,
    dropout=0.1
)

model = dict(
    type='RapSAM',
    # ... 其他配置 ...
    use_task_router=True,
    task_router=task_router,
    use_streaming_memory=True,
    streaming_memory=streaming_memory,
    use_prompt_fusion=True,
    prompt_fusion=prompt_fusion,
)
```

## 任务检测逻辑

模型会根据数据样本自动检测任务类型：

1. **交互任务**：
   - 检测到 `gt_instances_collected` 或 prompts（点/框/文本）
   - 图像数据 → 图像交互分割
   - 视频数据 → 视频交互分割

2. **VOS 任务**：
   - 检测到 `instances_ids` 在视频数据中
   - 启用 mask 传播和时序一致性

3. **全景分割**：
   - 默认任务，当没有检测到其他任务类型时

## 训练建议

1. **多任务联合训练**：
   - 使用混合数据集（COCO + YouTube-VIS + DAVIS + RefCOCO）
   - 模型会自动识别每个样本的任务类型并应用相应的处理

2. **任务特定配置**：
   - 可以通过配置文件调整不同任务的 decoder stages
   - 可以启用/禁用特定组件（memory、prompt fusion）

3. **性能优化**：
   - 对于实时推理，可以减少 decoder stages
   - 对于 VOS，可以调整 memory size

## 后续改进

1. **完善 PromptFusion**：
   - 实现从 SAMPromptEncoder 提取提示嵌入
   - 完善文本-视觉对齐损失

2. **完善 StreamingMemory**：
   - 实现 VOS 中的实际特征增强
   - 完善 DPSR 损失计算

3. **性能优化**：
   - 实现低秩注意力
   - 添加 token pruning

## 使用示例

```python
# 训练时，模型会自动检测任务类型
model = build_model(cfg.model)
losses = model.loss(batch_inputs, batch_data_samples)

# 推理时，根据输入数据自动路由
results = model.predict(batch_inputs, batch_data_samples)
```

## 注意事项

1. 所有多任务组件都是可选的，可以通过配置启用/禁用
2. 向后兼容：如果不启用多任务组件，模型行为与之前一致
3. 任务检测基于数据样本的结构，确保数据格式正确

