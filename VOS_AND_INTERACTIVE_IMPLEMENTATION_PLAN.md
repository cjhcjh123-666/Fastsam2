# VOS 和交互式分割实现方案总结

## 一、项目现状分析

### 1.1 当前支持的任务

| 任务类型 | 数据集 | 模型 | 配置文件 | 状态 |
|---------|--------|------|---------|------|
| **图像全景分割** | COCO Panoptic | `Mask2formerVideo` | `eval_rap_sam_coco.py` | ✅ 已支持 |
| **视频实例分割 (VIS)** | YouTube-VIS 2019/2021 | `Mask2formerVideoMinVIS` | `eval_rap_sam_yt19.py` | ✅ 已支持 |
| **视频全景分割** | COCO Panoptic Video | `Mask2formerVideo` | - | ⚠️ 框架支持，需配置 |
| **交互式分割** | COCO + SAM数据 | `Mask2formerVideo` | - | ⚠️ 部分支持（点、框），缺文本 |
| **VOS** | DAVIS | `Mask2formerVideo` | - | ❌ 未实现 |

### 1.2 代码结构

- **基础模型**: `seg/models/detectors/mask2former_vid.py` - 通用视频模型
- **VIS专用模型**: `seg/models/detectors/mask2former_vid_minvis.py` - 继承自基础模型，包含tube分割和query匹配
- **数据集拼接**: `seg/datasets/concat_dataset.py` - 已支持多数据集联合训练，通过 `data_tag` 区分
- **提示编码**: `seg/models/necks/sam_pe.py` - 支持点、框、mask提示编码
- **提示融合**: `seg/models/utils/prompt_fusion.py` - 已存在，支持文本提示融合

## 二、实现方案

### 2.1 数据集使用策略

**两个数据集，三种操作：**

1. **COCO Panoptic Video**
   - 用途1: 视频全景分割（`panoptic_on=True, num_stuff_classes=53`）
   - 用途2: 视频交互分割（点、框、text）+ SAM数据 + RefCOCO数据
   - 图像模式：天然支持（`num_frames=0` 时自动切换）

2. **DAVIS**
   - 用途: VOS 训练和推理
   - 特征：第一帧提供 mask，需要跨帧跟踪

### 2.2 任务类型判断逻辑

#### 判断位置

**推荐位置**: `seg/models/detectors/mask2former_vid.py` 的 `predict` 方法（158行附近）

**判断依据**:
1. **data_tag**: 通过 `ConcatOVDataset` 设置的标签
   - `'davis'` → VOS
   - `'yt19'`, `'yt21'` → VIS
   - `'sam'`, `'refcoco'` → 交互式分割
   - `'coco'` → 全景分割

2. **数据结构特征**:
   - `TrackDataSample` → 视频任务
   - `gt_instances_collected` 存在 → 交互式分割
   - `gt_instances.instances_ids` 存在 → VOS
   - `num_frames > 0` → 视频模式

#### 判断流程

```
输入数据
  ↓
检查 data_tag
  ↓
├─ 'davis' → VOS模式
│   └─ 检查第一帧是否有 instances_ids
│   └─ 执行第一帧mask匹配
│
├─ 'yt19'/'yt21' → VIS模式
│   └─ 使用 Mask2formerVideoMinVIS 的 tube 分割逻辑
│
├─ 'sam'/'refcoco' → 交互式分割模式
│   └─ 检查 gt_instances_collected
│   └─ 使用 prepare_for_dn_mo 转换提示
│
└─ 其他 → 全景分割模式
    └─ 正常处理
```

## 三、需要实现的功能

### 3.1 VOS 支持

#### 核心逻辑
1. **第一帧匹配**: 从第一帧的 GT mask 匹配到预测实例
   - 计算 IoU 矩阵
   - 贪心匹配或匈牙利算法匹配
   - 只保留匹配到的 query

2. **跨帧跟踪**: 复用现有的 `match_from_embeds` 逻辑
   - 使用 query features 进行跨帧匹配
   - 保持实例 ID 一致性

3. **实现位置**:
   - 在 `mask2former_vid.py` 的 `predict` 方法中添加判断
   - 添加 `_vos_first_frame_match` 方法
   - 复用 `Mask2formerVideoMinVIS` 的 tube 分割逻辑（长视频）

#### 数据格式
- DAVIS 数据集已配置，提供 `instances_ids`
- `preprocess_video_panoptic_gt` 已支持 `gt_instance_ids`

### 3.2 文本提示支持

#### 当前状态
- `PromptFusion` 模块已存在
- `TextEncoder` 已实现（支持 CLIP text encoder）
- RefCOCO 数据集已配置
- ✅ `SAMPromptEncoder` 已扩展支持文本提示
- ✅ `VLMTextGenerator` 已实现（可选VLM生成文本）

#### 实现内容
1. **在 `SAMPromptEncoder` 中集成文本**:
   - ✅ 扩展 `forward` 方法，支持 `with_text` 参数
   - ✅ 集成 `PromptFusion` 融合文本 embedding
   - ✅ 支持从 `instances.text` 或直接传入文本

2. **VLM文本生成器**:
   - ✅ 创建 `VLMTextGenerator` 模块
   - ✅ 支持使用VLM（如CoCa）为实例生成文本描述
   - ✅ 支持从类别ID生成文本（fallback）
   - ✅ 可用于训练时（为COCO实例生成文本）或推理时（为检测实例生成文本）

3. **在训练 pipeline 中**:
   - 确保 RefCOCO 的文本数据正确传递
   - 在 head 中处理文本提示
   - 可选：使用VLM为COCO实例生成文本描述

4. **推理时**:
   - 用户提供文本描述（直接输入）
   - 或使用VLM为检测到的实例生成文本描述
   - 通过 CLIP text encoder 编码
   - 与点/框 embedding 融合

### 3.2.1 VLM文本生成的使用场景

#### 场景1：训练时为COCO实例生成文本
对于COCO等没有文本描述的数据集，可以使用VLM为每个实例生成文本描述：

```python
# 在数据预处理pipeline中
vlm_generator = VLMTextGenerator(
    vlm_cfg=dict(type='CoCa', ...),  # 配置VLM模型
    use_class_names=True,  # 如果没有VLM，使用类别名称
)

# 为实例生成文本
texts = vlm_generator.generate_for_instances(
    instances=gt_instances,
    image=image_tensor,
    image_size=(h, w)
)
gt_instances.text = texts
```

#### 场景2：推理时为检测实例生成文本
在推理时，可以为检测到的实例生成文本描述，用于后续的交互式分割：

```python
# 检测后生成文本
detected_instances = model.predict(...)
texts = vlm_generator.generate_for_instances(
    instances=detected_instances,
    image=image_tensor,
    image_size=(h, w)
)
# 使用生成的文本进行交互式分割
```

#### 配置示例
```python
# 在模型配置中启用文本提示
prompt_encoder=dict(
    type='SAMPromptEncoder',
    use_text_prompt=True,  # 启用文本提示
    prompt_fusion=dict(
        type='PromptFusion',
        feat_channels=256,
        use_text_encoder=True,
        text_encoder=dict(
            type='TextEncoder',
            text_model_cfg=dict(type='CLIPTextEncoder', ...),
            feat_channels=256,
        ),
    ),
)

# 可选：添加VLM文本生成器
vlm_text_generator=dict(
    type='VLMTextGenerator',
    vlm_cfg=dict(type='CoCa', ...),  # 可选，如果不配置则使用类别名称
    use_class_names=True,
)
```

### 3.3 视频交互分割

#### 实现思路
- 复用图像交互分割的逻辑
- 在视频模式下，对每一帧应用交互提示
- 通过 `TrackDataSample` 传递提示信息

#### 关键点
- 提示可以来自第一帧，也可以每帧独立
- 需要处理视频中的时序一致性

## 四、代码修改位置总结

### 4.1 任务判断逻辑

**位置**: `seg/models/detectors/mask2former_vid.py` → `predict` 方法（158行附近）

**判断顺序**:
1. 检查 `data_tag`（最直接）
2. 检查 `gt_instances_collected`（交互式分割）
3. 检查 `gt_instances.instances_ids`（VOS）
4. 检查 `num_frames`（视频 vs 图像）

### 4.2 VOS 实现

**位置**: `seg/models/detectors/mask2former_vid.py`
- 添加 `_vos_first_frame_match` 方法
- 在 `predict` 方法中调用

**复用逻辑**:
- `Mask2formerVideoMinVIS` 的 tube 分割（长视频）
- `match_from_embeds` 的 query 匹配

### 4.3 文本提示集成

**位置**: `seg/models/necks/sam_pe.py`
- 扩展 `SAMPromptEncoder.forward` 方法
- 集成 `PromptFusion` 模块

**位置**: `seg/models/heads/rapsam_head.py` 或 `mask2former_vid.py`
- 在 forward 中处理文本 embedding

### 4.4 交互式分割推理

**位置**: `seg/models/detectors/mask2former_vid.py` → `predict` 方法（130行附近）

**现有逻辑**:
- `inference_sam` 标志已存在
- `gt_instances_collected` 检测已实现
- `prepare_for_dn_mo` 已实现提示转换

**需要完善**:
- 支持文本提示
- 支持视频交互分割

## 五、数据集配置

### 5.1 当前配置

配置文件: `configs/_base_/datasets/coco_panoptic_video_yt19_yt21_davis_vip_city_sam_ref.py`

已包含的数据集:
- COCO Panoptic Video
- YouTube-VIS 2019/2021
- DAVIS
- VIPSeg
- Cityscapes
- SAM (点、框交互)
- RefCOCO (文本交互)

### 5.2 data_tag 映射

| data_tag | 任务类型 | 处理方式 |
|----------|---------|---------|
| `'coco'` | 全景分割 | `panoptic_on=True` |
| `'yt19'`, `'yt21'` | VIS | `Mask2formerVideoMinVIS` + tube分割 |
| `'davis'` | VOS | 第一帧匹配 + 跨帧跟踪 |
| `'sam'` | 交互式（点、框） | `gt_instances_collected` + `prepare_for_dn_mo` |
| `'refcoco'` | 交互式（文本） | 文本编码 + 提示融合 |
| `'vip'`, `'city'` | 视频全景分割 | `panoptic_on=True, num_stuff_classes>0` |

## 六、实现优先级

### 阶段1: 基础判断逻辑
1. ✅ 在 `mask2former_vid.py` 中添加任务类型判断
2. ✅ 通过 `data_tag` 区分不同任务

### 阶段2: VOS 支持
1. 实现第一帧 mask 匹配
2. 复用 tube 分割逻辑（长视频）
3. 添加 VOS 评估器

### 阶段3: 文本提示
1. ✅ 集成 `PromptFusion` 到 `SAMPromptEncoder`
2. ✅ 创建 `VLMTextGenerator` 模块
3. 完善 RefCOCO 数据流
4. 支持文本+点/框混合提示
5. 集成VLM文本生成到训练/推理流程

### 阶段4: 视频交互分割
1. 扩展交互分割到视频场景
2. 处理时序一致性

## 七、关键技术点

### 7.1 VOS vs VIS 的区别

| 特性 | VIS | VOS |
|------|-----|-----|
| 输入 | 自动检测所有实例 | 第一帧提供 mask |
| 输出 | 所有实例的跟踪 | 指定目标的跟踪 |
| Instance ID | 自动生成 | 从第一帧继承 |
| 匹配方式 | Query embedding 匹配 | 第一帧 IoU 匹配 + embedding 跟踪 |

### 7.2 视频全景分割 vs 图像全景分割

- **相同点**: 使用相同的 `panoptic_fusion_head`，相同的融合逻辑
- **不同点**: 
  - 视频：逐帧调用 `panoptic_fusion_head.predict()`
  - 图像：直接调用 `panoptic_fusion_head.predict()`
- **自动切换**: 通过 `num_frames > 0` 判断

### 7.3 交互式分割推理流程

1. **准备提示**: 用户提供点/框/文本 → 放入 `gt_instances_collected`
2. **自动检测**: Head 检测到 `gt_instances_collected` → 进入交互模式
3. **提示编码**: `prepare_for_dn_mo` → 转换为 query embeddings
4. **推理**: 使用这些 query 进行分割

## 八、可复用的代码模块

### 8.1 完全复用
- ✅ `ConcatOVDataset`: 数据集拼接和类别映射
- ✅ `preprocess_video_panoptic_gt`: 视频 GT 预处理（已支持 `gt_instance_ids`）
- ✅ `match_from_embeds`: Query 跨帧匹配
- ✅ `PromptFusion`: 多模态提示融合
- ✅ DAVIS 数据集加载器
- ✅ VOS 评估器 (`VOSMetric`)

### 8.2 部分复用
- ⚠️ `Mask2formerVideoMinVIS`: tube 分割逻辑可复用，但需要适配 VOS
- ⚠️ `prepare_for_dn_mo`: 提示转换逻辑可复用，需要扩展支持文本

### 8.3 需要新增
- ❌ 第一帧 mask 匹配逻辑（约 50-100 行代码）
- ✅ 文本提示支持（`SAMPromptEncoder` 已扩展）
- ✅ VLM文本生成器（`VLMTextGenerator` 已实现）
- ⚠️ 文本提示在推理时的处理流程（需要集成到head中）
- ❌ VOS 专用推理方法

## 九、配置文件建议

### 9.1 VOS 配置
```python
# configs/rap_sam/eval_rap_sam_vos.py
model.update(
    type=Mask2formerVideo,  # 或 Mask2formerVideoMinVIS（长视频）
    test_cfg=dict(
        panoptic_on=False,
        instance_on=True,
        vos_mode=True,  # 开启 VOS 模式
    ),
)
```

### 9.2 视频全景分割配置
```python
# configs/rap_sam/eval_rap_sam_video_panoptic.py
model.update(
    type=Mask2formerVideo,
    panoptic_head=dict(
        num_stuff_classes=58,  # VIPSeg 的 stuff 类别数
    ),
    test_cfg=dict(
        panoptic_on=True,
        instance_on=True,
    ),
)
```

### 9.3 交互式分割配置
```python
# 使用 inference_sam=True
model.update(
    inference_sam=True,  # 开启交互式推理
)
```

## 十、总结

### 10.1 核心思路
1. **统一框架**: 在 `Mask2formerVideo` 基础上，通过 `data_tag` 和数据结构特征判断任务类型
2. **复用优先**: 最大化复用现有代码（tube分割、query匹配、提示编码）
3. **最小改动**: 只在关键位置添加判断逻辑，不破坏现有功能

### 10.2 实现路径
1. **第一步**: 在 `mask2former_vid.py` 添加任务判断（VOS/VIS/交互/全景）
2. **第二步**: 实现 VOS 第一帧匹配逻辑
3. **第三步**: 完善文本提示支持
4. **第四步**: 测试和优化

### 10.3 关键文件
- `seg/models/detectors/mask2former_vid.py` - 主要修改位置
- `seg/models/detectors/mask2former_vid_minvis.py` - VIS 推理逻辑参考
- `seg/models/necks/sam_pe.py` - 提示编码（需扩展文本支持）
- `seg/datasets/concat_dataset.py` - 数据集拼接（已完善）

### 10.4 数据流
```
数据集 → ConcatOVDataset → data_tag 标识
  ↓
Detector.predict() → 根据 data_tag 判断任务类型
  ↓
├─ VOS → 第一帧匹配 → 跨帧跟踪
├─ VIS → tube分割 → query匹配
├─ 交互式 → prepare_for_dn_mo → 提示编码
└─ 全景 → panoptic_fusion_head → 融合输出
```

---

**最后更新**: 2024年
**状态**: 方案设计完成，待实现


