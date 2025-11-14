# FastSAM2 / RapSAM

一个统一的多任务实时分割框架，支持图像/视频交互分割、视频对象分割（VOS）和全景分割。

## 📋 目录

- [特性](#特性)
- [支持的模型和任务](#支持的模型和任务)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [训练](#训练)
- [推理](#推理)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [引用](#引用)

## ✨ 特性

### 核心功能

- **多任务统一架构**：单模型支持多种分割任务
  - 图像交互分割（点、框、文本提示）
  - 视频交互分割（点、框、文本提示）
  - 视频对象分割（VOS）
  - 全景分割

- **智能任务路由**：自动检测任务类型并路由到相应处理路径
- **流式记忆管理**：VOS 任务中的长期/短期记忆机制
- **多模态提示融合**：融合点、框、文本等多种提示类型
- **实时推理**：针对 1080p 输入优化，目标 ≥ 25 FPS

### 技术亮点

- **TaskRouter**：自动任务类型检测和动态路由
- **StreamingMemoryAdapter**：VOS 记忆管理，支持自适应更新策略
- **PromptFusion**：多模态提示融合，支持文本-视觉对齐
- **Dual-Path Self-Refinement (DPSR)**：时序一致性增强

## 🎯 支持的模型和任务

### 模型架构

- **RapSAM**：多任务分割检测器
  - Backbone: ResNet / OpenCLIP (ConvNeXt/ResNet)
  - Neck: YOSONeck (Lite Deform FPN)
  - Head: RapSAMVideoHead (多阶段 query 更新)

### 支持的任务

| 任务类型 | 输入 | 输出 | 数据集 |
|---------|------|------|--------|
| 图像交互分割 | 图像 + 点/框/文本 | Mask | COCO, RefCOCO, SAM |
| 视频交互分割 | 视频 + 点/框/文本 | Mask 序列 | YouTube-VIS 2019/2021 |
| 视频对象分割 | 视频 + 第一帧标注 | Mask 序列 | DAVIS 2017, VIPSeg |
| 全景分割 | 图像/视频 | Panoptic Mask | COCO, Cityscapes |

## 🚀 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0
- mmdetection >= 3.0.0
- mmengine >= 0.8.0

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/cjhcjh123-666/Fastsam2.git
cd Fastsam2-main
```

2. **创建 conda 环境**
```bash
conda create -n rap_sam_fuxian python=3.8
conda activate rap_sam_fuxian
```

3. **安装依赖**
```bash
# 安装 PyTorch (根据您的 CUDA 版本)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装 mmdetection 和相关依赖
pip install mmdet mmengine mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# 安装其他依赖
pip install -r requirements.txt  # 如果有 requirements.txt
```

4. **安装项目**
```bash
pip install -e .
```

## 📦 数据集准备

### 支持的数据集

项目支持以下数据集：

- **COCO**：图像全景分割和交互分割
- **YouTube-VIS 2019/2021**：视频实例分割
- **DAVIS 2017**：视频对象分割
- **VIPSeg**：视频全景分割
- **Cityscapes**：城市街景全景分割
- **RefCOCO**：引用表达分割
- **SAM**：类别无关分割

### 数据集目录结构

```
data/
├── coco/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│       ├── panoptic_train2017.json
│       ├── panoptic_val2017.json
│       └── panoptic_train2017/
├── youtube_vis_2019/
│   ├── train/
│   └── valid/
├── davis/
│   └── DAVIS/
│       ├── Annotations/
│       └── ImageSets/
├── ref_seg/
│   └── refcoco/
└── ...
```

### 数据集配置

数据集配置位于 `configs/_base_/datasets/` 目录下。主要配置文件：

- `coco_panoptic_video_yt19_yt21_davis_vip_city_sam_ref.py`：多数据集混合训练配置

## 🏃 快速开始

### 1. 检查点准备

下载预训练权重到 `checkpoints/` 目录：

```bash
mkdir -p checkpoints

# ResNet-50 预训练权重
# 下载 resnet50-0676ba61.pth 到 checkpoints/

# OpenCLIP 预训练权重
# 下载 openclip_vitl14_pretrain.pt 到 checkpoints/
```

### 2. 配置文件

主要配置文件：`configs/rap_sam/rap_sam_r50_12e_adaptor.py`

### 3. 单卡训练

```bash
conda activate rap_sam_fuxian
cd /mnt/chenjiahui/Fastsam2-main
python tools/train.py configs/rap_sam/rap_sam_r50_12e_adaptor.py \
    --work-dir work_dirs/rap_sam_r50_12e
```

### 4. 多卡训练（推荐）

```bash
# 8 卡训练
bash tools/dist_train.sh configs/rap_sam/rap_sam_r50_12e_adaptor.py 8 \
    --work-dir work_dirs/rap_sam_r50_12e
```

## 🎓 训练

### 训练配置

训练配置在 `configs/rap_sam/rap_sam_r50_12e_adaptor.py` 中定义，包括：

- **模型配置**：backbone、neck、head 设置
- **多任务组件**：TaskRouter、StreamingMemory、PromptFusion
- **数据配置**：数据集路径、数据增强
- **训练策略**：学习率、优化器、损失函数

### 关键配置说明

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

# 任务特定Loss权重配置（核心创新）
# 当batch属于某个任务时，只有对应的loss生效，其他任务的loss权重为0
# 这样可以避免不同任务之间的loss冲突，同时保证所有模块都参与梯度计算
task_loss_weights = dict(
    # 图像交互分割（点、框、文本提示）
    interactive_image=dict(
        loss_cls=1.0, loss_mask=5.0, loss_dice=5.0, loss_iou=10.0,
        loss_prompt_align=0.5, loss_text_visual=0.3,
        loss_dpsr=0.0, loss_temporal=0.0, loss_panoptic=0.0,  # 屏蔽其他任务
    ),
    # 视频交互分割
    interactive_video=dict(
        loss_cls=1.0, loss_mask=5.0, loss_dice=5.0, loss_iou=10.0,
        loss_prompt_align=0.5, loss_text_visual=0.3, loss_temporal=1.0,
        loss_dpsr=0.0, loss_panoptic=0.0,
    ),
    # VOS (视频对象分割)
    vos=dict(
        loss_cls=1.0, loss_mask=5.0, loss_dice=5.0, loss_iou=0.0,
        loss_dpsr=2.0, loss_temporal=1.5, loss_memory_align=1.0,
        loss_prompt_align=0.0, loss_text_visual=0.0, loss_panoptic=0.0,
    ),
    # 全景分割
    panoptic=dict(
        loss_cls=2.0, loss_mask=5.0, loss_dice=5.0, loss_iou=0.0,
        loss_panoptic=1.0,
        loss_prompt_align=0.0, loss_text_visual=0.0, loss_dpsr=0.0,
        loss_temporal=0.0, loss_memory_align=0.0,
    ),
)

# DDP 配置（多任务训练必须）
find_unused_parameters = True  # 关键：防止NCCL Timeout，混合数据集训练必需
```

### 训练选项

```bash
# 启用混合精度训练
python tools/train.py configs/rap_sam/rap_sam_r50_12e_adaptor.py --amp

# 自动缩放学习率
python tools/train.py configs/rap_sam/rap_sam_r50_12e_adaptor.py --auto-scale-lr

# 从检查点恢复
python tools/train.py configs/rap_sam/rap_sam_r50_12e_adaptor.py --resume work_dirs/rap_sam_r50_12e/latest.pth
```

### 训练阶段

根据 `FASTSAM2_IMPLEMENTATION_PLAN.md`，训练分为多个阶段：

1. **Stage 1**：骨干网络 + 蒸馏训练
2. **Stage 2**：交互分割能力
3. **Stage 3**：VOS 模块（memory + DPSR）
4. **Stage 4**：多任务联合微调
5. **Stage 5**：推理优化

## 🔍 推理

### 使用 Demo

```bash
python demo/demo.py \
    --config configs/rap_sam/rap_sam_r50_12e_adaptor.py \
    --checkpoint work_dirs/rap_sam_r50_12e/latest.pth \
    --input demo/demo.jpg \
    --task interactive_image \
    --output demo/output.jpg
```

### 任务类型

- `interactive_image`：图像交互分割
- `interactive_video`：视频交互分割
- `vos`：视频对象分割
- `panoptic`：全景分割

### 评估

```bash
# 评估 COCO 全景分割
python tools/test.py configs/rap_sam/eval_rap_sam_coco.py \
    work_dirs/rap_sam_r50_12e/latest.pth

# 评估 YouTube-VIS
python tools/test.py configs/rap_sam/eval_rap_sam_yt19.py \
    work_dirs/rap_sam_r50_12e/latest.pth

# 评估交互分割
python tools/test.py configs/rap_sam/eval_rap_sam_prompt.py \
    work_dirs/rap_sam_r50_12e/latest.pth
```

## 📁 项目结构

```
Fastsam2-main/
├── configs/                 # 配置文件
│   ├── _base_/              # 基础配置
│   │   ├── datasets/        # 数据集配置
│   │   └── schedules/      # 训练策略
│   └── rap_sam/            # RapSAM 模型配置
├── seg/                    # 分割模块
│   ├── models/             # 模型定义
│   │   ├── backbones/      # 骨干网络
│   │   ├── necks/          # 颈部网络
│   │   ├── heads/          # 检测头
│   │   ├── detectors/      # 检测器
│   │   ├── utils/          # 工具模块
│   │   │   ├── task_router.py        # 任务路由
│   │   │   ├── memory_adapter.py     # 流式记忆
│   │   │   └── prompt_fusion.py      # 提示融合
│   │   └── data_preprocessor/        # 数据预处理
│   ├── datasets/           # 数据集
│   └── evaluation/         # 评估指标
├── ext/                    # 外部库
│   ├── sam/                # SAM 相关模块
│   ├── open_clip/          # OpenCLIP
│   └── davis2017/          # DAVIS 评估
├── tools/                  # 工具脚本
│   ├── train.py           # 训练脚本
│   ├── test.py            # 测试脚本
│   └── dist_train.sh      # 分布式训练
├── demo/                   # Demo 示例
├── checkpoints/            # 预训练权重
└── work_dirs/             # 训练输出
```

## ⚙️ 配置说明

### 多任务组件

#### TaskRouter

自动检测任务类型并配置相应的处理路径：

```python
task_router = dict(
    type='TaskRouter',
    feat_channels=256,
    num_decoder_stages=3,
    enable_streaming_memory=True,
    interactive_stages=3,    # 交互任务 decoder stages
    vos_stages=3,            # VOS 任务 decoder stages
    panoptic_stages=3        # 全景任务 decoder stages
)
```

#### StreamingMemoryAdapter

VOS 任务的记忆管理：

```python
streaming_memory = dict(
    type='StreamingMemoryAdapter',
    feat_channels=256,
    long_mem_size=10,        # 长期记忆大小
    short_mem_size=5,        # 短期记忆大小
    update_strategy='adaptive'  # 更新策略：FIFO/Quality/Adaptive
)
```

#### PromptFusion

多模态提示融合：

```python
prompt_fusion = dict(
    type='PromptFusion',
    feat_channels=256,
    num_heads=8,
    dropout=0.1,
    use_text_encoder=True,
    text_encoder=dict(
        type='TextEncoder',
        feat_channels=256,
        text_model_cfg=dict(
            type=OpenCLIPBackboneText,
            model_name='ViT-L-14',
            init_cfg=dict(
                type='clip_pretrain',
                checkpoint='checkpoints/openclip_vitl14_pretrain.pt'
            )
        )
    )
)
```

### 模型配置

主要模型配置在 `configs/rap_sam/rap_sam_r50_12e_adaptor.py`：

- **Backbone**：ResNet-50 或 OpenCLIP
- **Neck**：YOSONeck (Lite Deform FPN)
- **Head**：RapSAMVideoHead
- **损失函数**：分类损失、Mask 损失、Dice 损失
### 数据处理流程
数据流处理：
┌─────────────────────────────────────────────┐
│  DataLoader 加载数据                         │
└──────────────┬──────────────────────────────┘
               │
               ▼
     ┌────────────────────┐
     │  判断数据类型       │
     └────────┬───────────┘
              │
       ┌──────┴──────┐
       │             │
   视频数据      图像数据
(TrackDataSample) (DetDataSample)
       │             │
       ▼             ▼
   reshape        直接处理
       │             │
       └──────┬──────┘
              ▼
        特征提取 (同一backbone)
              │
              ▼
        panoptic_head.loss()
              │
              ▼
      TaskRouter检测任务类型
              │
              ▼
     计算所有可能的loss
              │
              ▼
    根据任务类型应用loss权重
    （屏蔽不相关的loss）
              │
              ▼
          返回masked losses

## ❓ 常见问题

### 1. 设备不匹配错误

**问题**：`RuntimeError: Expected all tensors to be on the same device`

**解决**：确保所有模型参数正确注册为 buffer 或 parameter。已修复 SAMPromptEncoder 的设备问题。

### 2. SyncBatchNorm 错误

**问题**：单卡训练时 SyncBatchNorm 报错

**解决**：单卡训练时使用普通 BN，多卡训练时使用 SyncBN。配置中已设置 `norm_cfg=dict(type='BN', requires_grad=True)`。

### 3. DDP 训练错误 / NCCL Timeout

**问题**：
- `find_unused_parameters` 相关错误
- `NCCL Timeout` 错误

**原因**：多任务模型中存在条件性使用的模块（如 TextEncoder、StreamingMemory），这些模块在某些batch中不参与前向传播，导致DDP同步失败。

**解决**：
1. **必须设置** `find_unused_parameters = True`（已在配置中设置）
2. **任务特定Loss Masking**：通过 `task_loss_weights` 配置，确保：
   - 所有loss都被计算（保证梯度流）
   - 根据任务类型自动屏蔽不相关的loss（权重设为0）
   - 避免不同任务的loss相互干扰

**示例**：
```python
# 当batch是图像交互分割任务时
task_loss_weights['interactive_image'] = {
    'loss_cls': 1.0,      # 激活
    'loss_iou': 10.0,     # 激活
    'loss_dpsr': 0.0,     # 屏蔽（VOS任务的loss）
    'loss_temporal': 0.0, # 屏蔽（视频任务的loss）
}
```

### 4. 内存不足

**问题**：训练时显存不足

**解决**：
- 减小 batch size
- 减少 decoder stages
- 使用梯度累积
- 启用混合精度训练 (`--amp`)

### 5. 数据集加载错误

**问题**：数据集路径或格式错误

**解决**：
- 检查数据集路径配置
- 确认数据集格式符合要求
- 查看日志中的具体错误信息

## 📊 性能指标

### 训练配置

- **硬件**：8×RTX 3090 / 8×RTX 4090 (24GB)
- **Batch Size**：根据数据集和 GPU 数量调整
- **学习率**：1e-4 (AdamW)
- **训练轮数**：12 epochs

### 目标性能

- **推理速度**：1080p 输入 ≥ 25 FPS
- **精度**：超越 SAM / SAM2 baseline

## 🔧 开发与贡献

### 代码规范

- 遵循 MMDetection 代码规范
- 使用类型注解
- 添加必要的文档字符串

### 调试建议

1. **检查任务检测**：在训练日志中查看任务类型是否正确识别
2. **验证组件状态**：确认 TaskRouter、StreamingMemory、PromptFusion 已正确初始化
3. **检查数据格式**：验证输入数据包含必要的字段（`gt_instances_collected`、`text` 等）

## 📚 相关文档

- `PROJECT_DIAGNOSIS_REPORT.md`：项目诊断报告
- `FASTSAM2_IMPLEMENTATION_PLAN.md`：实现计划
- `MULTI_TASK_REFACTORING_SUMMARY.md`：多任务重构总结
- `MULTI_TASK_ARCHITECTURE.md`：多任务架构说明

## 📝 更新日志

### 最新更新

- ✅ **修复 NCCL Timeout 问题**：设置 `find_unused_parameters = True`，解决多任务训练中的分布式同步问题
- ✅ **实现多任务 Loss Masking 机制**：引入 `task_loss_weights` 配置，根据任务类型自动激活/屏蔽不同loss，避免任务间干扰
- ✅ **优化任务路由机制**：TaskRouter 自动检测任务类型并应用相应的loss权重
- ✅ 修复 SAMPromptEncoder 设备不匹配问题
- ✅ 完善 StreamingMemory 的实际应用
- ✅ 优化 DPSR 损失计算
- ✅ 完善文本编码器集成
- ✅ 支持多数据集混合训练

## 📄 许可证

本项目遵循相应的开源许可证。请查看 LICENSE 文件了解详情。

## 🙏 致谢

- [MMDetection](https://github.com/open-mmlab/mmdetection)：检测框架
- [SAM](https://github.com/facebookresearch/segment-anything)：分割模型
- [OpenCLIP](https://github.com/mlfoundations/open_clip)：CLIP 实现

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues：[提交 Issue](https://github.com/cjhcjh123-666/Fastsam2/issues)

---

**注意**：本项目仍在积极开发中，API 可能会有变化。建议查看最新文档和代码。

