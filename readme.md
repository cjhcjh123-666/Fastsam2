## FastSAM2 多任务实时分割方案
# export HF_ENDPOINT=https://hf-mirror.com
# ps aux | grep rap_sam_fuxian| grep -v grep | awk '{print $2}' | xargs -r kill -9
### 1. 项目目标
- 构建统一的 FastSAM2 框架，覆盖图像/视频交互分割（点、框、文本）、视频对象分割（VOS）、全景分割。
- 在 8×RTX 3090 / 8×RTX 4090（24 GB）集群上实现实时推理（1080p 输入 ≥ 25 FPS），整体性能超越 SAM / SAM2。
- 完成 SAM2 → FastSAM2 的蒸馏训练，兼顾精度与效率。

### 2. 现有基础概述
| 模块 | 现状 | 需改进点 |
| --- | --- | --- |
| `seg/models/backbones/OpenCLIPBackbone` | CLIP 视觉分支，支持 ConvNeXt/ResNet；固定参数或全量训练 | 扩展双路径结构、引入 token pruning、支持蒸馏特征对齐 |
| `seg/models/necks/YOSONeck` | Lite deform FPN + 坐标编码 | 增加 mask 状态 refinement，兼容多任务输出 |
| `seg/models/heads/RapSAMVideoHead` | 多阶段 query 更新 + prompt/panoptic adaptor | 重构为多模态 TaskRouter、引入 Streaming Memory、低秩注意力 |
| `seg/models/detectors/RapSAM`/`Mask2formerVideo` | 统一 panoptic/instance 逻辑 | 支持多任务切换、VOS 专用流程、实时路径 |
| 数据 & 评估 | 支持 COCO、YouTube-VIS、交互评估器 | 融合 DAVIS、补齐 VOS 指标、扩展实时评测 |

### 3. 硬件与数据
- **硬件**：8×RTX 3090、8×RTX 4090；建议分集群训练，混集群时注意 NCCL 参数。
- **数据集**：
  - COCO（图像交互 + 全景）；
  - YouTube-VIS 2019（视频交互 + VOS）；
  - DAVIS 2017/2019（高质量 VOS）。
- **蒸馏教师**：SAM2 官方模型（需准备推理脚本和权重）。

### 4. 创新点设计

#### 4.1 改进骨干：双路径 Token Fusion + 动态裁剪
1. **结构**：
   - 在 `OpenCLIPBackbone` 新增几何轻量分支（MobileSAM 结构）：`geom_branch` 对高分辨率 feature map 做深度可分卷积提取边缘/轮廓。
   - 引入 `TokenFusionModule`：将 CLIP 语义 token 与几何 token 做 cross-attn 融合，输出语义增强特征。
2. **动态 token 策略**：
   - 在 `forward_func` 中添加 `TokenPruner`（基于注意力熵或显著性评分）：
     ```pseudo
     attn = compute_self_attention(x)
     keep_mask = topk(attn_entropy, ratio)
     x = x[keep_mask]
     ```
   - 视频模式下缓存前帧 token 分布，实现帧间共享。
3. **蒸馏接口**：
   - 新建 `backbones/distill_utils.py`，封装 SAM2 视觉 token 的投影对齐（L2 + Cosine）。
   - 训练脚本中提供 `--distill-weight` 控制损失权重。

#### 4.2 多任务轻量化结构：TaskRouter + Streaming Memory
1. **TaskRouter**（新增 `seg/models/utils/task_router.py`）：
   - 输入 prompt 类型（point/box/text/none）、视频长度、模式（interactive/vos/panoptic）。
   - 输出：解码器阶段数、激活 query 子集、是否启用 streaming memory。
2. **Streaming Memory Adapter**（新增 `seg/models/utils/memory_adapter.py`）：
   - 维护长期/短期 memory：`long_mem`（关键帧）、`short_mem`（最近帧）。
   - 提供接口 `update(frame_id, instance_embed, mask)` 与 `fetch(frame_id)`。
3. **RapSAMVideoHead 重构**：
   - 解耦 prompt adaptor 与 panoptic adaptor，按任务分别实例化。
   - 替换多头注意力为低秩近似（Nyström / Linformer）：
     ```python
     class LowRankMHSA(nn.Module):
         def forward(self, q, k, v):
             k_proj = self.k_proj(k)  # r << d
             v_proj = self.v_proj(v)
             attn = softmax(q @ k_proj.T / sqrt(d))
             return attn @ v_proj
     ```
   - 新增 `MaskStateRefiner`（小卷积网络）接在 `YOSONeck` 输出之后，快速修补边界，减少 transformer 负担。
4. **VOS 适配**：
   - 结合 memory adapter，实现 mask propagation；
   - 引入 `Dual-Path Self-Refinement (DPSR)`：上一帧 mask 与当前预测互监督（MSE+Dice），增强时序一致性。

#### 4.3 多模态交互增强：Cross-Prompt Fusion + 语言引导记忆
1. **Cross-Prompt Fusion**：
   - `SAMPromptEncoder` 输出点/框/掩码 embedding；
   - `OpenCLIPBackbone.get_text_model()` 输出文本 embedding；
   - 新增模块 `PromptFusion`（`seg/models/utils/prompt_fusion.py`）：
     ```python
     concat = torch.cat([point_embed, box_embed, text_embed], dim=1)
     fused = transformer_layer(concat)
     ```
   - 训练时加入文本-视觉多任务损失（对齐文本 prompt 与 mask 类别）。
2. **语言引导记忆**：
   - 在 memory adapter 中记录文本语义向量；
   - VOS 更新时通过文本相似度筛选目标，避免 drift。
3. **DPSR**：
   - 定义损失 `L_dpsr = λ1 * Dice(prev_mask, curr_pred) + λ2 * MSE(prev_feat, curr_feat)`。
   - 在 `RapSAMVideoHead.loss` 中加入额外项。

### 5. 训练与蒸馏流程

#### 5.1 数据预处理
- **COCO**：生成交互提示（随机选点/框 + 文本描述），保存在 `data/annotations/fastsam2_interactive.json`。
- **YouTube-VIS 2019**：
  - 生成视频 clip（长度 8~16），存储切片信息。
  - 提取关键帧并标注交互提示，另存 `ytvis_interactive_meta.pkl`。
- **DAVIS**：
  - 转换为统一格式（`TrackDataSample`），包含高质量 mask 序列。

#### 5.2 在线蒸馏框架
- 训练阶段常驻加载 SAM2 教师模型（`sam2_teacher` 模块），与学生网络共享 batch 图像：
  - 在训练循环中调用 `teacher.forward(images, prompts)` 取得视觉 token / mask 预测；
  - 支持 fp16/bf16 推理、DeviceList 指定教师所用 GPU。
- 蒸馏损失设计：
  - 视觉特征：`L_visual = α * ||f_student - W * f_teacher||_2 + β * (1 - cos)`；
  - Mask logit：`L_mask = γ * BCE(student_mask, teacher_mask)`；
  - Prompt 对齐：`L_prompt = δ * KL(student_prompt || teacher_prompt)`。
  - 推荐权重：α=1.0, β=0.5, γ=1.0, δ=0.2（可按训练效果调节）。
- Prompt 生成在线完成（点/框/文本），必要时在 dataloader 或 Hook 中缓存，以减少重复解析。

#### 5.3 训练阶段划分
| 阶段 | 模型组件 | 数据 | 主要目标 | 设备建议 |
| --- | --- | --- | --- | --- |
| Stage 1 | 改进骨干 + 蒸馏头 | COCO + YouTube-VIS | 学习双路径特征，完成蒸馏 | 8×4090 |
| Stage 2 | 重构 RapSAM head + TaskRouter | YouTube-VIS（交互）+ COCO | 实现实时交互能力 | 8×3090 |
| Stage 3 | VOS 模块（memory+DPSR） | YouTube-VIS + DAVIS | 稳定时序 + 高精度 VOS | 8×4090 |
| Stage 4 | 多任务联合微调 | 三数据混合 | 统一多任务表现 | 8×4090 |
| Stage 5 | 推理/部署优化 | Demo + Benchmark | 验证 FPS & 精度 | 8×3090 |

#### 5.4 训练脚本框架
- 新增 `tools/train_fast_sam2.py`：
  - 支持 `--stage`、`--teacher-config`、`--task-mode`；
  - 在训练循环调用 `sam2_teacher` 模块，实现在线蒸馏。
- 配置文件结构：
  - `configs/fastsam2/` 下新增：
    - `stage1_backbone_distill.py`
    - `stage2_interactive.py`
    - `stage3_vos.py`
    - `stage4_joint.py`
    - `runtime_default.py`（通用 runtime）。
- 各 stage 配置要对齐数据加载、模型组件开关（如 `use_task_router=True`）。

### 6. 推理与评测

#### 6.1 Demo/工具链
- 扩展 `demo/demo.py`：
  - 增加 `--task {interactive_image, interactive_video, vos, panoptic}`；
  - 视频模式下支持实时交互（监听点/框输入）、文本提示。
- 新增 `tools/benchmark/benchmark_realtime.py`：
  - 统计不同输入尺寸的 FPS + GPU 利用率；
  - 提供 log 输出，便于回归。

#### 6.2 评估指标
- `InteractiveEvaluator`：扩展支持多轮交互统计（NOC@0.5/0.8/0.9，mIoU@iter）。
- VOS 指标：集成 DAVIS 官方 `J&F` 评估脚本（保存结果后统一评估）。
- 全景分割：沿用 COCO Panoptic AP。

### 7. 实现细节清单

1. **代码目录调整**
   - `seg/models/backbones/openclip_backbone.py`：新增几何分支、token pruner、蒸馏 hook。
   - `seg/models/backbones/modules/`（新建目录）：放置 `token_pruner.py`、`fusion.py`。
   - `seg/models/utils/`：新增 `task_router.py`、`memory_adapter.py`、`prompt_fusion.py`、`distill_utils.py`。
   - `seg/models/heads/rapsam_head.py`：重构 head，拆分 adaptor，引入低秩注意力/记忆接口。
   - `seg/models/necks/ramsam_neck.py`：新增 `MaskStateRefiner` 模块及调用。
   - `seg/models/detectors/rapsam.py`：增加 `MODE` 参数（interactive/vos/panoptic），根据模式调用 TaskRouter。
   - `seg/models/data_preprocessor/vid_sam_preprocessor.py`：补充 VOS 数据处理（mask propagate、memory 占位符）。

2. **配置/脚本**
   - `configs/fastsam2/*`：分 stage 配置，定义模型组件开关、蒸馏权重、训练时长。
   - `tools/train_fast_sam2.py`、`tools/benchmark/benchmark_realtime.py`。

3. **训练策略**
   - 优化器：AdamW（lr=1e-4），backbone 分段学习率（几何分支 > CLIP）。
   - 学习率策略：Cosine decay + warmup（5 % epoch）。
   - 交互训练增加 prompt 随机性（点/框数量、文本描述多样）。
   - VOS 训练使用 curriculum：先短序列（4-6 帧）再长序列（12-16 帧）。

4. **性能优化**
   - 混合精度训练（AMP）。
   - 推理阶段支持 TensorRT/ONNX（后续拓展）。
   - 视频流采用渐进式解码（处理过的帧缓存，减少重复计算）。

### 8. 时间与里程碑（建议）
| 时间（周） | 任务 | 里程碑 |
| --- | --- | --- |
| 第 1-2 周 | 搭建 Stage1 蒸馏训练，完成骨干改造 | 获得蒸馏后骨干权重，FPS 初步验证 |
| 第 3-4 周 | 实现 TaskRouter + Streaming Memory，训练 Stage2 | 视频交互达到目标 FPS，交互指标优于 baseline |
| 第 5-6 周 | 集成 VOS（memory + DPSR），训练 Stage3 | DAVIS J&F 提升，时序稳定 |
| 第 7 周 | 多任务联合微调，统一推理 Demo | 单模型覆盖所有任务，demo 支持实时交互 |
| 第 8 周 | 完成基准评测与文档 | 输出完整 benchmark 与使用文档 |

### 9. 风险与对策
- **蒸馏不收敛**：调整蒸馏权重/特征对齐方式；尝试多层监督（骨干中间层）。
- **实时性能不足**：进一步调低 token 保留率、减少 decoder stage、启用 TensorRT。
- **交互/VOS 数据不足**：利用 SAM2 生成伪标签补充；对 YouTube-VIS/DAVIS 做数据增强。
- **多任务冲突**：采用任务权重动态调整（GradNorm），或阶段性交替训练。

### 10. 后续工作
- 完成上述代码与训练后，更新 README、撰写技术报告，并准备公开发布/论文撰写。
- 评估与部署：考虑将实时推理封装为服务（RESTful/WebSocket），支持在线交互。

---
本方案覆盖了 FastSAM2 的总体架构升级、模块改造、训练/蒸馏流程、评测部署等细节，可据此逐步实施。若后续需求有变，可在对应阶段及时调整。

