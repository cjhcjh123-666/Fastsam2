# 项目逻辑检查清单

## 一、数据流检查 ✅

### 1. 数据加载
- ✅ RefCOCO数据集加载文本到 `metainfo['text']`
- ✅ `PackDetInputs` 保留 `text` 在 `meta_keys` 中

### 2. 数据预处理 Pipeline
- ✅ `GeneratePoint` 创建 `gt_instances_collected` 并设置 `pb_labels` ✅ **已修复**
- ✅ `GenerateText` 从 `metainfo` 提取文本到 `gt_instances_collected.text`
- ✅ Pipeline顺序：`GeneratePoint` → `GenerateText`（正确）

### 3. 数据预处理器
- ✅ `SAMDataPreprocessor` 在推理时设置 `pb_labels`（训练时由 `GeneratePoint` 设置）

## 二、模型逻辑检查 ✅

### 1. Head配置
- ✅ `RapSAMVideoHead` 支持 `prompt_fusion` 参数 ✅ **已修复**
- ✅ `Mask2FormerVideoHead` 支持 `prompt_fusion` 参数

### 2. 训练逻辑
- ✅ `loss` 方法中 `prompt_training` 判断支持 `'refcoco'` ✅ **已修复**
- ✅ `prepare_for_dn_mo` 支持文本提示处理
- ✅ 文本融合逻辑正确

### 3. 推理逻辑
- ✅ `predict` 方法检查 `gt_instances_collected` 存在时设置 `prompt_training`

## 三、修复的问题

### 问题1: `pb_labels` 缺失 ✅ **已修复**
**错误**: `AttributeError: 'InstanceData' object has no attribute 'pb_labels'`
**原因**: `GeneratePoint` 创建 `gt_instances_collected` 时未设置 `pb_labels`
**修复**: 在 `GeneratePoint.transform` 中添加 `pb_labels` 设置

### 问题2: `prompt_training` 未正确设置 ✅ **已修复**
**原因**: `loss` 方法只检查 `data_tag == 'sam'`，未检查 `'refcoco'`
**修复**: 更新判断逻辑，支持 `['sam', 'refcoco']`

### 问题3: `RapSAMVideoHead` 不支持 `prompt_fusion` ✅ **已修复**
**原因**: `RapSAMVideoHead` 直接调用 `AnchorFreeHead.__init__`，未传递 `prompt_fusion`
**修复**: 在 `RapSAMVideoHead.__init__` 中手动初始化 `prompt_fusion`

## 四、数据流完整路径

```
RefCOCO数据集
  ↓
LoadAnnotations (text → metainfo['text'])
  ↓
PackDetInputs (保留text在metainfo)
  ↓
GeneratePoint (创建gt_instances_collected, 设置pb_labels) ✅
  ↓
GenerateText (text → gt_instances_collected.text)
  ↓
VideoPromptDataPreprocessor (数据预处理)
  ↓
Model.forward (data_tag='refcoco' → prompt_training=True) ✅
  ↓
Head.loss (prompt_training=True → 使用gt_instances_collected) ✅
  ↓
prepare_for_dn_mo (提取text, 使用PromptFusion融合) ✅
  ↓
训练/推理
```

## 五、关键配置检查

### 1. 数据集配置 ✅
- ✅ RefCOCO pipeline包含 `GenerateText`
- ✅ `data_tag` 正确设置为 `'refcoco'`

### 2. 模型配置
- ⚠️ 需要确保 `panoptic_head` 中有 `prompt_fusion` 配置
- ⚠️ 需要确保 `prompt_fusion` 中的 `text_encoder` 配置正确

### 3. 代码逻辑
- ✅ 所有必要的属性都已设置
- ✅ 所有判断逻辑都已更新

## 六、待测试项

1. ✅ `pb_labels` 是否正确设置
2. ✅ `prompt_training` 是否正确触发
3. ⚠️ 文本是否正确传递到 `prepare_for_dn_mo`
4. ⚠️ `PromptFusion` 是否正确处理文本
5. ⚠️ 训练是否能正常进行

## 七、潜在问题

### 1. 文本格式
- 确保 `gt_instances_collected.text` 是字符串列表
- 确保列表长度与 `point_coords` 数量匹配

### 2. PromptFusion配置
- 确保 `text_encoder` 配置正确
- 确保 `text_model_cfg` 中的checkpoint路径正确

### 3. 设备一致性
- 确保所有tensor在同一设备上
- 确保文本编码器在正确的设备上

---

**最后更新**: 2024年
**状态**: 主要问题已修复，待测试验证

