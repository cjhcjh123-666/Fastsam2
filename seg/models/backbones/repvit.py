import torch.nn as nn
import numpy as np
import itertools

from mmdet.registry import MODELS as BACKBONES
import logging
from mmengine.runner import load_checkpoint

from torch.nn.modules.batchnorm import _BatchNorm

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from timm.models.layers import SqueezeExcite

import torch
import math
from typing import Dict, Optional, Callable

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class RepViT(nn.Module):
    def __init__(self, cfgs, distillation=False, pretrained=None, init_cfg=None, out_indices=[], compression_config=None, **kwargs):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.compression_config = compression_config if isinstance(compression_config, dict) else None
        # 兼容来自配置的多余参数（如 frozen_stages/norm_cfg/norm_eval 等）
        self.extra_cfg = kwargs or {}

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)

        # --- Progressive/Auto-drop instrumentation ---
        self._impact_scores: Dict[int, float] = {}
        self._impact_momentum: float = 0.9
        self._impact_enabled: bool = False
        self._last_dropped: Optional[int] = None

        # --- LoRA instrumentation ---
        self._lora_enabled: bool = False
        
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()

        # Apply static compression if provided (e.g., drop specific layers)
        if self.compression_config is not None:
            drop_layers = self.compression_config.get('drop_layers', [])
            for li in sorted(set(drop_layers)):
                try:
                    self.progressive_drop_layer(li)
                except Exception:
                    pass

        # 若配置要求，启用块影响监控（用于自动选块）
        if isinstance(self.compression_config, dict):
            if (self.compression_config.get('enable_impact_monitor', False) or 
                self.compression_config.get('enable_auto_selection', False)):
                momentum = float(self.compression_config.get('impact_momentum', 0.9))
                self.enable_impact_monitor(momentum=momentum)
                print(f"✅ RepViT 影响监控已启用，动量={momentum}")

    # 在 RepViT 类的 init_weights 方法中找到以下代码段：
    def init_weights(self, pretrained=None):
        logger = logging.getLogger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                f'specify `Pretrained` in ' \
                                                f'[init_cfg](file:///data/zhangyafei/RMP-SAM_repvit/seg/models/backbones/repvit.py#L0-L0) in ' \
                                                f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            # 手动加载权重，正确处理包含'model'键的checkpoint格式
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            # 提取正确的state_dict
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
                
            # 加载权重到模型，使用strict=False忽略不匹配的键
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            # 报告加载结果
            if len(missing_keys) > 0:
                logger.info(f"Missing keys: {missing_keys[:10]}...")  # 只显示前10个
            if len(unexpected_keys) > 0:
                logger.info(f"Unexpected keys: {unexpected_keys[:10]}...")  # 只显示前10个
                
            loaded_keys = len([k for k in state_dict.keys() if k not in missing_keys])
            total_keys = len(list(self.state_dict().keys()))
            logger.info(f"Successfully loaded {loaded_keys}/{total_keys} parameters from {ckpt_path}")

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(RepViT, self).train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


    def forward(self, x):
        outs = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.out_indices:
                outs.append(x)
                # print(f"Layer {i} output shape: {x.shape}")
        assert(len(outs) == 4)
        return outs

    @torch.no_grad()
    def progressive_drop_layer(self, layer_index: int):
        """Drop a specific layer by replacing it with Identity.

        layer_index follows self.features index (0 is patch_embed).
        """
        if not isinstance(layer_index, int):
            return
        if layer_index < 0 or layer_index >= len(self.features):
            return
        # Do not drop the initial patch embedding aggressively; but allow if explicitly requested
        try:
            self.features[layer_index] = nn.Identity()
            self._last_dropped = int(layer_index)
        except Exception:
            pass

    def get_compression_status(self):
        """Return a lightweight status for compression hooks/monitors."""
        total_layers = len(self.features)
        dropped = 0
        for m in self.features:
            if isinstance(m, nn.Identity):
                dropped += 1
        return {
            'total_layers': total_layers,
            'dropped_layers': dropped,
        }

    # -------- Auto block selection (impact monitor) ---------
    def enable_impact_monitor(self, momentum: float = 0.9):
        """在每个块上注册前后钩子，记录输入/输出差异的EMA，作为块重要性（越小越不重要）。"""
        if self._impact_enabled:
            return
        self._impact_enabled = True
        self._impact_momentum = float(momentum)

        def _make_hook(idx: int) -> Callable:
            def hook(module, inputs, output):
                try:
                    x_in = inputs[0]
                    x_out = output
                    if not torch.is_tensor(x_in) or not torch.is_tensor(x_out):
                        return
                    # 对齐空间尺寸
                    if x_in.shape != x_out.shape:
                        return
                    diff = (x_out - x_in).abs().mean().detach()
                    prev = self._impact_scores.get(idx, None)
                    m = self._impact_momentum
                    val = diff.item()
                    if prev is None:
                        self._impact_scores[idx] = val
                    else:
                        self._impact_scores[idx] = m * prev + (1.0 - m) * val
                except Exception:
                    pass
            return hook

        # 注册在除去 patch_embed (index 0) 的块输出处
        for idx, module in enumerate(self.features):
            if isinstance(module, nn.Identity):
                continue
            if idx == 0:
                continue
            try:
                module.register_forward_hook(_make_hook(idx))
            except Exception:
                pass

    def get_block_impact_scores(self) -> Dict[int, float]:
        return {k: v for k, v in self._impact_scores.items() if not isinstance(self.features[k], nn.Identity)}

    def auto_select_block_to_drop(self, exclude: Optional[set] = None) -> Optional[int]:
        """选择影响分数最小的块索引（排除列表与关键层）。"""
        exclude = set(exclude or set())
        # 不丢弃 patch_embed 与 out_indices 所在层（尽量避免特征输出位）
        exclude.add(0)
        for oi in (self.out_indices or []):
            exclude.add(int(oi))
        candidates = {k: v for k, v in self.get_block_impact_scores().items() if k not in exclude}
        if not candidates:
            return None
        sel = min(candidates.items(), key=lambda kv: kv[1])[0]
        return int(sel)

    # ----------------- LoRA for 1x1 conv --------------------
    class LoRAConv2d(nn.Module):
        def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: int = 16):
            super().__init__()
            assert base.kernel_size == (1, 1) and base.groups == 1, 'LoRAConv2d 仅支持1x1、groups=1'
            self.base = base
            in_ch = base.in_channels
            out_ch = base.out_channels
            self.rank = int(rank)
            self.alpha = float(alpha)
            # 低秩分解：A: in->r, B: r->out
            self.lora_A = nn.Conv2d(in_ch, self.rank, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(self.rank, out_ch, kernel_size=1, bias=False)
            # 初始化为零增量
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = self.alpha / max(self.rank, 1)
            self.merged = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.merged or (not self.training and not self.lora_A.weight.requires_grad):
                return self.base(x)
            return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

        @torch.no_grad()
        def merge(self):
            if self.merged:
                return
            delta_w = (self.lora_B.weight @ self.lora_A.weight.view(self.rank, -1)).view_as(self.base.weight) * self.scaling
            self.base.weight += delta_w
            self.merged = True

    def enable_lora(self, rank: int = 8, alpha: int = 16):
        """为 1x1 pointwise 卷积注入 LoRA，针对丢块后的情况进行优化"""
        if self._lora_enabled:
            return
        self._lora_enabled = True
        
        # 获取被丢弃的层
        dropped_layers = set()
        if self.compression_config and 'drop_layers' in self.compression_config:
            dropped_layers = set(self.compression_config['drop_layers'])
        
        lora_count = 0
        enhanced_count = 0
        
        for layer_idx, layer_module in enumerate(self.features):
            # 跳过被丢弃的层
            if layer_idx in dropped_layers:
                continue
                
            # 判断是否为需要增强的关键层
            is_critical = self._is_critical_for_compression(layer_idx, dropped_layers)
            current_rank = rank * 2 if is_critical else rank
            current_alpha = alpha * 2 if is_critical else alpha
            
            # 为该层的Conv2d_BN模块注入LoRA
            for m in layer_module.modules():
                if isinstance(m, Conv2d_BN):
                    c: nn.Module = m.c
                    if isinstance(c, nn.Conv2d) and c.kernel_size == (1, 1) and c.groups == 1:
                        try:
                            lora = RepViT.LoRAConv2d(c, rank=current_rank, alpha=current_alpha)
                            m.c = lora
                            lora_count += 1
                            if is_critical:
                                enhanced_count += 1
                        except Exception:
                            pass
        
        print(f"✅ LoRA注入完成: {lora_count}个1x1卷积")
        print(f"   其中{enhanced_count}个关键层使用增强LoRA (rank={rank*2})")
        if dropped_layers:
            print(f"   被丢弃层: {sorted(dropped_layers)}")
    
    def _is_critical_for_compression(self, layer_idx: int, dropped_layers: set) -> bool:
        """判断某层是否因压缩而变为关键层，需要增强LoRA"""
        if not dropped_layers:
            return False
            
        # 相邻层被丢弃的层需要增强
        for dropped_idx in dropped_layers:
            if abs(layer_idx - dropped_idx) <= 2:  # 2层范围内
                return True
        
        # 在256通道段且有层被丢弃的情况下，所有256通道层都需要增强
        if 9 <= layer_idx <= 21:  # 256通道段
            return bool(dropped_layers.intersection(range(9, 22)))
        
        return False

    @torch.no_grad()
    def merge_lora(self):
        if not self._lora_enabled:
            return
        for m in self.modules():
            if isinstance(m, RepViT.LoRAConv2d):
                try:
                    m.merge()
                except Exception:
                    pass

from timm.models import register_model


@BACKBONES.register_module()
def repvit_m0_9(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]

    return RepViT(cfgs, init_cfg=init_cfg, pretrained=pretrained, distillation=distillation, out_indices=out_indices, **kwargs)



@BACKBONES.register_module()
def repvit_m1_0(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  56, 1, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  112, 0, 0, 2],
        [3,   2,  112, 1, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  224, 0, 1, 2],
        [3,   2,  224, 1, 1, 1],
        [3,   2,  224, 0, 1, 1],
        [3,   2,  224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 448, 0, 1, 2],
        [3,   2, 448, 1, 1, 1],
        [3,   2, 448, 0, 1, 1]
    ]
    return RepViT(cfgs, init_cfg=init_cfg, pretrained=pretrained, distillation=distillation, out_indices=out_indices, **kwargs)


@BACKBONES.register_module()
def repvit_m1_1(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, init_cfg=init_cfg, pretrained=pretrained, distillation=distillation, out_indices=out_indices, **kwargs)

@BACKBONES.register_module()
def repvit_m1_5(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, init_cfg=init_cfg, pretrained=pretrained, distillation=distillation, out_indices=out_indices)


@BACKBONES.register_module()
def repvit_m2_3(pretrained=False, num_classes = 1000, distillation=False, init_cfg=None, out_indices=[], **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 0, 2],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  320, 0, 1, 2],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        # [3,   2, 320, 1, 1, 1],
        # [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 640, 0, 1, 2],
        [3,   2, 640, 1, 1, 1],
        [3,   2, 640, 0, 1, 1],
        # [3,   2, 640, 1, 1, 1],
        # [3,   2, 640, 0, 1, 1]
    ]    
    return RepViT(cfgs, init_cfg=init_cfg, pretrained=pretrained, distillation=distillation, out_indices=out_indices)