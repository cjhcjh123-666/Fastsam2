"""
author: chenjiahui
date: 2025-11-11
description: 在线编码 text 为 CLIP 向量 z_text，缓存后复用
"""
from ext.open_clip import open_clip
import torch

class TextPromptAdapter:
    def __init__(self, model_name='ViT-B-32', pretrained='openai'):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().cuda()

    @torch.no_grad()
    def encode(self, text_list):
        toks = self.tokenizer(text_list).cuda()
        z = self.model.encode_text(toks)
        z = z / z.norm(dim=-1, keepdim=True)
        return z