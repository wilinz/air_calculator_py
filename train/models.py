#!/usr/bin/env python3
# Copyright 2026 wilinz.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decoder-only 模型定义：StrokeEncoder / VisualPrefixEncoder / CausalTransformerDecoder。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stroke_features import FEATURE_DIM
from image_encoder import ImageEncoder
from stroke_renderer import IMG_H, IMG_W


# ── 笔画编码器 ──────────────────────────────────────────────────────── #

class StrokeEncoder(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, d_model: int = 256,
                 n_out: int = 32, nhead: int = 8, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.pool = nn.AdaptiveAvgPool1d(n_out)
        self.n_out = n_out
        self.d_model = d_model

    def forward(self, src, src_pad_mask=None):
        x = self.proj(src)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        xt = x.transpose(1, 2)
        if xt.device.type == 'mps':
            xt = self.pool(xt.cpu()).to(xt.device)
        else:
            xt = self.pool(xt)
        return xt.transpose(1, 2)


# ── 视觉前缀编码器 ──────────────────────────────────────────────────── #

class VisualPrefixEncoder(nn.Module):
    """编码图像+笔画为 prefix tokens (B, N_prefix, d_model)。

    modality:
      - 'both'        : 图像 64 token + 笔画 32 token 拼接（默认）
      - 'image_only'  : 仅图像 64 token（消融：去笔画分支）
      - 'stroke_only' : 仅笔画 32 token（消融：去图像分支）
    """

    def __init__(self, d_model: int,
                 d_img: int = 384, n_img: int = 64,
                 d_stroke: int = 256, n_stroke: int = 32,
                 modality: str = 'both'):
        super().__init__()
        assert modality in ('both', 'image_only', 'stroke_only'), modality
        self.modality = modality

        self.use_img = modality in ('both', 'image_only')
        self.use_stroke = modality in ('both', 'stroke_only')

        n_prefix = 0
        if self.use_img:
            self.img_encoder = ImageEncoder(d_model=d_img, img_h=IMG_H, img_w=IMG_W)
            self.img_proj = nn.Linear(d_img, d_model)
            self.img_norm = nn.LayerNorm(d_model)
            n_prefix += n_img
        if self.use_stroke:
            self.stroke_encoder = StrokeEncoder(
                input_dim=FEATURE_DIM, d_model=d_stroke,
                n_out=n_stroke, nhead=8, n_layers=2,
            )
            self.stroke_proj = nn.Linear(d_stroke, d_model)
            self.stroke_norm = nn.LayerNorm(d_model)
            n_prefix += n_stroke
        self.n_prefix = n_prefix

    def forward(self, img, stroke, stroke_pad_mask=None):
        embs = []
        if self.use_img:
            img_tok = self.img_encoder(img)
            embs.append(self.img_norm(self.img_proj(img_tok)))
        if self.use_stroke:
            stroke_tok = self.stroke_encoder(stroke, stroke_pad_mask)
            embs.append(self.stroke_norm(self.stroke_proj(stroke_tok)))
        return torch.cat(embs, dim=1) if len(embs) > 1 else embs[0]


# ── 因果 Transformer Decoder（从头训练）─────────────────────────────── #

class CausalTransformerDecoder(nn.Module):
    """从头训练的 decoder-only transformer。

    输入序列: [prefix(96) | BOS | tok1 | ... | tokN]
    - prefix 部分由 VisualPrefixEncoder 提供 embeddings（不经过 token embedding）
    - 文本部分使用 token embedding + positional encoding
    - 统一因果自注意力：文本 token 可 attend 到所有 prefix token
    """

    def __init__(self, vocab_size: int, d_model: int = 384,
                 nhead: int = 8, n_layers: int = 6,
                 dropout: float = 0.1, max_seq: int = 256):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定：embedding 与 output head 共享
        self.head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for p in self.layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, prefix_embeds, token_ids, prefix_pad_mask=None):
        """
        Args:
            prefix_embeds: (B, N_prefix, d_model)
            token_ids:     (B, T_text) — [BOS, tok1, ..., tokN]
        Returns:
            logits: (B, T_text, vocab_size)
        """
        B, N_pre, _ = prefix_embeds.shape
        T_text = token_ids.size(1)
        total_len = N_pre + T_text

        text_emb = self.tok_embed(token_ids)
        seq_emb = torch.cat([prefix_embeds, text_emb], dim=1)

        positions = torch.arange(total_len, device=seq_emb.device).unsqueeze(0)
        seq_emb = seq_emb + self.pos_embed(positions)
        seq_emb = self.drop(seq_emb)

        # 严格因果 mask：prefix 内部全连，文本下三角，prefix 不可见 future text
        causal_mask = torch.zeros(total_len, total_len, device=seq_emb.device)
        text_mask = torch.triu(
            torch.full((T_text, T_text), float('-inf'), device=seq_emb.device),
            diagonal=1,
        )
        causal_mask[N_pre:, N_pre:] = text_mask
        causal_mask[:N_pre, N_pre:] = float('-inf')

        out = self.layers(seq_emb, mask=causal_mask)
        out = self.norm(out)

        text_out = out[:, N_pre:, :]
        logits = self.head(text_out)
        return logits

    @torch.no_grad()
    def generate(self, prefix_embeds, bos_id, eos_id, max_len=64):
        """自回归生成（greedy）。返回不含 BOS 的 token id 列表。"""
        device = prefix_embeds.device
        generated = [bos_id]

        for _ in range(max_len):
            token_ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.forward(prefix_embeds, token_ids)
            next_id = logits[0, -1].argmax().item()

            if next_id == eos_id:
                break
            generated.append(next_id)

        return generated[1:]

    @torch.no_grad()
    def generate_beam(self, prefix_embeds, bos_id, eos_id, max_len=64, beam_width=5):
        """Beam search 生成。返回最佳序列（不含 BOS）。"""
        device = prefix_embeds.device

        beams = [(0.0, [bos_id])]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, seq in beams:
                token_ids = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.forward(prefix_embeds, token_ids)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)

                topk = log_probs.topk(beam_width)
                for k in range(beam_width):
                    tok = topk.indices[k].item()
                    new_score = score + topk.values[k].item()
                    new_seq = seq + [tok]

                    if tok == eos_id:
                        norm_score = new_score / (len(new_seq) - 1)
                        completed.append((norm_score, new_seq[1:]))
                    else:
                        candidates.append((new_score, new_seq))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

            if len(completed) >= beam_width:
                break

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]
        else:
            best = beams[0][1]
            return best[1:]
