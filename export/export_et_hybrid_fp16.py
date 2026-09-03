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

"""
ExecuTorch 三件套导出（按平台分支选 backend）：

  iOS:     prefix_enc / prefill / step 全部 Core ML fp16（compute_unit=ALL，
           ANE/GPU/CPU 自动调度）。三件套合计 ~138 MB，为当前 iOS 部署版。
           prefix_enc 必须走 PrefixEncUnrolledModule（手术展开 stroke encoder
           的 nn.TransformerEncoderLayer），否则 in_proj_weight 双消费触发
           Core ML fp16 lower 图重写 bug，cos 跌到 0.55、端侧 EM 0%。
           详见 docs_final/mobile/2026-05-05_prefix_enc展开修复.md。

  Android: prefix_enc / prefill / step 全部 XNNPACK fp32。此路径作为 fp32
           ExecuTorch 基线保留；Android 实际部署已切 LiteRT，见
           air_calculator_py/export/export_tf_android_fp32.py。

用法：
  cd air_calculator_py/export
  python3 export_et_hybrid_fp16.py --platform ios \
    --ckpt ../../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
    --data-dir ../../dataset/mathwriting-2024 \
    --out ../../weights/exports/executorch_fp16_ios
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── PrefixEnc 标准版（Android XNNPACK 路径）─────────────────────────

class PrefixEncModule(nn.Module):
    """VisualPrefixEncoder 导出包装。img + stroke → prefix_embeds 固定形状。
    AdaptiveAvgPool1d 用 avg_pool2d 实现（T 固定 → kernel = T // n_out）。"""
    def __init__(self, prefix_enc: nn.Module, stroke_len: int):
        super().__init__()
        pe = prefix_enc
        se = pe.stroke_encoder
        self.img_encoder   = pe.img_encoder
        self.img_proj      = pe.img_proj
        self.img_norm      = pe.img_norm
        self.stroke_proj_in  = se.proj
        self.stroke_enc      = se.encoder
        self.n_stroke_out    = se.n_out
        self.stroke_pool_k   = stroke_len // se.n_out
        self.stroke_proj_out = pe.stroke_proj
        self.stroke_norm     = pe.stroke_norm
        self.stroke_len      = stroke_len

    def forward(self, img: torch.Tensor,
                stroke: torch.Tensor,
                stroke_real_len: torch.Tensor) -> torch.Tensor:
        if self.use_img:
            img_tok = self.img_encoder(img)
            img_emb = self.img_norm(self.img_proj(img_tok))

        x = self.stroke_proj_in(stroke)
        positions = torch.arange(self.stroke_len, device=stroke.device)
        pad_mask = positions.unsqueeze(0) >= stroke_real_len.long()
        x = self.stroke_enc(x, src_key_padding_mask=pad_mask)

        xt = x.transpose(1, 2).unsqueeze(2)
        xt = F.avg_pool2d(xt, kernel_size=(1, self.stroke_pool_k),
                          stride=(1, self.stroke_pool_k))
        xt = xt.squeeze(2)
        stroke_tok = xt.transpose(1, 2)
        stroke_emb = self.stroke_norm(self.stroke_proj_out(stroke_tok))
        if not self.use_img:
            return stroke_emb
        return torch.cat([img_emb, stroke_emb], dim=1)


def _build_logit_bias(tokens: list) -> torch.Tensor:
    bias    = torch.zeros(len(tokens))
    boost   = set('0123456789+-=.,;()[]{}/')
    penalize = {'^': -2.0, '_': -1.0}
    for i, tok in enumerate(tokens):
        if tok in boost:
            bias[i] = 0.3
        elif tok in penalize:
            bias[i] = penalize[tok]
    return bias


# ── KV-cache decoder 模块（in_proj_weight 单消费展开版）──────────────

def _layer_prefill(layer, h, nhead, head_dim, d_model):
    """norm_first encoder layer 全展开，附带 K/V 输出。
    in_proj_weight 只被消费 1 次 → PT2E 静态量化无 partition cycle。"""
    h_norm = layer.norm1(h)
    qkv = F.linear(h_norm, layer.self_attn.in_proj_weight,
                   layer.self_attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    q_mh = q.unflatten(-1, (nhead, head_dim)).transpose(1, 2)
    k_mh = k.unflatten(-1, (nhead, head_dim)).transpose(1, 2)
    v_mh = v.unflatten(-1, (nhead, head_dim)).transpose(1, 2)

    out = F.scaled_dot_product_attention(q_mh, k_mh, v_mh)
    out = out.transpose(1, 2).reshape(h.shape[0], h.shape[1], d_model)
    out = F.linear(out, layer.self_attn.out_proj.weight,
                   layer.self_attn.out_proj.bias)
    h = h + out

    h_norm2 = layer.norm2(h)
    ff = layer.activation(F.linear(h_norm2, layer.linear1.weight, layer.linear1.bias))
    ff = F.linear(ff, layer.linear2.weight, layer.linear2.bias)
    h = h + ff
    return h, k, v


class DecoderPrefillKVModule(nn.Module):
    """Prefill：返回单个 (n_layers, 2, 1, n_prefix, d_model) KV 张量。"""
    def __init__(self, decoder: nn.Module, n_prefix: int,
                 n_layers: int, d_model: int, nhead: int):
        super().__init__()
        self.n_prefix = n_prefix
        self.n_layers = n_layers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.pos_embed = decoder.pos_embed
        self.drop = decoder.drop
        self.layer_list = decoder.layers.layers

    def forward(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.n_prefix, device=prefix_embeds.device).unsqueeze(0)
        h = self.drop(prefix_embeds + self.pos_embed(positions))
        kv_list = []
        for layer in self.layer_list:
            h, k, v = _layer_prefill(layer, h, self.nhead, self.head_dim, self.d_model)
            kv_list.append(torch.stack([k, v], dim=0))
        return torch.stack(kv_list, dim=0)


class DecoderTextStepKVModule(nn.Module):
    """单步 decode：(token_id, pos, past_kv) → (logits, new_kv)。"""
    def __init__(self, decoder: nn.Module, logit_bias: torch.Tensor,
                 n_layers: int, d_model: int, nhead: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.tok_embed = decoder.tok_embed
        self.pos_embed = decoder.pos_embed
        self.drop = decoder.drop
        self.layer_list = decoder.layers.layers
        self.norm = decoder.norm
        self.head = decoder.head
        self.register_buffer('logit_bias', logit_bias)

    def _layer_step(self, x, past_k, past_v, layer):
        x_norm = layer.norm1(x)
        qkv = F.linear(x_norm, layer.self_attn.in_proj_weight,
                       layer.self_attn.in_proj_bias)
        q, k_new, v_new = qkv.chunk(3, dim=-1)

        k_full = torch.cat([past_k, k_new], dim=1)
        v_full = torch.cat([past_v, v_new], dim=1)

        q_mh = q.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
        k_mh = k_full.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
        v_mh = v_full.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)

        out = F.scaled_dot_product_attention(q_mh, k_mh, v_mh)
        out = out.transpose(1, 2).reshape(1, 1, self.d_model)
        out = F.linear(out, layer.self_attn.out_proj.weight,
                       layer.self_attn.out_proj.bias)
        x = x + out

        x_norm2 = layer.norm2(x)
        ff = layer.activation(F.linear(x_norm2, layer.linear1.weight, layer.linear1.bias))
        ff = F.linear(ff, layer.linear2.weight, layer.linear2.bias)
        x = x + ff
        return x, k_full, v_full

    def forward(self, token_id, pos, past_kv):
        pos_idx = pos.long().view(1, 1)
        x = self.drop(self.tok_embed(token_id.long()) + self.pos_embed(pos_idx))

        kv_per_layer = past_kv.unbind(0)
        new_kv_list = []
        for layer, kv_l in zip(self.layer_list, kv_per_layer):
            past_k = kv_l[0]
            past_v = kv_l[1]
            x, k_full, v_full = self._layer_step(x, past_k, past_v, layer)
            new_kv_list.append(torch.stack([k_full, v_full], dim=0))
        new_kv = torch.stack(new_kv_list, dim=0)

        logits = self.head(self.norm(x[:, -1, :])) + self.logit_bias
        return logits, new_kv


# ── PrefixEnc 展开版（iOS Core ML fp16 必需）─────────────────────────

def _stroke_layer_forward(layer: nn.Module, h: torch.Tensor,
                          nhead: int, head_dim: int, d_model: int,
                          key_pad_mask):
    """norm_first nn.TransformerEncoderLayer 全展开（self-attn + FF）。
    qkv 只算一次，避免 in_proj_weight 隐式多消费。
    out_proj / linear1 / linear2 显式 .to(h.dtype)，兼容 weight-only fp16 路径。
    !! in_proj_weight / in_proj_bias 严禁加 .to() —— 单消费敏感，详见
       docs_final/2026-05-05_prefix_enc展开修复.md。"""
    h_norm = layer.norm1(h)
    qkv = F.linear(h_norm, layer.self_attn.in_proj_weight,
                   layer.self_attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    B, T = h.shape[0], h.shape[1]
    q_mh = q.unflatten(-1, (nhead, head_dim)).transpose(1, 2)
    k_mh = k.unflatten(-1, (nhead, head_dim)).transpose(1, 2)
    v_mh = v.unflatten(-1, (nhead, head_dim)).transpose(1, 2)

    if key_pad_mask is not None:
        attn_mask = torch.zeros(B, 1, 1, T, dtype=h.dtype, device=h.device)
        attn_mask = attn_mask.masked_fill(
            key_pad_mask[:, None, None, :], float('-inf'))
    else:
        attn_mask = None
    out = F.scaled_dot_product_attention(q_mh, k_mh, v_mh, attn_mask=attn_mask)
    out = out.transpose(1, 2).reshape(B, T, d_model)
    out = F.linear(out,
                   layer.self_attn.out_proj.weight.to(out.dtype),
                   layer.self_attn.out_proj.bias.to(out.dtype))
    h = h + out

    h_norm2 = layer.norm2(h)
    ff = layer.activation(F.linear(h_norm2,
                                   layer.linear1.weight.to(h_norm2.dtype),
                                   layer.linear1.bias.to(h_norm2.dtype)))
    ff = F.linear(ff,
                  layer.linear2.weight.to(ff.dtype),
                  layer.linear2.bias.to(ff.dtype))
    h = h + ff
    return h


class PrefixEncUnrolledModule(nn.Module):
    """与 PrefixEncModule 接口一致的展开版。stroke 分支手动展开 TransformerEncoderLayer，
    绕过 Core ML fp16 / PT2E XNNPACK int8 共有的 in_proj_weight 双消费图重写 bug。
    详见 docs_final/2026-05-05_prefix_enc展开修复.md。"""
    def __init__(self, prefix_enc: nn.Module, stroke_len: int):
        super().__init__()
        pe = prefix_enc
        se = pe.stroke_encoder
        self.use_img = getattr(pe, 'use_img', True)
        if self.use_img:
            self.img_encoder = pe.img_encoder
            self.img_proj = pe.img_proj
            self.img_norm = pe.img_norm
        self.stroke_proj_in = se.proj
        self.stroke_layers = se.encoder.layers
        self.stroke_n_out = se.n_out
        self.stroke_pool_k = stroke_len // se.n_out
        self.stroke_proj_out = pe.stroke_proj
        self.stroke_norm = pe.stroke_norm
        self.stroke_len = stroke_len
        self.stroke_d = se.d_model
        attn0 = self.stroke_layers[0].self_attn
        self.stroke_nhead = attn0.num_heads
        self.stroke_head_dim = self.stroke_d // self.stroke_nhead

    def forward(self, img: torch.Tensor,
                stroke: torch.Tensor,
                stroke_real_len: torch.Tensor) -> torch.Tensor:
        if self.use_img:
            img_tok = self.img_encoder(img)
            img_emb = self.img_norm(self.img_proj(img_tok))

        x = self.stroke_proj_in(stroke)
        positions = torch.arange(self.stroke_len, device=stroke.device)
        pad_mask = positions.unsqueeze(0) >= stroke_real_len.long()
        for layer in self.stroke_layers:
            x = _stroke_layer_forward(
                layer, x,
                self.stroke_nhead, self.stroke_head_dim, self.stroke_d,
                pad_mask)

        xt = x.transpose(1, 2).unsqueeze(2)
        xt = F.avg_pool2d(xt,
                          kernel_size=(1, self.stroke_pool_k),
                          stride=(1, self.stroke_pool_k))
        xt = xt.squeeze(2)
        stroke_tok = xt.transpose(1, 2)
        stroke_emb = self.stroke_norm(self.stroke_proj_out(stroke_tok))
        if not self.use_img:
            return stroke_emb
        return torch.cat([img_emb, stroke_emb], dim=1)


# ── 混合精度模块（fp16 linear/matmul + fp32 LayerNorm/SDPA） ──────────
# 目的：XNNPACK 不会对 fp16 自动做累加 fp32（Core ML 会），所以在数值敏感的
# LayerNorm / scaled_dot_product_attention 上手动留 fp32，其他算子走 fp16。
# 外部接口仍 fp32（prefix_embeds / past_kv / logits / new_kv），App 不动。

_F16 = torch.float16
_F32 = torch.float32


def _to_fp16_keep_ln_fp32(decoder_mod: torch.nn.Module) -> torch.nn.Module:
    """整体 .half()，再把所有 LayerNorm 权重 / bias 还原 fp32（含子层）。"""
    decoder_mod.half()
    for m in decoder_mod.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.float()
    return decoder_mod


def _attn_mixed(q, k, v, nhead, head_dim, seq_shape, out_proj):
    """fp16 输入 → SDPA 内部 fp32 → fp16 out_proj。"""
    q_mh = q.unflatten(-1, (nhead, head_dim)).transpose(1, 2).to(_F32)
    k_mh = k.unflatten(-1, (nhead, head_dim)).transpose(1, 2).to(_F32)
    v_mh = v.unflatten(-1, (nhead, head_dim)).transpose(1, 2).to(_F32)
    out = F.scaled_dot_product_attention(q_mh, k_mh, v_mh)
    bsz, t, d_model = seq_shape
    out = out.transpose(1, 2).reshape(bsz, t, d_model).to(_F16)
    return F.linear(out, out_proj.weight, out_proj.bias)


class _PrefillMixedFp16(torch.nn.Module):
    """Prefill 混合精度。外部 fp32 prefix_embeds → fp32 KV。"""
    def __init__(self, decoder, n_prefix, n_layers, d_model, nhead):
        super().__init__()
        self.n_prefix = n_prefix
        self.n_layers = n_layers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.pos_embed = decoder.pos_embed   # fp16
        self.drop = decoder.drop
        self.layer_list = decoder.layers.layers  # fp16 except LN

    def forward(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        # pos_embed 是 fp16 → 加之前 cast
        positions = torch.arange(self.n_prefix, device=prefix_embeds.device).unsqueeze(0)
        pe = self.pos_embed(positions).to(_F32)
        h = self.drop(prefix_embeds + pe)  # fp32 残差流

        kv_list = []
        for layer in self.layer_list:
            h_norm = layer.norm1(h)                        # fp32 LN
            h_norm_h = h_norm.to(_F16)
            qkv = F.linear(h_norm_h, layer.self_attn.in_proj_weight,
                           layer.self_attn.in_proj_bias)   # fp16
            q, k, v = qkv.chunk(3, dim=-1)
            attn_out = _attn_mixed(
                q, k, v, self.nhead, self.head_dim,
                (h.shape[0], h.shape[1], self.d_model),
                layer.self_attn.out_proj,
            )
            h = h + attn_out.to(_F32)

            h_norm2 = layer.norm2(h)                       # fp32 LN
            h_norm2_h = h_norm2.to(_F16)
            ff = layer.activation(F.linear(h_norm2_h, layer.linear1.weight, layer.linear1.bias))
            ff = F.linear(ff, layer.linear2.weight, layer.linear2.bias)
            h = h + ff.to(_F32)

            kv_list.append(torch.stack([k, v], dim=0))     # KV 存 fp16
        kv = torch.stack(kv_list, dim=0)
        return kv.to(_F32)                                 # 出口 fp32


class _StepMixedFp16(torch.nn.Module):
    """Step 混合精度。外部 fp32 past_kv → fp32 (logits, new_kv)。"""
    def __init__(self, decoder, logit_bias, n_layers, d_model, nhead):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.tok_embed = decoder.tok_embed   # fp16
        self.pos_embed = decoder.pos_embed   # fp16
        self.drop = decoder.drop
        self.layer_list = decoder.layers.layers
        self.norm = decoder.norm             # fp32（LN 还原过）
        self.head = decoder.head             # fp16 Linear
        self.register_buffer('logit_bias', logit_bias.to(_F32))

    def forward(self, token_id, pos, past_kv):
        # past_kv 入参 fp32（外部接口），cast fp16 进入计算
        past_kv_h = past_kv.to(_F16)
        pos_idx = pos.long().view(1, 1)
        emb = (self.tok_embed(token_id.long()) + self.pos_embed(pos_idx)).to(_F32)
        x = self.drop(emb)  # fp32

        kv_per_layer = past_kv_h.unbind(0)
        new_kv_list = []
        for layer, kv_l in zip(self.layer_list, kv_per_layer):
            past_k = kv_l[0]   # fp16
            past_v = kv_l[1]   # fp16

            x_norm = layer.norm1(x)                        # fp32 LN
            x_norm_h = x_norm.to(_F16)
            qkv = F.linear(x_norm_h, layer.self_attn.in_proj_weight,
                           layer.self_attn.in_proj_bias)
            q, k_new, v_new = qkv.chunk(3, dim=-1)
            k_full = torch.cat([past_k, k_new], dim=1)     # fp16
            v_full = torch.cat([past_v, v_new], dim=1)
            attn_out = _attn_mixed(
                q, k_full, v_full, self.nhead, self.head_dim,
                (1, 1, self.d_model),
                layer.self_attn.out_proj,
            )
            x = x + attn_out.to(_F32)

            x_norm2 = layer.norm2(x)
            x_norm2_h = x_norm2.to(_F16)
            ff = layer.activation(F.linear(x_norm2_h, layer.linear1.weight, layer.linear1.bias))
            ff = F.linear(ff, layer.linear2.weight, layer.linear2.bias)
            x = x + ff.to(_F32)

            new_kv_list.append(torch.stack([k_full, v_full], dim=0))
        new_kv = torch.stack(new_kv_list, dim=0)

        last_norm = self.norm(x[:, -1, :])                 # fp32
        logits = F.linear(last_norm.to(_F16),
                          self.head.weight, self.head.bias).to(_F32)
        logits = logits + self.logit_bias
        return logits, new_kv.to(_F32)


# ── 后端 lower 函数 ─────────────────────────────────────────────────

def _lower_xnnpack_fp32(exported, name: str):
    """prefix_enc 用：fp32 XNNPACK，CPU 后备由 ExecuTorch 自动接管未 partition 的 op。"""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f'  [{name}] lower → XNNPACK fp32')
    return to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()


def _lower_coreml(exported, name: str, precision='fp16'):
    """decoder iOS：Core ML，compute_unit=ALL 让 NE/GPU/CPU 都可用。
    precision='fp16' 走 ANE 友好的半精度；'fp32' 留作基线对比（可能落到 GPU/CPU）。"""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.apple.coreml.compiler.coreml_preprocess import CoreMLBackend
    import coremltools as ct

    prec = ct.precision.FLOAT16 if precision == 'fp16' else ct.precision.FLOAT32
    print(f'  [{name}] lower → Core ML {precision} (compute_unit=ALL)')
    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_unit=ct.ComputeUnit.ALL,
        compute_precision=prec,
    )
    return to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
    ).to_executorch()


def _lower_coreml_fp16(exported, name: str):
    return _lower_coreml(exported, name, precision='fp16')


def _lower_coreml_fp32(exported, name: str):
    return _lower_coreml(exported, name, precision='fp32')


def _lower_xnnpack_fp16(exported, name: str):
    """decoder Android fp16：XNNPACK 自动按 graph dtype 选 fp16 kernel。
    需要模型/example_inputs 已经 cast 成 fp16；ARMv8.2-a FP16 扩展（A55/A75+）原生支持。
    不被 partition 的 op 落到 portable kernel 走 fp16，可能慢于 fp32 partition——必须 dump 覆盖率确认。"""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f'  [{name}] lower → XNNPACK fp16')
    return to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()


def _lower_coreml_multi(eps: dict, precision='fp16'):
    """多方法一次 lower：{method_name: ExportedProgram} → 单个 PTE。
    prefill 与 step 共享同一份 decoder 权重，合并后省掉一份权重副本。"""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.backends.apple.coreml.compiler.coreml_preprocess import CoreMLBackend
    import coremltools as ct

    prec = ct.precision.FLOAT16 if precision == 'fp16' else ct.precision.FLOAT32
    print(f'  [decoder_multi] lower → Core ML {precision} (methods={list(eps)})')
    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_unit=ct.ComputeUnit.ALL,
        compute_precision=prec,
    )
    return to_edge_transform_and_lower(
        eps,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
    ).to_executorch()


def _lower_xnnpack_multi(eps: dict, fp16: bool = False):
    """Android 多方法版本。"""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f'  [decoder_multi] lower → XNNPACK {"fp16" if fp16 else "fp32"} (methods={list(eps)})')
    return to_edge_transform_and_lower(
        eps,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()


# ── 通用导出 ────────────────────────────────────────────────────────

def _export(module, example_inputs, dynamic_shapes, lower_fn, out_path: Path, name: str):
    from torch.export import export

    print(f'\n[{name}] export …')
    ep = export(module, example_inputs, dynamic_shapes=dynamic_shapes, strict=False)
    et = lower_fn(ep, name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et.buffer)
    print(f'[{name}] saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)')


# ── 主流程 ──────────────────────────────────────────────────────────

def _write_vocab(out_dir, vocab):
    (out_dir / 'vocab.json').write_text(
        json.dumps({
            'tokens': vocab.idx2token,
            'bos_idx': vocab.BOS,
            'eos_idx': vocab.EOS,
            'pad_idx': vocab.PAD,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def main():
    try:
        import executorch  # noqa: F401
    except ImportError:
        print('需要 executorch'); sys.exit(1)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--platform', required=True, choices=['ios', 'android'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', default=None,
                    help='默认 export_v3_fp16_hybrid_{platform}')
    ap.add_argument('--max-decode', type=int, default=32)
    ap.add_argument('--merge-decoder', action='store_true',
                    help='把 prefill 与 step 合并成单个多签名 decoder.pte（共享权重，省一份副本）')
    ap.add_argument('--android-decoder-fp16', action='store_true',
                    help='Android: decoder 走 XNNPACK fp16（实验性，需在 ARMv8.2-a FP16 设备上验证 EM 与延迟）')
    ap.add_argument('--ios-decoder-fp32', action='store_true',
                    help='iOS: decoder 走 Core ML fp32（基线对比用，留意 ANE 不接受 fp32 会落 GPU/CPU）')
    ap.add_argument('--only', choices=['all', 'prefix_enc', 'decoder'], default='all',
                    help='只导其中一件。改了 prefix_enc 的签名时不必连带重导 decoder')
    args = ap.parse_args()

    out_dir = Path(args.out or f'export_v3_fp16_hybrid_{args.platform}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 平台后端选择 ───────────────────────────────────────────────
    if args.platform == 'ios':
        if args.ios_decoder_fp32:
            prefill_lower = _lower_coreml_fp32
            step_lower = _lower_coreml_fp32
            backend_tag = 'prefill=coreml_fp32, step=coreml_fp32 (基线)'
        else:
            prefill_lower = _lower_coreml_fp16
            step_lower = _lower_coreml_fp16
            backend_tag = 'prefill=coreml_fp16, step=coreml_fp16'
        decoder_dtype = torch.float32  # 上层 graph 都按 fp32 描述，Core ML 编译时转目标精度
    elif args.android_decoder_fp16:
        # Android 实验：decoder 内部 fp16 / XNNPACK，外部接口仍 fp32（包装类做 cast）
        prefill_lower = _lower_xnnpack_fp16
        step_lower = _lower_xnnpack_fp16
        decoder_dtype = torch.float32  # 外部 example 维持 fp32；内部 .half() 在包装时做
        backend_tag = 'prefill=xnnpack_fp16, step=xnnpack_fp16 (实验, fp32 IO)'
    else:
        # Android: 全 XNNPACK fp32（fp32 ExecuTorch 基线；实际部署走 TFLite）
        prefill_lower = _lower_xnnpack_fp32
        step_lower = _lower_xnnpack_fp32
        decoder_dtype = torch.float32
        backend_tag = 'prefill=xnnpack_fp32, step=xnnpack_fp32'
    print(f'\n=== 平台: {args.platform}  prefix=XNNPACK fp32  {backend_tag} ===\n')

    # ── 加载 ckpt ──────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cargs = ckpt.get('args', {})
    d_model = cargs.get('d_model', 512)
    n_layers = cargs.get('n_layers', 8)
    nhead = cargs.get('nhead', 8)
    max_src = cargs.get('max_src', 512)
    n_stroke_tok = cargs.get('n_stroke_tok', 32)
    modality = cargs.get('modality', 'both')
    n_prefix = (64 if modality in ('both', 'image_only') else 0) + \
               (n_stroke_tok if modality in ('both', 'stroke_only') else 0)
    max_seq = ckpt['decoder']['pos_embed.weight'].shape[0]

    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import Vocabulary
    from models import VisualPrefixEncoder, CausalTransformerDecoder

    vocab = Vocabulary.load(Path(args.data_dir) / 'vocab.json')
    prefix_enc = VisualPrefixEncoder(d_model=d_model, n_stroke=n_stroke_tok,
                                     modality=modality).eval()
    print(f'  [modality] {modality}  n_prefix={n_prefix}')
    decoder = CausalTransformerDecoder(
        vocab_size=len(vocab), d_model=d_model, nhead=nhead,
        n_layers=n_layers, max_seq=max_seq,
    ).eval()
    prefix_enc.load_state_dict(ckpt['prefix_enc'])
    decoder.load_state_dict(ckpt['decoder'])
    logit_bias = _build_logit_bias(vocab.idx2token)

    from torch.export import Dim

    # ── 1. prefix_enc ─────────────────────────────────────────────
    # 两个平台都用 PrefixEncUnrolledModule：手术展开 stroke encoder 的
    # nn.TransformerEncoderLayer，绕过 in_proj_weight 双消费触发的图重写 bug。
    # 这个 bug 在 Core ML fp16 与 XNNPACK 上都存在（见该类 docstring 与
    # docs_final/mobile/2026-05-05_prefix_enc展开修复.md），所以不分平台。
    # 后端仍按平台分：iOS 走 Core ML fp16，Android 走 XNNPACK fp32。
    #
    # 上面的 PrefixEncModule 是展开版出现之前的标准实现，现已无人调用——
    # 它的 forward 引用 self.use_img 而 __init__ 从未设过，纯轨迹模型上直接崩。
    enc_mod = PrefixEncUnrolledModule(prefix_enc, stroke_len=max_src).eval()
    if args.platform == 'ios':
        enc_lower = _lower_coreml_fp16
        enc_tag = 'PrefixEncUnrolledModule + Core ML fp16'
    else:
        enc_lower = _lower_xnnpack_fp32
        enc_tag = 'PrefixEncUnrolledModule + XNNPACK fp32'
    print(f'  [prefix_enc] {enc_tag}')
    skip_enc = args.only == 'decoder'
    skip_dec = args.only == 'prefix_enc'
    # stroke_real_len 走 f32，与 Android 的 .tflite 签名一致。
    #
    # 两端共用同一份 Rust 解码代码（mwh-decode），它按 f32 传这个输入。iOS 这边
    # 原先导成 int32，ExecuTorch 在 execute 时直接拒绝：
    #   Input 2 has unexpected scalar type: expected Int but was Float
    # 于是 iOS 上识别每次都失败。图里 stroke_real_len 只参与 `positions >=
    # real_len.long()` 这一个比较，收浮点多一个 cast，Core ML 吃得下
    # （TFLite 那边交不出去，所以 Android 脚本用铰链差绕开，见其注释）。
    enc_ex = (
        torch.zeros(1, 1, 64, 256),
        torch.zeros(1, max_src, 13),
        torch.tensor([16.0], dtype=torch.float32),
    )
    if not skip_enc:
        _export(
            enc_mod, enc_ex,
            {'img': None, 'stroke': None, 'stroke_real_len': None},
            enc_lower,
            out_dir / 'prefix_enc.pte',
            'prefix_enc',
        )

    # ── 2. decoder prefill：Core ML / XNNPACK fp32 / XNNPACK 混合精度 fp16 ──
    if skip_dec:
        print('\n[--only prefix_enc] 跳过 decoder')
        _write_vocab(out_dir, vocab)
        return
    if args.platform == 'android' and args.android_decoder_fp16:
        decoder_fp16 = _to_fp16_keep_ln_fp32(decoder)
        prefill_mod = _PrefillMixedFp16(
            decoder_fp16, n_prefix, n_layers, d_model, nhead).eval()
    else:
        prefill_mod = DecoderPrefillKVModule(
            decoder, n_prefix, n_layers, d_model, nhead).eval().to(decoder_dtype)
    prefill_ex = (torch.zeros(1, n_prefix, d_model, dtype=decoder_dtype),)
    if not args.merge_decoder:
        _export(
            prefill_mod, prefill_ex,
            None,
            prefill_lower,
            out_dir / 'decoder_prefill_kv.pte',
            'decoder_prefill',
        )

    # ── 3. decoder step：Core ML / XNNPACK fp32 / XNNPACK 混合精度 fp16 ──
    if args.platform == 'android' and args.android_decoder_fp16:
        # decoder 已经在 prefill 段被 _to_fp16_keep_ln_fp32 处理过，复用
        step_mod = _StepMixedFp16(
            decoder, logit_bias, n_layers, d_model, nhead).eval()
    else:
        step_mod = DecoderTextStepKVModule(
            decoder, logit_bias, n_layers, d_model, nhead).eval().to(decoder_dtype)
    kv_len = Dim('kv_len', min=n_prefix, max=n_prefix + args.max_decode + 2)
    # token_id / pos 同样走 f32，理由与 prefix_enc 的 stroke_real_len 一样：
    # 共用的 Rust 解码循环按 f32 传（见 mwh-decode 里「导出侧的 onehot 查表收
    # 浮点」那段注释），iOS 这边原先收 int32，step 每步都被 ExecuTorch 拒掉：
    #   Input 0 has unexpected scalar type: expected Int but was Float
    # forward 里本来就是 token_id.long() / pos.long()，收浮点只是把那个 cast
    # 从调用方挪进图里，整数值在 f32 下精确表示，没有精度问题。
    step_ex = (
        torch.zeros(1, 1, dtype=torch.float32),
        torch.tensor([float(n_prefix)], dtype=torch.float32),
        torch.zeros(n_layers, 2, 1, n_prefix, d_model, dtype=decoder_dtype),
    )
    if args.merge_decoder:
        # 合并：prefill + step 进同一个 PTE，两个方法共享 decoder 权重
        from torch.export import export as _tex
        print('\n[decoder] 合并导出 prefill + step 多签名 …')
        eps = {
            'prefill': _tex(prefill_mod, prefill_ex, strict=False),
            'step': _tex(step_mod, step_ex,
                         dynamic_shapes={'token_id': None, 'pos': None,
                                         'past_kv': {3: kv_len}},
                         strict=False),
        }
        if args.platform == 'ios':
            et = _lower_coreml_multi(
                eps, precision='fp32' if args.ios_decoder_fp32 else 'fp16')
        else:
            et = _lower_xnnpack_multi(eps, fp16=args.android_decoder_fp16)
        merged = out_dir / 'decoder.pte'
        merged.write_bytes(et.buffer)
        print(f'[decoder] saved → {merged}  ({merged.stat().st_size / 1024:.0f} KB)')
    else:
        _export(
            step_mod, step_ex,
            {'token_id': None, 'pos': None, 'past_kv': {3: kv_len}},
            step_lower,
            out_dir / 'decoder_step_kv.pte',
            'decoder_step',
        )

    # ── 4. vocab.json ─────────────────────────────────────────────
    _write_vocab(out_dir, vocab)


    # ── 5. 体积汇总 ───────────────────────────────────────────────
    print(f'\n─── {args.platform} 三件套体积 ───')
    total = 0
    for f in sorted(out_dir.glob('*.pte')):
        sz = f.stat().st_size
        total += sz
        print(f'  {f.name:35s} {sz / 1024 / 1024:6.1f} MB')
    print(f'  {"合计":35s} {total / 1024 / 1024:6.1f} MB')
    print(f'\n下一步：把 {out_dir}/ 拷到 air_calculator/platform_models/{args.platform}/')


if __name__ == '__main__':
    main()
