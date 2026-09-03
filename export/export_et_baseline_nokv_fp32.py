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
ExecuTorch fp32 两件套基线导出（XNNPACK 全平台兼容，无 KV-cache，每步全 forward）。

作为论文 §6 量化对照实验的 fp32 baseline；当前实际部署用 KV-cache 三件套版
（iOS 见 export_et_hybrid_fp16.py，Android 见 export_tf_android_fp32.py）。

输出：
  prefix_enc.pte    —— img + stroke → prefix_embeds (1, 96, d_model)
  decoder_step.pte  —— prefix_embeds + token_ids → logits (1, vocab_size)
  vocab.json

用法：
  cd air_calculator_py/export
  python3 export_et_baseline_nokv_fp32.py \
    --ckpt ../../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
    --data-dir ../../dataset/mathwriting-2024 \
    --out ../../weights/exports/executorch_fp32
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))

import torch
import torch.nn as nn

# ── 可导出子模块 ────────────────────────────────────────────────────

class PrefixEncModule(nn.Module):
    """
    VisualPrefixEncoder 导出包装。
    输入:
      img    (1, 1, IMG_H, IMG_W)  float32  固定尺寸
      stroke (1, STROKE_LEN, 13)   float32  固定长度（Dart 侧补零到 STROKE_LEN）
    输出:
      prefix_embeds (1, N_prefix, d_model)  float32  N_prefix=n_img+n_stroke 固定

    AdaptiveAvgPool1d(n_out) 用 avg_pool2d 实现（与训练完全等价，需要固定 T）：
      T 固定 → kernel_size = T // n_out（整除），avg_pool2d 数值与 adaptive_avg_pool1d 相同。
    """
    def __init__(self, prefix_enc: nn.Module, stroke_len: int):
        super().__init__()
        pe = prefix_enc
        se = pe.stroke_encoder

        # 图像分支
        self.img_encoder   = pe.img_encoder
        self.img_proj      = pe.img_proj
        self.img_norm      = pe.img_norm

        # 笔画分支（拆开，用 avg_pool2d 替换 AdaptiveAvgPool1d）
        self.stroke_proj_in  = se.proj
        self.stroke_enc      = se.encoder
        self.n_stroke_out    = se.n_out                # 32
        self.stroke_pool_k   = stroke_len // se.n_out  # 固定 kernel_size
        self.stroke_proj_out = pe.stroke_proj
        self.stroke_norm     = pe.stroke_norm
        self.stroke_len      = stroke_len

    def forward(self, img: torch.Tensor,
                stroke: torch.Tensor,
                stroke_real_len: torch.Tensor) -> torch.Tensor:
        """
        stroke_real_len: shape (1,) int32，Dart 侧传入实际 stroke 帧数，
                         用于生成 padding mask，使 TransformerEncoder 忽略补零部分。
        """
        # ── 图像分支 ──────────────────────────────────────────────
        img_tok = self.img_encoder(img)
        img_emb = self.img_norm(self.img_proj(img_tok))

        # ── 笔画分支 ──────────────────────────────────────────────
        x = self.stroke_proj_in(stroke)          # (1, stroke_len, d_stroke)

        # Padding mask：True = 该位置是补零，应被 Transformer 忽略
        positions = torch.arange(self.stroke_len, device=stroke.device)  # (stroke_len,)
        pad_mask = positions.unsqueeze(0) >= stroke_real_len.long()       # (1, stroke_len)

        x = self.stroke_enc(x, src_key_padding_mask=pad_mask)            # (1, stroke_len, d_stroke)

        # avg_pool2d 实现 AdaptiveAvgPool1d：(1, d, stroke_len) → (1, d, n_out)
        xt = x.transpose(1, 2).unsqueeze(2)      # (1, d_stroke, 1, stroke_len)
        xt = torch.nn.functional.avg_pool2d(
            xt, kernel_size=(1, self.stroke_pool_k), stride=(1, self.stroke_pool_k)
        )                                        # (1, d_stroke, 1, n_out)
        xt = xt.squeeze(2)                       # (1, d_stroke, n_out)
        stroke_tok = xt.transpose(1, 2)          # (1, n_out, d_stroke)

        stroke_emb = self.stroke_norm(self.stroke_proj_out(stroke_tok))

        return torch.cat([img_emb, stroke_emb], dim=1)  # (1, n_prefix, d_model)


class DecoderStepModule(nn.Module):
    """
    CausalTransformerDecoder 单步推理包装（export-friendly，不调用 model.forward()）。
    输入:
      prefix_embeds (1, N_prefix, d_model)  float32  固定形状
      token_ids     (1, t)                  int32    t 动态（Dart 无 int64）
    输出:
      logits (1, vocab_size)  float32  仅最后位置
    """
    def __init__(self, decoder: nn.Module, logit_bias: torch.Tensor, n_prefix: int):
        super().__init__()
        # 直接引用 decoder 子模块，避免调用含 data-dependent 操作的 forward()
        self.tok_embed = decoder.tok_embed
        self.pos_embed = decoder.pos_embed
        self.drop      = decoder.drop
        self.layers    = decoder.layers
        self.norm      = decoder.norm
        self.head      = decoder.head
        self.n_prefix  = n_prefix
        self.register_buffer('logit_bias', logit_bias)

    def forward(self, prefix_embeds: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        N_pre   = self.n_prefix                   # 静态常量 96
        T_text  = token_ids.shape[1]              # 动态
        total   = N_pre + T_text

        text_emb = self.tok_embed(token_ids.long())            # (1, T, d)
        seq_emb  = torch.cat([prefix_embeds, text_emb], dim=1) # (1, total, d)

        positions = torch.arange(total, device=seq_emb.device).unsqueeze(0)
        seq_emb   = self.drop(seq_emb + self.pos_embed(positions))

        # Causal mask（export-friendly：用 generate_square_subsequent_mask + 块拼接）
        text_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            T_text, device=seq_emb.device, dtype=seq_emb.dtype)  # (T, T) upper=-inf

        zeros_pp  = torch.zeros(N_pre,   N_pre,   device=seq_emb.device, dtype=seq_emb.dtype)
        neginf_pt = torch.full ((N_pre,  T_text), float('-inf'), device=seq_emb.device, dtype=seq_emb.dtype)
        zeros_tp  = torch.zeros(T_text,  N_pre,   device=seq_emb.device, dtype=seq_emb.dtype)

        top    = torch.cat([zeros_pp, neginf_pt], dim=1)   # (N_pre,  total)
        bottom = torch.cat([zeros_tp,  text_mask], dim=1)  # (T_text, total)
        mask   = torch.cat([top, bottom], dim=0)           # (total,  total)

        out = self.layers(seq_emb, mask=mask, is_causal=False)
        out = self.norm(out)
        return self.head(out[:, -1, :]) + self.logit_bias   # (1, vocab)


# ── logit 偏置 ──────────────────────────────────────────────────────

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


# ── 导出辅助 ────────────────────────────────────────────────────────

def _export_pte(module: nn.Module, example_inputs: tuple,
                dynamic_shapes: dict, out_path: Path, name: str,
                use_xnnpack: bool = True) -> None:
    from torch.export import export, Dim          # noqa: F401
    from executorch.exir import EdgeCompileConfig
    from executorch.exir import to_edge_transform_and_lower

    print(f'\n导出 {name}{"（XNNPACK）" if use_xnnpack else "（portable）"}...')
    exported = export(module, example_inputs, dynamic_shapes=dynamic_shapes, strict=False)
    partitioner = []
    if use_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioner = [XnnpackPartitioner()]
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=partitioner,
    )
    et_prog = edge.to_executorch()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(et_prog.buffer)
    size_kb = out_path.stat().st_size / 1024
    print(f'  已保存: {out_path}  ({size_kb:.0f} KB)')


# ── 主流程 ──────────────────────────────────────────────────────────

def main() -> None:
    try:
        import executorch  # noqa: F401
    except ImportError:
        print('请先安装 executorch: pip install executorch')
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',     required=True, help='best_epochXXX/model.pt 路径')
    ap.add_argument('--data-dir', required=True, help='mathwriting-2024 目录（含 vocab.json）')
    ap.add_argument('--out',      default='export_v3', help='输出目录')
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'加载 checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {})

    d_model      = ckpt_args.get('d_model',      512)
    n_layers     = ckpt_args.get('n_layers',     8)
    nhead        = ckpt_args.get('nhead',        8)
    max_tgt      = ckpt_args.get('max_tgt',      64)
    max_src      = ckpt_args.get('max_src',      512)
    n_stroke_tok = ckpt_args.get('n_stroke_tok', 32)
    n_img_tok    = 64
    n_prefix     = n_img_tok + n_stroke_tok       # 96
    max_seq      = ckpt['decoder']['pos_embed.weight'].shape[0]
    print(f'[Config] d_model={d_model} n_layers={n_layers} nhead={nhead}')
    print(f'[Config] n_prefix={n_prefix} max_seq={max_seq}')

    # 词表（必须用 vocab.json，不用 bpe_vocab.json）
    vocab_path = data_dir / 'vocab.json'
    if not vocab_path.exists():
        print(f'找不到 {vocab_path}')
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import Vocabulary
    from models import VisualPrefixEncoder, CausalTransformerDecoder

    vocab = Vocabulary.load(vocab_path)
    vocab_size = len(vocab)
    print(f'[Vocab] size={vocab_size}')

    # 重建模型
    prefix_enc = VisualPrefixEncoder(d_model=d_model, n_stroke=n_stroke_tok).eval()
    decoder    = CausalTransformerDecoder(
        vocab_size=vocab_size, d_model=d_model,
        nhead=nhead, n_layers=n_layers, max_seq=max_seq,
    ).eval()
    prefix_enc.load_state_dict(ckpt['prefix_enc'])
    decoder.load_state_dict(ckpt['decoder'])

    logit_bias = _build_logit_bias(vocab.idx2token)

    from torch.export import Dim
    IMG_H, IMG_W = 64, 256

    # ── 1. prefix_enc.pte ───────────────────────────────────────────
    # stroke_len 固定为 max_src，Dart 侧补零对齐；avg_pool2d kernel = max_src//n_stroke_tok
    assert max_src % n_stroke_tok == 0, f'max_src({max_src}) 必须整除 n_stroke_tok({n_stroke_tok})'
    enc_mod = PrefixEncModule(prefix_enc, stroke_len=max_src).eval()
    img_ex          = torch.zeros(1, 1, IMG_H, IMG_W, dtype=torch.float32)
    stroke_ex       = torch.zeros(1, max_src, 13,      dtype=torch.float32)
    stroke_real_len = torch.tensor([16], dtype=torch.int32)  # 任意合法示例值
    _export_pte(
        enc_mod,
        (img_ex, stroke_ex, stroke_real_len),
        dynamic_shapes={
            'img':             None,
            'stroke':          None,   # stroke 形状全静态
            'stroke_real_len': None,   # 标量，内容动态但形状静态
        },
        out_path=out_dir / 'prefix_enc.pte',
        name='prefix_enc',
        use_xnnpack=True,   # 关键：启用 XNNPACK delegate，端侧 5-10× 提速
    )

    # ── 2. decoder_step.pte ─────────────────────────────────────────
    dec_mod      = DecoderStepModule(decoder, logit_bias, n_prefix=n_prefix).eval()
    prefix_ex    = torch.zeros(1, n_prefix, d_model, dtype=torch.float32)
    token_ids_ex = torch.zeros(1, 4, dtype=torch.int32)
    _export_pte(
        dec_mod,
        (prefix_ex, token_ids_ex),
        dynamic_shapes={
            'prefix_embeds': None,
            'token_ids':     {1: Dim('tgt_len', min=1, max=max_tgt + 2)},
        },
        out_path=out_dir / 'decoder_step.pte',
        name='decoder_step',
    )

    # ── 3. vocab.json ───────────────────────────────────────────────
    out_vocab = out_dir / 'vocab.json'
    meta = {
        'tokens':  vocab.idx2token,
        'bos_idx': vocab.BOS,
        'eos_idx': vocab.EOS,
        'pad_idx': vocab.PAD,
    }
    with open(out_vocab, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {out_vocab}  ({vocab_size} tokens)')
    print('\n全部导出完成！')
    print(f'请将以下文件复制到 Flutter assets/models/：')
    print(f'  {out_dir}/prefix_enc.pte')
    print(f'  {out_dir}/decoder_step.pte')
    print(f'  {out_dir}/vocab.json')


if __name__ == '__main__':
    main()
