#!/usr/bin/env python3
"""
裸 Core ML 导出（iOS）：不经 ExecuTorch，直接 torch.export → coremltools → .mlpackage。

现状是 PyTorch → ExecuTorch（CoreMLPartitioner）→ .pte，Core ML 只是 ExecuTorch
里的一个 delegate：图先被切成若干分区，能吃的交给 Core ML，剩下的留在 ExecuTorch
的 portable kernel 上，运行时还要驮着 ExecuTorch 那套。这份脚本把中间那层去掉，
整张图直接交给 Core ML，产出系统原生的 .mlpackage。

三个图与 ExecuTorch 那份完全一致（直接复用同一批包装模块，保证是同一个网络、
同一组权重，比较才有意义）：

  prefix_enc.mlpackage        img + stroke + real_len → prefix embeddings
  decoder_prefill.mlpackage   prefix embeds           → 初始 KV
  decoder_step.mlpackage      token + pos + past_kv   → logits + new KV

输入/输出名字显式钉住，Rust 侧的 coreml 后端按名字取，不依赖 coremltools 的
自动命名（它会随 torch 版本变）。

用法：
  cd air_calculator_py/export
  python3 export_coreml_ios.py \
    --ckpt ../../experiments/2026-09-01_stroke_only_full_pipeline/stroke_v4_best_ep040_acc0.7650.pt \
    --data-dir ../../dataset/mathwriting-2024 \
    --out ../../weights/exports/coreml_ios_v4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

# 与 ExecuTorch 那份共用包装模块：同一个网络、同一组权重，两条路才可比。
from export_et_hybrid_fp16 import (
    DecoderPrefillKVModule,
    DecoderTextStepKVModule,
    PrefixEncUnrolledModule,
    _build_logit_bias,
)


class FixedKvPrefill(torch.nn.Module):
    """prefill 的定长 KV 版：算完把 KV 补零到 [kv_max]。

    与 step 那边配套。补零的部分由 step 的掩码挡住，不参与注意力。
    """

    def __init__(self, inner: torch.nn.Module, kv_max: int):
        super().__init__()
        self.inner = inner
        self.kv_max = kv_max

    def forward(self, prefix):
        kv = self.inner(prefix)                       # (L, 2, 1, n_prefix, d)
        pad = self.kv_max - kv.shape[3]
        if pad <= 0:
            return kv
        return torch.nn.functional.pad(kv, (0, 0, 0, pad))


class FixedKvStep(torch.nn.Module):
    """单步 decode 的定长 KV 版：(token_id, pos, past_kv) → (logits, new_kv)，
    past_kv / new_kv 的形状恒为 (L, 2, 1, kv_max, d)。

    为什么要这一版：原版让 KV 沿序列维增长，导出到 Core ML 就是一个
    RangeDim。Core ML 对 flexible shape 是「遇到新形状再规划一次」，而自回归
    解码每一步的 kv_len 都不同，等于每步都在重规划——实测每步 52–61ms，而
    ExecuTorch 那条同样的图只要 5–7ms。

    定长之后有两处要绕开动态索引，否则 ANE 上又会退化：

      写入   不用 index_copy，用 one-hot 乘加：
             k_full = past_k * (1 - w) + k_new * w，w = (arange(kv_max) == pos)
      注意力 不靠裁剪长度，用加性掩码把 pos 之后的位置压到 -1e4。
             prefill 补的那段零正是靠它排除在外。
    """

    def __init__(self, inner: torch.nn.Module, kv_max: int, n_layers: int,
                 d_model: int, nhead: int):
        super().__init__()
        self.kv_max = kv_max
        self.n_layers = n_layers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.tok_embed = inner.tok_embed
        self.pos_embed = inner.pos_embed
        self.drop = inner.drop
        self.layer_list = inner.layer_list
        self.norm = inner.norm
        self.head = inner.head
        self.register_buffer('logit_bias', inner.logit_bias)
        self.register_buffer('kv_positions', torch.arange(kv_max, dtype=torch.float32))

    def _layer(self, x, past_k, past_v, layer, write, attn_mask):
        import torch.nn.functional as F

        x_norm = layer.norm1(x)
        qkv = F.linear(x_norm, layer.self_attn.in_proj_weight,
                       layer.self_attn.in_proj_bias)
        q, k_new, v_new = qkv.chunk(3, dim=-1)

        # one-hot 写入：k_new 是 (1,1,d)，靠广播落到 write 指定的那一槽。
        k_full = past_k * (1.0 - write) + k_new * write
        v_full = past_v * (1.0 - write) + v_new * write

        q_mh = q.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
        k_mh = k_full.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
        v_mh = v_full.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)

        out = F.scaled_dot_product_attention(q_mh, k_mh, v_mh, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(1, 1, self.d_model)
        out = F.linear(out, layer.self_attn.out_proj.weight,
                       layer.self_attn.out_proj.bias)
        x = x + out

        x_norm2 = layer.norm2(x)
        ff = layer.activation(F.linear(x_norm2, layer.linear1.weight, layer.linear1.bias))
        ff = F.linear(ff, layer.linear2.weight, layer.linear2.bias)
        return x + ff, k_full, v_full

    def forward(self, token_id, pos, past_kv):
        pos_idx = pos.long().view(1, 1)
        x = self.drop(self.tok_embed(token_id.long()) + self.pos_embed(pos_idx))

        p = pos.reshape(1)
        # (1, kv_max, 1)：这一步的 k/v 写到第 pos 槽
        write = (self.kv_positions == p).to(x.dtype).view(1, self.kv_max, 1)
        # (1, 1, 1, kv_max)：能看到 0..pos，其余压掉。-1e4 而不是 -inf——
        # fp16 下 -inf 参与 softmax 会出 NaN。
        allow = (self.kv_positions <= p).to(x.dtype)
        attn_mask = ((1.0 - allow) * -1e4).view(1, 1, 1, self.kv_max)

        new_kv_list = []
        for i, layer in enumerate(self.layer_list):
            past_k = past_kv[i, 0]
            past_v = past_kv[i, 1]
            x, k_full, v_full = self._layer(x, past_k, past_v, layer, write, attn_mask)
            new_kv_list.append(torch.stack([k_full, v_full], dim=0))
        new_kv = torch.stack(new_kv_list, dim=0)

        logits = self.head(self.norm(x[:, -1, :])) + self.logit_bias
        return logits, new_kv


def load_parts(ckpt_path: str, data_dir: str):
    """加载 ckpt，建出 prefix_enc / decoder 两个 eval 模块及形状参数。

    这段与 export_et_hybrid_fp16.main() 里的对应段落一致——那边写在 main 里，
    没有可复用的函数，照抄一份比重构那个能用的脚本稳妥。
    """
    from dataset import Vocabulary
    from models import CausalTransformerDecoder, VisualPrefixEncoder

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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

    vocab = Vocabulary.load(Path(data_dir) / 'vocab.json')
    prefix_enc = VisualPrefixEncoder(
        d_model=d_model, n_stroke=n_stroke_tok, modality=modality).eval()
    decoder = CausalTransformerDecoder(
        vocab_size=len(vocab), d_model=d_model, nhead=nhead,
        n_layers=n_layers, max_seq=max_seq,
    ).eval()
    prefix_enc.load_state_dict(ckpt['prefix_enc'])
    decoder.load_state_dict(ckpt['decoder'])

    return {
        'vocab': vocab,
        'prefix_enc': prefix_enc,
        'decoder': decoder,
        'logit_bias': _build_logit_bias(vocab.idx2token),
        'd_model': d_model,
        'n_layers': n_layers,
        'nhead': nhead,
        'max_src': max_src,
        'n_prefix': n_prefix,
        'modality': modality,
    }


def convert(ep, inputs, outputs, out_path: Path, name: str, precision):
    """ExportedProgram → .mlpackage。"""
    import coremltools as ct

    t0 = time.time()
    print(f'  [{name}] ct.convert …')
    # torch 2.11 的 torch.export 默认给出 TRAINING dialect，coremltools 只接
    # ATEN / EDGE；跑一遍分解降到 ATEN。ExecuTorch 那条路是它自己内部做的，
    # 直连时得自己来。
    ep = ep.run_decompositions({})
    model = ct.convert(
        ep,
        inputs=inputs,
        outputs=outputs,
        # ANE 只吃 fp16；fp32 会把整张图推到 GPU/CPU 上。
        compute_precision=precision,
        # 让系统在 ANE/GPU/CPU 之间自己调度，与 ExecuTorch 那份的
        # CoreMLBackend.generate_compile_specs(compute_unit=ALL) 对齐。
        compute_units=ct.ComputeUnit.ALL,
        # iOS17 起 mlprogram 才支持这里用到的动态形状与 fp16 IO。
        minimum_deployment_target=ct.target.iOS17,
        convert_to='mlprogram',
    )
    model.save(str(out_path))
    size = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file())
    print(f'  [{name}] saved → {out_path.name}  '
          f'({size / 1024 / 1024:.1f} MB, {time.time() - t0:.0f}s)')
    return size


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-decode', type=int, default=32)
    ap.add_argument('--fp32', action='store_true',
                    help='fp32 基线对比用；ANE 不吃 fp32，会落到 GPU/CPU')
    ap.add_argument('--split-decoder', action='store_true',
                    help='prefill / step 各出一个 .mlpackage（各带一份权重，'
                         '95.7MB）。缺省合并成多函数模型共享权重')
    args = ap.parse_args()

    import coremltools as ct
    from torch.export import export as texport

    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== 裸 Core ML 导出  precision={"fp32" if args.fp32 else "fp16"} '
          f'compute_units=ALL  target=iOS17 ===\n')

    p = load_parts(args.ckpt, args.data_dir)
    d_model, n_layers, n_prefix = p['d_model'], p['n_layers'], p['n_prefix']
    print(f'  [modality] {p["modality"]}  n_prefix={n_prefix}  '
          f'd_model={d_model}  n_layers={n_layers}')

    total = 0
    enc_size = 0

    # ── 1. prefix_enc ────────────────────────────────────────────────────────
    # 必须用展开版：stroke encoder 的 nn.TransformerEncoderLayer 里 in_proj_weight
    # 被双消费，会触发 Core ML fp16 lower 的图重写 bug（cos 0.55、端侧 EM 0%）。
    # 那个坑与走不走 ExecuTorch 无关，是 Core ML 这一侧的，所以这里照样绕开。
    enc_mod = PrefixEncUnrolledModule(p['prefix_enc'], stroke_len=p['max_src']).eval()
    enc_ex = (
        torch.zeros(1, 1, 64, 256),
        torch.zeros(1, p['max_src'], 13),
        torch.tensor([16], dtype=torch.int32),
    )
    enc_ep = texport(enc_mod, enc_ex, strict=False)
    enc_size = convert(
        enc_ep,
        inputs=[
            ct.TensorType(name='img', shape=enc_ex[0].shape),
            ct.TensorType(name='stroke', shape=enc_ex[1].shape),
            ct.TensorType(name='stroke_real_len', shape=enc_ex[2].shape,
                          dtype=np.int32),
        ],
        outputs=[ct.TensorType(name='prefix')],
        out_path=out_dir / 'prefix_enc.mlpackage',
        name='prefix_enc',
        precision=precision,
    )
    total += enc_size

    # KV 缓冲的定长。与 Rust 侧的 MAX_DECODE 对齐：n_prefix + max_decode + 2。
    kv_max = n_prefix + args.max_decode + 2

    # ── 2. decoder prefill ───────────────────────────────────────────────────
    # 输出补零到 kv_max，让整条解码链路的形状恒定，Rust 那边不用改。
    prefill_mod = FixedKvPrefill(
        DecoderPrefillKVModule(p['decoder'], n_prefix, n_layers, d_model, p['nhead']).eval(),
        kv_max,
    ).eval()
    prefill_ex = (torch.zeros(1, n_prefix, d_model),)
    prefill_ep = texport(prefill_mod, prefill_ex, strict=False)
    total += convert(
        prefill_ep,
        inputs=[ct.TensorType(name='prefix', shape=prefill_ex[0].shape)],
        outputs=[ct.TensorType(name='kv')],
        out_path=out_dir / 'decoder_prefill.mlpackage',
        name='decoder_prefill',
        precision=precision,
    )

    # ── 3. decoder step ──────────────────────────────────────────────────────
    # 形状全静态：KV 定长 kv_max，写入靠 one-hot、可见范围靠加性掩码。
    # 原先这里把 KV 的序列维声明成 RangeDim，Core ML 每步都要为新形状重新
    # 规划，实测每步 52–61ms（ExecuTorch 同一张图 5–7ms）。见 FixedKvStep。
    step_mod = FixedKvStep(
        DecoderTextStepKVModule(
            p['decoder'], p['logit_bias'], n_layers, d_model, p['nhead']).eval(),
        kv_max, n_layers, d_model, p['nhead'],
    ).eval()
    step_ex = (
        torch.zeros(1, 1, dtype=torch.float32),
        torch.tensor([float(n_prefix)], dtype=torch.float32),
        torch.zeros(n_layers, 2, 1, kv_max, d_model),
    )
    step_ep = texport(step_mod, step_ex, strict=False)
    total += convert(
        step_ep,
        inputs=[
            ct.TensorType(name='token_id', shape=step_ex[0].shape),
            ct.TensorType(name='pos', shape=step_ex[1].shape),
            ct.TensorType(name='past_kv', shape=step_ex[2].shape),
        ],
        outputs=[ct.TensorType(name='logits'), ct.TensorType(name='new_kv')],
        out_path=out_dir / 'decoder_step.mlpackage',
        name='decoder_step',
        precision=precision,
    )

    # ── 3.5 合并成多函数模型 ─────────────────────────────────────────────────
    #
    # prefill 与 step 是同一个 decoder 的两种用法，权重完全相同。拆成两个
    # .mlpackage 就各存一份（合计 95.7MB）；合并成一个多函数模型后 coremltools
    # 会做权重去重，落到 50MB 上下——与合并版 .pte 同一量级。
    #
    # 多函数模型要 iOS 18 / macOS 15 起。加载时由 MLModelConfiguration.functionName
    # 选函数，见 coreml_shim.m 的 cml_model_load。
    if not args.split_decoder:
        from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

        desc = MultiFunctionDescriptor()
        desc.add_function(str(out_dir / 'decoder_prefill.mlpackage'), 'main', 'prefill')
        desc.add_function(str(out_dir / 'decoder_step.mlpackage'), 'main', 'step')
        # 默认函数取 step：它是解码循环里跑几十次的那个。
        desc.default_function_name = 'step'
        merged = out_dir / 'decoder.mlpackage'
        if merged.exists():
            shutil.rmtree(merged)
        save_multifunction(desc, str(merged))
        for part in ('decoder_prefill.mlpackage', 'decoder_step.mlpackage'):
            shutil.rmtree(out_dir / part)
        msize = sum(f.stat().st_size for f in merged.rglob('*') if f.is_file())
        print(f'  [decoder] 合并 prefill + step → decoder.mlpackage '
              f'({msize / 1024 / 1024:.1f} MB，原先两份合计 '
              f'{total / 1024 / 1024:.1f} MB)')
        total = enc_size + msize

    # ── 4. vocab.json ────────────────────────────────────────────────────────
    vocab = p['vocab']
    (out_dir / 'vocab.json').write_text(
        json.dumps({
            'tokens': vocab.idx2token,
            'bos_idx': vocab.BOS,
            'eos_idx': vocab.EOS,
            'pad_idx': vocab.PAD,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(f'\n合计 {total / 1024 / 1024:.1f} MB → {out_dir}')
    print('下一步：python3 bench_coreml_vs_pte.py 比一下延迟')


if __name__ == '__main__':
    main()
