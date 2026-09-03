#!/usr/bin/env python3
"""
裸 Core ML（.mlpackage）与 ExecuTorch+CoreML delegate（.pte）的对照。

两件事分开看：

  正确性  两条路的输出各自与 PyTorch fp32 比余弦相似度。Core ML fp16 lower
          出过图重写 bug（prefix_enc 那次 cos 掉到 0.55、端侧 EM 0%），所以
          延迟再好看，先过这一关才有意义。

  延迟    prefix_enc / prefill / step 各测一遍。step 是自回归循环里跑几十次
          的那个，权重最重，它才是端到端时延的大头。

跑在 Mac 上：Apple Silicon 的 ANE 与 iPhone 同源但不同代，绝对值不能直接搬到
手机上，两条路的**相对**关系可以。

用法：
  cd air_calculator_py/export
  python3 bench_coreml_vs_pte.py \
    --coreml ../../weights/exports/coreml_ios_v4 \
    --pte ../../air_calculator/platform_models/ios \
    --ckpt ../../experiments/2026-09-01_stroke_only_full_pipeline/stroke_v4_best_ep040_acc0.7650.pt \
    --data-dir ../../dataset/mathwriting-2024
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from export_coreml_ios import load_parts
from export_et_hybrid_fp16 import (
    DecoderPrefillKVModule,
    DecoderTextStepKVModule,
    PrefixEncUnrolledModule,
)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('nan')
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float('nan')


def stats(name: str, x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float64)
    bad = int((~np.isfinite(x)).sum())
    return (f'{name}: shape={x.shape} min={np.nanmin(x):.3g} max={np.nanmax(x):.3g}'
            + (f'  非有限值 {bad} 个' if bad else ''))


def timeit(fn, warmup: int, iters: int) -> tuple[float, float]:
    """返回（中位数毫秒, 最快毫秒）。取中位数而不是均值：ANE 上偶发的调度
    抖动会把均值拉偏，而我们关心的是常态。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coreml', required=True, help='.mlpackage 所在目录')
    ap.add_argument('--pte', default=None, help='.pte 所在目录；不给则只测 Core ML')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    args = ap.parse_args()

    import coremltools as ct

    p = load_parts(args.ckpt, args.data_dir)
    d_model, n_layers, n_prefix = p['d_model'], p['n_layers'], p['n_prefix']
    max_src = p['max_src']

    # ── 输入：固定随机种子，三条路喂同一份 ──────────────────────────────
    rng = np.random.default_rng(0)
    img = np.zeros((1, 1, 64, 256), dtype=np.float32)
    stroke = rng.standard_normal((1, max_src, 13)).astype(np.float32)
    real_len = np.array([64], dtype=np.int32)

    # ── PyTorch fp32 基准 ───────────────────────────────────────────────
    enc_mod = PrefixEncUnrolledModule(p['prefix_enc'], stroke_len=max_src).eval()
    prefill_mod = DecoderPrefillKVModule(
        p['decoder'], n_prefix, n_layers, d_model, p['nhead']).eval()
    step_mod = DecoderTextStepKVModule(
        p['decoder'], p['logit_bias'], n_layers, d_model, p['nhead']).eval()

    with torch.no_grad():
        t_prefix = enc_mod(torch.from_numpy(img), torch.from_numpy(stroke),
                           torch.from_numpy(real_len))
        t_kv = prefill_mod(t_prefix)
        t_logits, t_newkv = step_mod(
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([n_prefix], dtype=torch.int32),
            t_kv,
        )
    ref = {
        'prefix': t_prefix.numpy(),
        'kv': t_kv.numpy(),
        'logits': t_logits.numpy(),
    }
    print(f'\nPyTorch fp32 基准就绪  prefix={ref["prefix"].shape}  '
          f'kv={ref["kv"].shape}  logits={ref["logits"].shape}')

    results = {}

    # ── Core ML ─────────────────────────────────────────────────────────
    print('\n─── 裸 Core ML (.mlpackage) ───')
    cml_dir = Path(args.coreml)
    m_enc = ct.models.MLModel(str(cml_dir / 'prefix_enc.mlpackage'))
    m_pre = ct.models.MLModel(str(cml_dir / 'decoder_prefill.mlpackage'))
    m_step = ct.models.MLModel(str(cml_dir / 'decoder_step.mlpackage'))

    def c_enc():
        return m_enc.predict({'img': img, 'stroke': stroke,
                              'stroke_real_len': real_len})

    o = c_enc()
    c_prefix = o[next(iter(o))]
    def c_pre():
        return m_pre.predict({'prefix': c_prefix.astype(np.float32)})
    o = c_pre()
    c_kv = o[next(iter(o))]

    tok = np.zeros((1, 1), dtype=np.int32)
    pos = np.array([n_prefix], dtype=np.int32)
    def c_step():
        return m_step.predict({'token_id': tok, 'pos': pos,
                               'past_kv': c_kv.astype(np.float32)})
    o = c_step()
    # 输出名按导出时钉的来取；万一 coremltools 改了名，退回按形状认。
    c_logits = o.get('logits')
    if c_logits is None:
        c_logits = next(v for v in o.values() if v.size == ref['logits'].size)

    print('  ' + stats('prefix', c_prefix))
    print('  ' + stats('kv', c_kv))
    print('  ' + stats('logits', c_logits))
    results['coreml'] = {
        'cos_prefix': cos(c_prefix, ref['prefix']),
        'cos_kv': cos(c_kv, ref['kv']),
        'cos_logits': cos(c_logits, ref['logits']),
        'enc': timeit(c_enc, args.warmup, args.iters),
        'pre': timeit(c_pre, args.warmup, args.iters),
        'step': timeit(c_step, args.warmup, args.iters),
    }

    # ── ExecuTorch ──────────────────────────────────────────────────────
    if args.pte:
        print('─── ExecuTorch + Core ML delegate (.pte) ───')
        from executorch.runtime import Runtime

        rt = Runtime.get()
        pte_dir = Path(args.pte)
        prog_enc = rt.load_program(pte_dir / 'prefix_enc.pte')
        e_enc_m = prog_enc.load_method('forward')

        # 合并版 decoder.pte 是多函数 Core ML 模型，本机装的 executorch wheel
        # 按旧 SDK 编（要 macOS 15+ SDK 才能 init），加载会失败；退回分开的
        # 两个单方法 .pte——同一批权重，延迟可比。
        merged = pte_dir / 'decoder.pte'
        e_pre_m = e_step_m = None
        if merged.exists():
            try:
                prog_dec = rt.load_program(merged)
                e_pre_m = prog_dec.load_method('prefill')
                e_step_m = prog_dec.load_method('step')
                print('  decoder.pte（多函数）加载成功')
            except Exception as exc:
                print(f'  decoder.pte 多函数加载失败（{exc}），退回单方法 .pte')
                e_pre_m = e_step_m = None
        if e_pre_m is None:
            e_pre_m = rt.load_program(pte_dir / 'decoder_prefill_kv.pte').load_method('forward')
            e_step_m = rt.load_program(pte_dir / 'decoder_step_kv.pte').load_method('forward')

        ti = (torch.from_numpy(img), torch.from_numpy(stroke), torch.from_numpy(real_len))
        def e_enc():
            return e_enc_m.execute(ti)
        e_prefix = e_enc()[0]

        def e_pre():
            return e_pre_m.execute((e_prefix,))
        e_kv = e_pre()[0]

        step_in = (torch.zeros(1, 1, dtype=torch.int32),
                   torch.tensor([n_prefix], dtype=torch.int32), e_kv)
        def e_step():
            return e_step_m.execute(step_in)
        e_out = e_step()

        results['executorch'] = {
            'cos_prefix': cos(e_prefix.numpy(), ref['prefix']),
            'cos_kv': cos(e_kv.numpy(), ref['kv']),
            'cos_logits': cos(e_out[0].numpy(), ref['logits']),
            'enc': timeit(e_enc, args.warmup, args.iters),
            'pre': timeit(e_pre, args.warmup, args.iters),
            'step': timeit(e_step, args.warmup, args.iters),
        }

    # ── 报告 ────────────────────────────────────────────────────────────
    print('\n' + '=' * 68)
    print(f'{"":14s}{"prefix_enc":>17s}{"prefill":>17s}{"step":>17s}')
    print('-' * 68)
    for name, r in results.items():
        print(f'{name:14s}' + ''.join(
            f'{r[k][0]:11.2f}ms{"":>4s}' for k in ('enc', 'pre', 'step')))
    print('-' * 68)
    print('中位数；括号内为最快一次')
    for name, r in results.items():
        print(f'  {name:12s}' + '  '.join(
            f'{k}={r[k][0]:.2f} (min {r[k][1]:.2f})' for k in ('enc', 'pre', 'step')))
    print('\n与 PyTorch fp32 的余弦相似度（越接近 1 越好，prefix 低于 0.99 要当心）')
    for name, r in results.items():
        print(f'  {name:12s} prefix={r["cos_prefix"]:.4f}  '
              f'kv={r["cos_kv"]:.4f}  logits={r["cos_logits"]:.4f}')
    if len(results) == 2:
        c, e = results['coreml'], results['executorch']
        print('\n裸 Core ML 相对 ExecuTorch：')
        for k, label in (('enc', 'prefix_enc'), ('pre', 'prefill'), ('step', 'step')):
            ratio = e[k][0] / c[k][0]
            verdict = f'快 {ratio:.2f}×' if ratio > 1 else f'慢 {1 / ratio:.2f}×'
            print(f'  {label:12s} {verdict}')


if __name__ == '__main__':
    main()
