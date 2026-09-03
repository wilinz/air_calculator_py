"""在主机上跑一遍导出的 tflite，核对识别结果。

上手机装一次要几分钟，这里几秒钟就能知道导出是对是错——fp16 量化那轮踩的
坑（mask 常量 -1e9 在 fp16 里溢出成 -inf，取行的矩阵乘里 0×(-inf)=NaN）就是
先在这里暴露出来的，输出会从正常 LaTeX 变成整串 <PAD> 或同一个 token 无限
重复。

用法：
    python3 air_calculator_py/export/verify_tflite.py weights/exports/litert_stroke_v4_final
    python3 air_calculator_py/export/verify_tflite.py <目录> --samples 20

目录里要有 prefix_enc.tflite / decoder.tflite / vocab.json，即导出脚本的产物。
样本取自 app 打包的那份 benchmark_samples.jsonl，和端上跑的是同一批笔画。
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stroke_features import SequenceFeatureExtractor, FEATURE_DIM  # noqa: E402
from stroke_renderer import render_strokes, IMG_H, IMG_W  # noqa: E402

MAX_SRC = 512
SAMPLES = ROOT / 'air_calculator/assets/benchmark_samples.jsonl'


def _out_order(name):
    """输出名形如 output_0 / output_12，按数字排序而不是字典序。"""
    digits = ''.join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def build_inputs(traces):
    """把一条样本的笔画转成 prefix_enc 的三个输入。

    与 Rust 侧 mwh-decode 喂的完全一致：渲染图、13 维笔画特征、真实长度。
    stroke_only 的模型不看图像那一路，但接口仍是三个输入，所以照给。
    """
    strokes = [[(p[0], p[1]) for p in tr] for tr in traces]
    times = [[p[2] / 1000.0 for p in tr] for tr in traces]

    feat = SequenceFeatureExtractor(max_len=MAX_SRC).extract(strokes, times)
    real_len = min(len(feat), MAX_SRC)
    src = np.zeros((1, MAX_SRC, FEATURE_DIM), dtype=np.float32)
    src[0, :real_len] = feat[:real_len]

    img = render_strokes(strokes).astype(np.float32).reshape(1, 1, IMG_H, IMG_W)
    return img, src, np.array([real_len], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir', type=Path)
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--max-decode', type=int, default=32)
    args = ap.parse_args()

    from ai_edge_litert.interpreter import Interpreter

    vocab = json.loads((args.model_dir / 'vocab.json').read_text())
    tokens, bos, eos = vocab['tokens'], vocab['bos_idx'], vocab['eos_idx']

    enc = Interpreter(model_path=str(args.model_dir / 'prefix_enc.tflite'))
    dec = Interpreter(model_path=str(args.model_dir / 'decoder.tflite'))
    encode = enc.get_signature_runner()
    prefill = dec.get_signature_runner('prefill')
    step = dec.get_signature_runner('decode')

    lines = SAMPLES.read_text().splitlines()[: args.samples]
    hit = 0
    for line in lines:
        s = json.loads(line)
        img, src, real_len = build_inputs(s['traces'])
        prefix = encode(args_0=img, args_1=src, args_2=real_len)['output_0']
        # prefill 返回 2·n_layers 个 KV 张量，按输出名排序取出。
        kv_out = prefill(args_0=prefix)
        kv = [kv_out[k] for k in sorted(kv_out, key=_out_order)]

        n_prefix = prefix.shape[1]
        token = np.array([float(bos)], dtype=np.float32)
        out = []
        for i in range(args.max_decode):
            feed = {'args_0': token,
                    'args_1': np.array([float(n_prefix + i)], dtype=np.float32)}
            for j, t in enumerate(kv):
                feed[f'args_{j + 2}'] = t
            r = step(**feed)
            keys = sorted(r, key=_out_order)
            logits = r[keys[0]]
            kv = [r[k] for k in keys[1:]]
            nxt = int(np.argmax(logits[0]))
            if nxt == eos:
                break
            out.append(nxt)
            token = np.array([float(nxt)], dtype=np.float32)

        # 第一步喂的是 BOS，模型会把它原样吐回来，从第二个 token 起才是内容。
        pred = ''.join(tokens[i] for i in out if tokens[i] != '<BOS>')
        # 样本里的标注是 HTML 转义过的（&amp; / &lt;），先还原再比。
        label = html.unescape(s['normalized_label']).replace(' ', '')
        ok = pred.replace(' ', '') == label
        hit += ok
        print(f'{"✓" if ok else "✗"} 标注 {label!r}\n  预测 {pred!r}')

    print(f'\n{hit}/{len(lines)} 完全一致')


if __name__ == '__main__':
    main()
