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
一步管线：LaTeX 公式 → InkML 手写笔迹

将 latex_to_bboxes.py + synth_from_bboxes.py 合并为单一入口。

用法（批量）:
  python3 synth/latex_to_inkml.py \
    --input data/generated_latex/round1.jsonl \
    --out-dir output/targeted_v1 \
    --augment 1 \
    --error-types all

用法（交互测试）:
  python3 synth/latex_to_inkml.py --interactive
  python3 synth/latex_to_inkml.py --interactive --formula "x^2 + \\frac{1}{2}"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from PIL import Image as PILImage, ImageDraw
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_STROKE_COLORS = [
    '#E63946', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#795548', '#607D8B', '#F44336', '#3F51B5',
]


def render_strokes(strokes: list, height: int = 200, width: int = 800,
                   bg: str = '#FAFAFA') -> 'PILImage.Image | None':
    """将笔画列表渲染为 PIL Image。需要 Pillow。"""
    if not _PIL_OK:
        return None
    all_pts = [(float(p[0]), float(p[1])) for s in strokes for p in s]
    if not all_pts:
        return None
    xs, ys = zip(*all_pts)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin = 20
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    base_scale = min((width - 2 * margin) / x_range, (height - 2 * margin) / y_range)
    x_scale = min((width  - 2 * margin) / x_range, base_scale * 2.5)
    y_scale = min((height - 2 * margin) / y_range, base_scale * 2.5)
    x_off = (width  - x_range * x_scale) / 2
    y_off = (height - y_range * y_scale) / 2

    def px(x, y):
        return (int((x - x_min) * x_scale + x_off),
                int((y - y_min) * y_scale + y_off))

    img  = PILImage.new('RGB', (width, height), bg)
    draw = ImageDraw.Draw(img)
    lw   = max(3, height // 40)

    for i, s in enumerate(strokes):
        color = _STROKE_COLORS[i % len(_STROKE_COLORS)]
        pts = [px(float(p[0]), float(p[1])) for p in s]
        xs_s = [p[0] for p in pts]
        ys_s = [p[1] for p in pts]
        ext = max(max(xs_s) - min(xs_s), max(ys_s) - min(ys_s)) if pts else 0
        if len(pts) == 1 or ext < lw * 2:
            xc = sum(xs_s) // len(xs_s)
            yc = sum(ys_s) // len(ys_s)
            r = lw // 2
            draw.ellipse([xc - r, yc - r, xc + r, yc + r], fill=color)
        else:
            draw.line(pts, fill=color, width=lw, joint='curve')
    return img

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent.parent.parent / 'train'))

from latex_to_bboxes import formula_to_bboxes, formulas_to_bboxes  # noqa: E402
from synth_from_bboxes import (  # noqa: E402
    build_symbol_index, synthesize_formula, write_inkml,
    passes_filter, ERROR_TYPE_TOKENS,
    DATA_DIR, ASSETS_DIR,
)


def _run_batch(args):
    items = []
    with open(args.input) as f:
        for line in f:
            items.append(json.loads(line))
    print(f"[synth] 读入 {len(items)} 条公式")

    # ── Step 1: bbox（分批处理，避免单次 lualatex 超时）──
    formulas = [it.get('formula', '') for it in items]
    print("[synth] Step 1: 提取 bbox（lualatex）…")
    BBOX_BATCH = 500
    bbox_map: dict = {}
    for batch_start in range(0, len(formulas), BBOX_BATCH):
        batch = formulas[batch_start: batch_start + BBOX_BATCH]
        with tempfile.TemporaryDirectory() as tmpdir:
            partial = formulas_to_bboxes(batch, workdir=Path(tmpdir))
        for local_idx, bboxes in partial.items():
            bbox_map[batch_start + local_idx] = bboxes
        done = batch_start + len(batch)
        print(f"  [{done}/{len(formulas)}] 已解析 {len(bbox_map)} 条", flush=True)

    records = []
    for i, item in enumerate(items):
        bboxes = bbox_map.get(i)
        if not bboxes:
            continue
        records.append({
            'label': item.get('formula', ''),
            'normalizedLabel': item.get('formula', ''),
            'error_type': item.get('error_type', 'unknown'),
            'bboxes': bboxes,
        })
    print(f"[synth] bbox 解析成功: {len(records)} / {len(items)}")

    # ── 过滤 ──
    merged_tokens: set = set()
    for et in args.error_types:
        cfg = ERROR_TYPE_TOKENS.get(et, {})
        merged_tokens.update(cfg.get('include_any') or [])
    combined_filter = {
        'include_any': list(merged_tokens) if merged_tokens else None,
        'min_tokens': max(
            (ERROR_TYPE_TOKENS.get(et, {}).get('min_tokens', 3) for et in args.error_types),
            default=3,
        ),
        'max_tokens': 40,
    }
    records = [r for r in records if passes_filter(r, combined_filter)]
    print(f"[synth] 过滤后: {len(records)} 条")
    if not records:
        print("无匹配，退出")
        return

    # ── Step 2: synth ──
    print("[synth] Step 2: 合成 InkML…")
    sym_index = build_symbol_index(Path(args.data_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    generated = skipped = 0

    rng.shuffle(records)
    for rec in records:
        if generated >= args.n:
            break
        for aug_i in range(args.augment):
            if generated >= args.n:
                break
            strokes = synthesize_formula(
                rec, sym_index, rng, rng_np,
                local_noise=args.local_noise,
                global_noise=args.global_noise,
                affine_strength=args.affine_strength,
            )
            if strokes is None:
                skipped += 1
                continue
            fname = f"{generated:06d}.inkml"
            write_inkml(strokes, rec['label'], rec['normalizedLabel'],
                        out_dir / fname, f"synth_{generated:06d}")
            generated += 1

    print(f"[synth] 完成: {generated} 个 inkml，跳过 {skipped} 条")
    print(f"[synth] 输出: {out_dir}")


def _run_interactive(formula: str | None, args):
    """交互模式：从终端输入 LaTeX，实时生成 inkml 并打印摘要。"""
    sym_index = build_symbol_index(Path(args.data_dir))
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)

    def _process_one(fml: str):
        print(f"\n  → bbox…", end=' ', flush=True)
        bboxes = formula_to_bboxes(fml)
        if not bboxes:
            print("FAILED (lualatex 解析失败)")
            return
        tokens = [b['token'] for b in bboxes]
        print(f"ok  ({len(tokens)} tokens: {' '.join(tokens[:12])}{'…' if len(tokens)>12 else ''})")

        rec = {
            'label': fml,
            'normalizedLabel': fml,
            'error_type': 'interactive',
            'bboxes': bboxes,
        }
        strokes = synthesize_formula(
            rec, sym_index, rng, rng_np,
            local_noise=args.local_noise,
            global_noise=args.global_noise,
            affine_strength=args.affine_strength,
        )
        if strokes is None:
            print("  → synth FAILED (符号库覆盖不足)")
            return

        import hashlib, time
        uid = hashlib.md5(f"{fml}{time.time()}".encode()).hexdigest()[:8]
        stem = f"interactive_{uid}"
        out_path = out_dir / f"{stem}.inkml"
        write_inkml(strokes, fml, fml, out_path, stem)

        total_pts = sum(len(s) for s in strokes)
        print(f"  → synth ok  {len(strokes)} strokes  {total_pts} pts")
        print(f"  → inkml: {out_path}")

        # 生成预览图
        img = render_strokes(strokes, height=200, width=800)
        if img is not None:
            img_path = out_dir / f"{stem}.png"
            img.save(img_path)
            print(f"  → 预览: {img_path}")
        else:
            print("  → (安装 pillow 可生成预览图: pip install pillow)")

    if formula:
        _process_one(formula)
        return

    print("=" * 60)
    print("  LaTeX → InkML 交互测试")
    print("  输入 LaTeX 公式（不含 $ ），回车生成；输入 q 退出")
    print("=" * 60)
    while True:
        try:
            fml = input("\nLaTeX> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not fml or fml.lower() in ('q', 'quit', 'exit'):
            break
        _process_one(fml)


def main():
    ap = argparse.ArgumentParser(
        description="LaTeX → InkML 一步合成管线（含交互测试模式）"
    )
    ap.add_argument('--interactive', action='store_true',
                    help='启动交互模式，从终端逐条输入公式')
    ap.add_argument('--formula', default=None,
                    help='单条公式（与 --interactive 配合使用，不加则进入交互循环）')

    # 批量模式参数
    ap.add_argument('--input', default=None,
                    help='批量模式：输入 jsonl（{formula, error_type}）')
    ap.add_argument('--out-dir', default=None,
                    help='输出目录（批量 / 交互模式均可指定）')
    ap.add_argument('--augment', type=int, default=1,
                    help='每条公式增强份数（默认 1）')
    ap.add_argument('--n', type=int, default=999999,
                    help='最多生成数量（批量模式，默认不限）')
    ap.add_argument('--error-types', nargs='+', default=['all'],
                    choices=list(ERROR_TYPE_TOKENS.keys()),
                    help='错误类型过滤（默认 all）')
    ap.add_argument('--data-dir', default=str(DATA_DIR))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--local-noise', type=float, default=0.02)
    ap.add_argument('--global-noise', type=float, default=0.005)
    ap.add_argument('--affine-strength', type=float, default=0.06)
    args = ap.parse_args()

    if args.interactive or args.formula:
        _run_interactive(args.formula, args)
    elif args.input:
        if not args.out_dir:
            ap.error("批量模式需要 --out-dir")
        _run_batch(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
