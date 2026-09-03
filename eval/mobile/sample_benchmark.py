#!/usr/bin/env python3
"""Sample 500 InkML files from validation set, stratified by sequence length.

输出 JSONL（每行一条 JSON 对象），包含：
- label: 原始 LaTeX 标签（<annotation type="label">）
- normalized_label: 规范化 LaTeX（<annotation type="normalizedLabel">），评测时优先用
- n_strokes / n_points
- traces: [[x,y,t], ...]
"""

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

VALID_DIR = Path(__file__).parent / "data/mathwriting-2024/valid"
OUTPUT = Path(__file__).parent.parent / "air_calculator/assets/benchmark_samples.jsonl"
SEED = 42
TOTAL = 500

# 分层：按总点数拆 4 个桶，均匀分配，覆盖短→长序列
# 这样各延迟区间都有足够样本做统计
BUCKETS = [
    ("<50 pts",   0,  50, 125),
    ("50-150",   50, 150, 125),
    ("150-350", 150, 350, 125),
    ("350+",    350, 99999, 125),
]


def parse_inkml(path: Path):
    """Parse an InkML file. Returns (label, normalized_label, [trace_points])."""
    text = path.read_text()

    m = re.search(r'<annotation type="label">(.+?)</annotation>', text, re.DOTALL)
    label = m.group(1) if m else ""

    m = re.search(r'<annotation type="normalizedLabel">(.+?)</annotation>', text, re.DOTALL)
    normalized_label = m.group(1) if m else ""

    traces = []
    for m in re.finditer(r'<trace[\s>][^>]*>(.+?)</trace>', text, re.DOTALL):
        points = []
        for token in m.group(1).strip().split(","):
            parts = token.strip().split()
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                t = float(parts[2]) if len(parts) > 2 else 0.0
                points.append((x, y, t))
        if points:
            traces.append(points)
    return label, normalized_label, traces


def main():
    random.seed(SEED)

    inkml_files = sorted(VALID_DIR.glob("*.inkml"))
    print(f"Found {len(inkml_files)} InkML files in {VALID_DIR}")

    # Collect metadata
    records = []
    for f in inkml_files:
        try:
            label, normalized_label, traces = parse_inkml(f)
            n_strokes = len(traces)
            n_points = sum(len(t) for t in traces)
            records.append({
                "file": f.name,
                "label": label,
                "normalized_label": normalized_label,
                "n_strokes": n_strokes,
                "n_points": n_points,
                "traces": traces,
            })
        except Exception as e:
            print(f"  skip {f.name}: {e}", file=sys.stderr)

    print(f"Parsed {len(records)} valid files")

    # 按总点数分层，均匀分配
    by_points = defaultdict(list)
    points_all = sorted(r["n_points"] for r in records)
    print(f"\nPoints distribution: min={points_all[0]} max={points_all[-1]} "
          f"p10={points_all[len(points_all)//10]} "
          f"p50={points_all[len(points_all)//2]} "
          f"p90={points_all[len(points_all)*9//10]}")

    for r in records:
        p = r["n_points"]
        for name, lo, hi, _ in BUCKETS:
            if lo <= p < hi:
                by_points[name].append(r)
                break

    print("\nPoints bucket distribution:")
    for name, lo, hi, alloc in BUCKETS:
        pool = by_points[name]
        print(f"  {name}: {len(pool)} total (alloc {alloc})")

    # 均匀采样
    sampled = []
    for name, lo, hi, alloc in BUCKETS:
        pool = by_points[name]
        n = min(alloc, len(pool))
        chosen = random.sample(pool, n)
        chosen.sort(key=lambda r: r["n_points"])
        sampled.extend(chosen)
        print(f"  {name}: sampled {len(chosen)}, "
              f"points range {chosen[0]['n_points']}-{chosen[-1]['n_points']}")

    # 不足 TOTAL：从富余桶补齐
    shortfall = TOTAL - len(sampled)
    if shortfall > 0:
        sampled_files = {r["file"] for r in sampled}
        for name, _lo, _hi, _alloc in BUCKETS:
            if shortfall <= 0:
                break
            available = [r for r in by_points[name] if r["file"] not in sampled_files]
            extra = random.sample(available, min(shortfall, len(available)))
            sampled.extend(extra)
            shortfall -= len(extra)
            print(f"  +{len(extra)} extra from {name}")

    # Shuffle sampled
    random.shuffle(sampled)

    # 输出 JSONL：每行一条紧凑 JSON，主端 isolate 可逐行解析，避免一次性 jsonDecode
    output = []
    for r in sampled:
        output.append({
            "label": r["label"],
            "normalized_label": r["normalized_label"],
            "n_strokes": r["n_strokes"],
            "n_points": r["n_points"],
            "traces": r["traces"],
        })

    os.makedirs(OUTPUT.parent, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    # Stats
    total_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\nWritten {len(output)} samples to {OUTPUT} ({total_mb:.1f} MB)")

    # Summary stats
    lens = [r["n_strokes"] for r in output]
    pts = [r["n_points"] for r in output]
    print(f"Stroke count: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")
    print(f"Points:      min={min(pts)} max={max(pts)} mean={sum(pts)/len(pts):.1f}")


if __name__ == "__main__":
    main()
