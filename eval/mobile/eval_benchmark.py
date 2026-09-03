"""离线评测端侧基准导出的 results.jsonl。

输入：results.jsonl（来自 Flutter benchmark 页面 → Export 按钮）
每行一条 JSON：{label, pred, token_ids, encMs, prefillMs, decStepAvgMs, totalMs, tokenCount, eval_mode}

报告两个指标：
- ExpRate (Exact Match Rate): pred 与 label 严格相等的比例。同时报告归一化后版本（去空白 / 去 \\, / 去 {}）。
- char-level CER: 1 - mean(LevenshteinDist(pred, label) / max(1, len(label)))。
  公式识别学界更常报告 token-level CER，但这要求把 label 重新切分回 vocab token，复杂且易出错；
  这里用 char-CER 作为简单可复现的基线指标，eval_mode 切换前后对比足够说明问题。

用法：
    python eval_benchmark.py /path/to/results.jsonl
    python eval_benchmark.py results_eval.jsonl results_normal.jsonl  # 多文件对比
"""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean


def levenshtein(a: str, b: str) -> int:
    """标准 Levenshtein 距离，O(len(a)*len(b)) 时空。"""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


_NORMALIZE_PATTERNS = [
    (re.compile(r"\\,|\\;|\\!|\\:|\\ "), ""),  # 显式空格命令
    (re.compile(r"\s+"), ""),                  # 普通空白
    (re.compile(r"\{([^{}])\}"), r"\1"),      # 单字符 {x} → x（仅一遍，足够覆盖 a^{2} 这种）
]


def normalize(s: str) -> str:
    out = s
    for pat, rep in _NORMALIZE_PATTERNS:
        out = pat.sub(rep, out)
    return out


def evaluate(path: Path) -> dict:
    rows = []
    meta = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 端侧导出第一行带 "__meta__":true，作为设备/模型信息 header
            if obj.get("__meta__"):
                meta = obj
                continue
            rows.append(obj)

    n = len(rows)
    if n == 0:
        return {"path": str(path), "n": 0, **{f"meta.{k}": v for k, v in meta.items() if k != "__meta__"}}

    exp_strict = 0
    exp_norm = 0
    cer_vals = []
    cer_norm_vals = []
    enc_ms = []
    total_ms = []
    eval_modes = set()

    for r in rows:
        label = r.get("label", "")
        pred = r.get("pred", "")
        eval_modes.add(r.get("eval_mode"))

        if pred == label:
            exp_strict += 1

        nl, np_ = normalize(label), normalize(pred)
        if nl == np_:
            exp_norm += 1

        denom = max(1, len(label))
        cer_vals.append(levenshtein(pred, label) / denom)
        denom_n = max(1, len(nl))
        cer_norm_vals.append(levenshtein(np_, nl) / denom_n)

        enc_ms.append(r.get("encMs", 0))
        total_ms.append(r.get("totalMs", 0))

    dev = meta.get("device", {}) if isinstance(meta.get("device"), dict) else {}
    # 兼容两种 meta 形态：旧版扁平、新版 device:{...}
    if not dev and "platform" in meta:
        dev = meta

    if dev.get("platform") == "ios":
        device_label = f"iOS {dev.get('system_version','?')} {dev.get('machine','')} ({dev.get('name','')})".strip()
    elif dev.get("platform") == "android":
        device_label = f"Android {dev.get('release','?')} sdk={dev.get('sdk_int','?')} {dev.get('manufacturer','')} {dev.get('model','')} abi={dev.get('abi','')}".strip()
    else:
        device_label = f"{dev.get('platform','?')} {dev.get('os_version','')}".strip()

    mem = meta.get("memory", {}) or {}

    return {
        "path": str(path),
        "n": n,
        "device": device_label,
        "build_mode": meta.get("build_mode", ""),
        "app_version": meta.get("app_version", ""),
        "timestamp": meta.get("timestamp", ""),
        "eval_mode": eval_modes.pop() if len(eval_modes) == 1 else f"mixed:{eval_modes}",
        "exp_rate_strict": exp_strict / n,
        "exp_rate_normalized": exp_norm / n,
        "cer_char": mean(cer_vals),
        "cer_char_normalized": mean(cer_norm_vals),
        "avg_enc_ms": mean(enc_ms),
        "avg_total_ms": mean(total_ms),
        "rss_peak_mb": round(mem.get("rss_peak_mb", 0), 1),
        "rss_before_mb": round(mem.get("rss_before_mb", 0), 1),
        "total_mem_mb": round(mem.get("total_mb", 0), 0),
        "available_mem_mb": round(mem.get("available_mb", 0), 0),
        "num_processors": mem.get("num_processors", dev.get("num_processors", 0)),
    }


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path, help="results.jsonl 文件，可多个")
    args = ap.parse_args()

    reports = [evaluate(p) for p in args.paths]

    cols = [
        "path", "n", "device", "build_mode", "app_version", "timestamp",
        "eval_mode",
        "exp_rate_strict", "exp_rate_normalized",
        "cer_char", "cer_char_normalized",
        "avg_enc_ms", "avg_total_ms",
        "num_processors", "total_mem_mb", "available_mem_mb",
        "rss_peak_mb", "rss_before_mb",
    ]

    # 单文件 → 纵向输出；多文件 → 表格便于对比
    if len(reports) == 1:
        for k in cols:
            print(f"{k:>22}: {fmt(reports[0].get(k))}")
    else:
        widths = {c: max(len(c), max(len(fmt(r.get(c))) for r in reports)) for c in cols}
        header = " | ".join(c.ljust(widths[c]) for c in cols)
        sep = "-+-".join("-" * widths[c] for c in cols)
        print(header)
        print(sep)
        for r in reports:
            print(" | ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    sys.exit(main())
