#!/usr/bin/env python3
"""
从 inkml 目录提取 normalizedLabel，输出为 synth/latex_to_inkml.py 可用的 jsonl。

用法:
  python3 extract_labels.py \
    --inkml-dir /root/data/mathwriting-2024/train \
    --out /root/data/train_labels.jsonl
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {'ink': 'http://www.w3.org/2003/InkML'}


def parse_label(path: Path) -> str:
    try:
        root = ET.parse(path).getroot()
        for ann in root.findall('ink:annotation', NS):
            if ann.get('type', '') == 'normalizedLabel':
                return (ann.text or '').strip()
    except Exception:
        pass
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inkml-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--augment', type=int, default=1,
                    help='每条公式重复几次（默认1，latex_to_inkml.py 自带 augment 所以这里一般不需要）')
    args = ap.parse_args()

    inkml_dir = Path(args.inkml_dir)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(inkml_dir.glob('*.inkml'))
    print(f'[Extract] {len(files):,} inkml 文件')

    ok = skip = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(files):
            label = parse_label(p)
            if not label:
                skip += 1
                continue
            for _ in range(args.augment):
                f.write(json.dumps({'formula': label}, ensure_ascii=False) + '\n')
            ok += 1
            if (i + 1) % 50000 == 0:
                print(f'  {i+1:,}/{len(files):,}  ok={ok:,}')

    print(f'[Extract] 完成: ok={ok:,} skip={skip:,} → {out_path}')
    print(f'[Extract] 总行数: {ok * args.augment:,}')


if __name__ == '__main__':
    main()
