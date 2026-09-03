#!/usr/bin/env python3
"""
将 inkml 目录批量转换为 parquet 文件，加速数据集加载。

输出：对于目录 /path/to/train，生成 /path/to/train.parquet
Schema：
  label       : string   (normalizedLabel)
  strokes_pkl : binary   (pickle.dumps(list[np.ndarray shape (N,3)]))

用法：
  python3 build_parquet.py /path/to/train /path/to/synthetic_v2 ...
  python3 build_parquet.py --all --data-dir /root/data/mathwriting-2024
"""
import argparse
import os
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NS = {'ink': 'http://www.w3.org/2003/InkML'}
BATCH = 4096   # 每批写入行数


def parse_inkml(path: Path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        label = ''
        for ann in root.findall('ink:annotation', NS):
            if ann.get('type', '') == 'normalizedLabel':
                label = (ann.text or '').strip()
                break
        if not label:
            return None
        strokes = []
        for trace in root.findall('ink:trace', NS):
            text = (trace.text or '').strip()
            if not text:
                continue
            pts = []
            for token in text.split(','):
                vals = token.strip().split()
                if len(vals) >= 3:
                    pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
            if pts:
                strokes.append(np.array(pts, dtype=np.float32))
        if not strokes:
            return None
        return label, strokes
    except Exception:
        return None


def _parse_worker(path_str):
    return parse_inkml(Path(path_str))


def convert_dir(inkml_dir: Path, out_path: Path, workers: int = None):
    files = sorted(inkml_dir.glob('*.inkml'))
    if not files:
        print(f'[skip] {inkml_dir} — 没有 inkml 文件', flush=True)
        return

    print(f'[Convert] {inkml_dir.name}/ ({len(files):,} 文件) → {out_path.name}', flush=True)

    workers = workers or min(cpu_count(), 8)
    schema = pa.schema([
        ('label', pa.string()),
        ('strokes_pkl', pa.binary()),
    ])

    writer = pq.ParquetWriter(str(out_path), schema, compression='snappy')
    ok = skip = 0
    labels_buf = []
    strokes_buf = []

    def _flush():
        nonlocal ok
        if not labels_buf:
            return
        table = pa.table({
            'label': pa.array(labels_buf, type=pa.string()),
            'strokes_pkl': pa.array(strokes_buf, type=pa.binary()),
        }, schema=schema)
        writer.write_table(table)
        ok += len(labels_buf)
        labels_buf.clear()
        strokes_buf.clear()

    path_strs = [str(p) for p in files]
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap(_parse_worker, path_strs, chunksize=64)):
            if result is None:
                skip += 1
            else:
                label, strokes = result
                labels_buf.append(label)
                strokes_buf.append(pickle.dumps(strokes))
            if len(labels_buf) >= BATCH:
                _flush()
            if (i + 1) % 50000 == 0:
                print(f'  {i+1:,}/{len(files):,}  ok={ok+len(labels_buf):,}', flush=True)

    _flush()
    writer.close()
    print(f'[Convert] 完成: ok={ok:,} skip={skip:,} → {out_path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='*', help='要转换的 inkml 目录')
    ap.add_argument('--data-dir', default=None, help='批量转换该目录下所有子目录')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--subdirs', nargs='*',
                    default=['train', 'valid', 'test', 'synthetic', 'synthetic_v2',
                             'hard_cases_v1', 'hard_cases_v2', 'hard_cases_v3',
                             'latex_formulas_v1'],
                    help='配合 --data-dir 指定子目录名')
    args = ap.parse_args()

    targets = []

    # 从 --data-dir 收集
    if args.data_dir:
        data_dir = Path(args.data_dir)
        for name in args.subdirs:
            d = data_dir / name
            if d.exists():
                targets.append(d)

    # 从位置参数收集
    for d in args.dirs:
        targets.append(Path(d))

    if not targets:
        print('没有找到要转换的目录，请指定 --data-dir 或直接传目录路径', flush=True)
        sys.exit(1)

    for d in targets:
        if not d.exists():
            print(f'[skip] 不存在: {d}', flush=True)
            continue
        out = d.parent / f'{d.name}.parquet'
        if out.exists():
            print(f'[skip] 已存在: {out}', flush=True)
            continue
        convert_dir(d, out, args.workers)


if __name__ == '__main__':
    main()
