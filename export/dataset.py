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

"""MathWriting 数据集加载器（v2，BPE tokenizer 支持，已移除难样本过采样）"""

import json
import os
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stroke_features import SequenceFeatureExtractor, FEATURE_DIM
from stroke_renderer import render_strokes, IMG_H, IMG_W

DATA_DIR = Path(os.environ.get(
    'MATHWRITING_DIR',
    str(Path(__file__).resolve().parent.parent.parent / 'dataset' / 'mathwriting-2024')))
NS = {'ink': 'http://www.w3.org/2003/InkML'}
_LATEX_TOKEN_RE = re.compile(r'\\[a-zA-Z]+|\\.|\S')


# ── InkML 解析 ───────────────────────────────────────────────────── #

def parse_inkml(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    norm = ''
    for ann in root.findall('ink:annotation', NS):
        if ann.get('type', '') == 'normalizedLabel':
            norm = (ann.text or '').strip()
            break
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
    return norm, strokes


_BARE_SUB_RE  = re.compile(r'([_^])([^{\\⁠\s])')   # _x or ^x (single non-brace char)

def normalize_label(label: str) -> str:
    """统一下标/上标格式：_x / ^x → _{x} / ^{x}，消除 synthetic 和 human 标注差异。"""
    return _BARE_SUB_RE.sub(r'\1{\2}', label)


def tokenize(label: str) -> list:
    return _LATEX_TOKEN_RE.findall(normalize_label(label))


# ── Vocabulary（支持 BPE）───────────────────────────────────────── #

class Vocabulary:
    SPECIAL = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    PAD = 0; BOS = 1; EOS = 2; UNK = 3

    def __init__(self, tokens: list):
        all_tokens = self.SPECIAL + list(tokens)
        self.idx2token = all_tokens
        self.token2idx = {t: i for i, t in enumerate(all_tokens)}
        self._bpe_trie = self._build_bpe_trie()

    def _build_bpe_trie(self):
        merged = sorted(
            [t for t in self.idx2token if len(_LATEX_TOKEN_RE.findall(t)) > 1],
            key=lambda t: len(_LATEX_TOKEN_RE.findall(t)),
            reverse=True,
        )
        if not merged:
            return None
        trie: dict = {}
        for m in merged:
            atoms = tuple(_LATEX_TOKEN_RE.findall(m))
            node = trie
            for a in atoms:
                node = node.setdefault(a, {})
            node['__end__'] = m
        return trie

    def encode(self, tokens: list) -> list:
        if self._bpe_trie is not None:
            tokens = self._bpe_merge(tokens)
        return [self.token2idx.get(t, self.UNK) for t in tokens]

    def _bpe_merge(self, atoms: list) -> list:
        result = []
        i = 0
        trie = self._bpe_trie
        while i < len(atoms):
            node = trie
            j = i
            last_match = None
            last_j = i
            while j < len(atoms) and atoms[j] in node:
                node = node[atoms[j]]
                j += 1
                if '__end__' in node:
                    last_match = node['__end__']
                    last_j = j
            if last_match is not None:
                result.append(last_match)
                i = last_j
            else:
                result.append(atoms[i])
                i += 1
        return result

    def decode(self, ids) -> list:
        n = len(self.SPECIAL)
        raw = [self.idx2token[i] for i in ids if n <= i < len(self.idx2token)]
        atoms = []
        for t in raw:
            atoms.extend(_LATEX_TOKEN_RE.findall(t))
        return atoms

    def __len__(self):
        return len(self.idx2token)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.idx2token, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Vocabulary':
        with open(path, encoding='utf-8') as f:
            idx2token = json.load(f)
        return cls(idx2token[len(cls.SPECIAL):])


def get_vocabulary(data_dir: Path = DATA_DIR) -> Vocabulary:
    bpe = data_dir / 'bpe_vocab.json'
    char = data_dir / 'vocab.json'
    if bpe.exists():
        print(f'[Vocab] 加载 BPE vocab: {bpe}')
        return Vocabulary.load(bpe)
    if char.exists():
        print(f'[Vocab] 加载字符级 vocab: {char}')
        return Vocabulary.load(char)
    raise FileNotFoundError(f'找不到 vocab 文件，请先运行 build_bpe_vocab.py: {data_dir}')


# ── 数据增强 ─────────────────────────────────────────────────────── #

def _smooth_noise_1d(n, n_anchors, amplitude, rng):
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    ax = np.linspace(0, n - 1, n_anchors)
    ay = rng.normal(0, amplitude, n_anchors).astype(np.float32)
    return np.interp(np.arange(n), ax, ay).astype(np.float32)


def _augment(raw: list) -> list:
    rng = np.random.default_rng()
    result = []
    for s in raw:
        s = s.copy()
        if len(s) > 4:
            keep = rng.random(len(s)) > rng.uniform(0.05, 0.1)
            keep[0] = keep[-1] = True
            s = s[keep]
        xy = s[:, :2]
        x_range = max(xy[:, 0].max() - xy[:, 0].min(), 1.0)
        y_range = max(xy[:, 1].max() - xy[:, 1].min(), 1.0)
        ratio = rng.uniform(0.04, 0.08)
        n_anchors = rng.integers(4, 7)
        s[:, 0] += _smooth_noise_1d(len(s), n_anchors, ratio * x_range, rng)
        s[:, 1] += _smooth_noise_1d(len(s), n_anchors, ratio * y_range, rng)
        result.append(s)
    if not result:
        return raw
    all_xy = np.concatenate([s[:, :2] for s in result], axis=0)
    cx, cy = all_xy.mean(axis=0)
    angle = rng.uniform(-8, 8) * np.pi / 180
    scale = rng.uniform(0.88, 1.12)
    shear = rng.uniform(-0.08, 0.08)
    cos_a, sin_a = np.cos(angle) * scale, np.sin(angle) * scale
    A = np.array([[cos_a, -sin_a + shear * cos_a],
                  [sin_a,  cos_a + shear * sin_a]], dtype=np.float32)
    for s in result:
        s[:, :2] = (s[:, :2] - [cx, cy]) @ A.T + [cx, cy]
    factor = rng.uniform(0.7, 1.3)
    for s in result:
        s[:, 2] = s[:, 2] * factor
    return result


def _augment_image(img: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    if rng.random() < 0.5:
        img = img * rng.uniform(0.75, 1.0)
    if rng.random() < 0.3:
        mask = rng.random(img.shape) < rng.uniform(0.001, 0.005)
        img = np.where(mask, 1.0 - img, img)
    return img.astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────── #

class MathWritingDataset(Dataset):

    def __init__(self, split, vocab, max_src=512, max_tgt=128,
                 data_dir=DATA_DIR, max_samples=None, augment=False,
                 oversample_hard=1, calc_only=False):
        self.vocab = vocab
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.augment = augment
        self.extractor = SequenceFeatureExtractor(max_len=max_src)

        inkml_dir = Path(data_dir) / split
        parquet_path = inkml_dir.parent / f'{inkml_dir.name}.parquet'
        tag = str(inkml_dir)

        self._strokes: list = []
        self._labels: list = []

        if parquet_path.exists():
            self._load_parquet(parquet_path, vocab, max_tgt, max_samples, tag)
        else:
            self._load_inkml(inkml_dir, vocab, max_tgt, max_samples, tag)

        print(f'[Dataset:{tag}] {len(self._labels):,} 有效样本')

    def _load_parquet(self, parquet_path, vocab, max_tgt, max_samples, tag):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError('请安装 pyarrow: pip install pyarrow')
        print(f'[Dataset:{tag}] 从 parquet 加载: {parquet_path.name}', flush=True)
        table = pq.read_table(str(parquet_path), columns=['label', 'strokes_pkl'])
        n = len(table)
        if max_samples and n > max_samples:
            rng = np.random.default_rng(42)
            idx = sorted(rng.choice(n, max_samples, replace=False).tolist())
            table = table.take(idx)
            n = len(table)
        labels_col = table.column('label')
        strokes_col = table.column('strokes_pkl')
        skip = 0
        for i in range(n):
            try:
                norm = labels_col[i].as_py()
                if not norm:
                    skip += 1; continue
                toks = tokenize(norm)
                if not toks:
                    skip += 1; continue
                ids = [vocab.BOS] + vocab.encode(toks) + [vocab.EOS]
                if len(ids) > max_tgt:
                    skip += 1; continue
                strokes = pickle.loads(strokes_col[i].as_py())
                if not strokes:
                    skip += 1; continue
                self._strokes.append(strokes)
                self._labels.append(ids)
            except Exception:
                skip += 1
        print(f'[Dataset:{tag}] parquet 加载完成: ok={len(self._labels):,} skip={skip:,}', flush=True)

    def _load_inkml(self, inkml_dir, vocab, max_tgt, max_samples, tag):
        paths = sorted(inkml_dir.glob('*.inkml'))
        if max_samples and len(paths) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(paths), max_samples, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        print(f'[Dataset:{tag}] 加载 {len(paths):,} 个文件...', flush=True)
        for i, p in enumerate(paths):
            try:
                norm, strokes = parse_inkml(p)
                if not norm or not strokes:
                    continue
                toks = tokenize(norm)
                if not toks:
                    continue
                ids = [vocab.BOS] + vocab.encode(toks) + [vocab.EOS]
                if len(ids) > max_tgt:
                    continue
                self._strokes.append(strokes)
                self._labels.append(ids)
            except Exception:
                pass
            if (i + 1) % 50000 == 0:
                print(f'  {i+1:,}/{len(paths):,} → {len(self._labels):,} 有效', flush=True)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        raw = self._strokes[idx]
        ids = self._labels[idx]
        if self.augment:
            raw = _augment(raw)
        strokes_xy = [[(float(p[0]), float(p[1])) for p in s] for s in raw]
        timestamps  = [[float(p[2]) / 1000.0        for p in s] for s in raw]
        feat = self.extractor.extract(strokes_xy, timestamps)
        T = min(len(feat), self.max_src)
        feat = feat[:T].astype(np.float32)
        lw = np.random.randint(1, 4) if self.augment else 2
        img = render_strokes(strokes_xy, line_width=lw).astype(np.float32)
        if self.augment:
            img = _augment_image(img)
        img = img[np.newaxis]
        return (
            torch.from_numpy(feat),
            T,
            torch.tensor(ids, dtype=torch.long),
            torch.from_numpy(img),
        )

# ── Collate ──────────────────────────────────────────────────────── #

from torch.nn.utils.rnn import pad_sequence  # noqa: E402


def collate_fn(batch, pad_stroke: int = 0):
    feats, lens, ids_list, imgs = zip(*batch)

    stroke_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    stroke_lens = torch.tensor(lens, dtype=torch.long)
    T_max = stroke_padded.size(1)
    stroke_mask = (torch.arange(T_max).unsqueeze(0) >= stroke_lens.unsqueeze(1))

    imgs = torch.stack(imgs)
    return stroke_padded, stroke_mask, imgs, ids_list
