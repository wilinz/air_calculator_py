#!/usr/bin/env python3
"""
官方合成管线实现：LaTeX bboxes + symbols 符号库 → InkML

原理（与 MathWriting 官方一致）：
  1. synthetic-bboxes.jsonl  提供每个 token 在 LaTeX 坐标系下的精确 bbox
       LaTeX 坐标：x 向右，y 向上，baseline = y=0
  2. symbols/ + symbols.jsonl  提供每个 token 的真实手写笔画样本
       屏幕坐标：x 向右，y 向下，绝对位置任意
  3. 对每个 (token, bbox) 对：
       a. 从符号库随机采样该 token 的一条笔画轨迹
       b. 将笔画归一化到 [0,1]×[0,1]（屏幕坐标）
       c. Y 轴翻转（屏幕坐标→LaTeX坐标）
       d. 缩放 + 平移到目标 bbox
  4. 拼接所有笔画，加时间戳偏移，写出 InkML

用法 A（批量从 synthetic-bboxes.jsonl 过滤生成）:
  python3 synth_from_bboxes.py \
    --mode filter \
    --filter-tokens x X \\nu \\partial \\tilde \\overline \
    --n 5000 --augment 3 \
    --out-dir ../data/mathwriting-2024/targeted_round1

用法 B（给定自定义 bbox jsonl 生成）:
  python3 synth_from_bboxes.py \
    --mode custom \
    --bbox-file my_formulas_bboxes.jsonl \
    --augment 5 \
    --out-dir ../data/mathwriting-2024/targeted_custom
"""

import argparse
import hashlib
import json
import pickle
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Perlin / Simplex 噪声（使用 noise 库，与 Kotlin SimplexNoise.noise2 同源） #

from noise import pnoise1, snoise2  # pip install noise


def perlin_noise_1d(n: int, amplitude: float, rng: np.random.Generator,
                    frequency: float = 4.0) -> np.ndarray:
    """
    1-D Perlin 噪声，与 Kotlin SimplexNoise.noise2(x, 0.0) 等价。

    用随机 base 偏移隔离不同笔画/维度，
    用 frequency 控制噪声波动频率（段数）。
    """
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    base = float(rng.uniform(0, 1000))
    xs = np.linspace(base, base + frequency, n)
    noise = np.array([pnoise1(x) for x in xs], dtype=np.float32)
    return noise * amplitude


def add_perlin_noise(strokes: list, rng_np: np.random.Generator,
                     local_ratio: float = 0.06,
                     n_anchors_range: tuple = (3, 6)) -> list:
    """对每条笔画独立施加 Perlin 局部噪声（模拟手抖）。"""
    result = []
    for s in strokes:
        ns = s.copy()
        n = len(ns)
        if n < 2:
            result.append(ns)
            continue
        xy = ns[:, :2]
        x_range = max(float(xy[:, 0].max() - xy[:, 0].min()), 1.0)
        y_range = max(float(xy[:, 1].max() - xy[:, 1].min()), 1.0)
        freq = float(rng_np.integers(n_anchors_range[0], n_anchors_range[1]))
        ns[:, 0] += perlin_noise_1d(n, local_ratio * x_range, rng_np, frequency=freq)
        ns[:, 1] += perlin_noise_1d(n, local_ratio * y_range, rng_np, frequency=freq)
        result.append(ns)
    return result


def add_global_perlin_noise(strokes: list, rng_np: np.random.Generator,
                             global_ratio: float = 0.015,
                             n_anchors: int = 5) -> list:
    """对全局笔画施加统一 Perlin 漂移噪声（模拟手腕整体抖动）。"""
    if not strokes:
        return strokes

    all_pts = np.concatenate([s[:, :2] for s in strokes], axis=0)
    x_range = max(float(all_pts[:, 0].max() - all_pts[:, 0].min()), 1.0)
    y_range = max(float(all_pts[:, 1].max() - all_pts[:, 1].min()), 1.0)

    total_pts = sum(len(s) for s in strokes)
    nx = perlin_noise_1d(total_pts, global_ratio * x_range, rng_np, frequency=n_anchors)
    ny = perlin_noise_1d(total_pts, global_ratio * y_range, rng_np, frequency=n_anchors)

    result = []
    cursor = 0
    for s in strokes:
        ns = s.copy()
        n = len(ns)
        ns[:, 0] += nx[cursor:cursor + n]
        ns[:, 1] += ny[cursor:cursor + n]
        result.append(ns)
        cursor += n
    return result

_here = Path(__file__).parent
_root = _here.parent  # synth/
sys.path.insert(0, str(_root.parent.parent / 'train'))
DATA_DIR = _root.parent.parent / 'data' / 'mathwriting-2024'
ASSETS_DIR = _root / 'assets'
NS = 'http://www.w3.org/2003/InkML'


# ── 解析 InkML ──────────────────────────────────────────────────── #

def parse_strokes(path: Path) -> list:
    """解析 inkml 文件，返回 list[ndarray(N, 3)]，列为 [X, Y, T]。"""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        strokes = []
        for trace in root.findall(f'{{{NS}}}trace'):
            pts = []
            for tok in (trace.text or '').strip().split(','):
                v = tok.strip().split()
                if len(v) >= 2:
                    t = float(v[2]) if len(v) >= 3 else 0.0
                    pts.append([float(v[0]), float(v[1]), t])
            if pts:
                strokes.append(np.array(pts, dtype=np.float32))
        return strokes
    except Exception:
        return []


# ── 符号库索引 ───────────────────────────────────────────────────── #

def build_symbol_index(data_dir: Path = DATA_DIR) -> dict:
    """
    构建 token → list[(stroke_list, writer_id)] 索引。

    优先加载 assets/symbol_library.pkl（由 build_symbol_library.py 预先提取），
    无需 mathwriting-2024 原始 inkml 文件即可运行。
    回退到从 data_dir 实时解析 inkml。
    """
    # ── 快速路径：从预提取的 pkl 加载 ──
    pkl_path = ASSETS_DIR / 'symbol_library.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            index = pickle.load(f)
        total = sum(len(v) for v in index.values())
        print(f"[SymbolIndex] 从 {pkl_path.name} 加载: {len(index)} 种符号  {total} 个样例")
        return index

    # ── 慢速路径：从原始 inkml 构建 ──
    print("[SymbolIndex] 未找到 symbol_library.pkl，从原始 inkml 构建（较慢）…")
    sym_jsonl = data_dir / 'symbols.jsonl'
    train_dir = data_dir / 'train'
    sym_dir   = data_dir / 'symbols'

    # sourceSampleId → 文件路径（优先 train/）
    sid2path: dict[str, Path] = {}
    for d in (train_dir, sym_dir):
        if d.exists():
            for p in d.glob('*.inkml'):
                if p.stem not in sid2path:
                    sid2path[p.stem] = p

    # 缓存已解析的源文件（一个源文件可能被多个 symbol 引用）
    _stroke_cache: dict[str, list] = {}

    # 加载人工排除列表
    exclude_file = data_dir / 'excluded_samples.json'
    excluded_ids: set = set()
    if exclude_file.exists():
        excluded_ids = set(json.loads(exclude_file.read_text()))
        print(f"[SymbolIndex] 加载人工排除列表: {len(excluded_ids)} 条")

    index: dict[str, list] = defaultdict(list)
    with open(sym_jsonl) as f:
        for line in f:
            d = json.loads(line)
            label = d['label']
            sid   = d['sourceSampleId']
            idxs  = d['strokeIndices']

            sample_id = f"{sid}:{','.join(map(str, idxs))}"
            if sample_id in excluded_ids:
                continue
            if sid not in sid2path:
                continue
            if sid not in _stroke_cache:
                _stroke_cache[sid] = parse_strokes(sid2path[sid])
            all_strokes = _stroke_cache[sid]

            sel = [all_strokes[i] for i in idxs if i < len(all_strokes)]
            if sel:
                index[label].append((sel, sid))  # (strokes, writer_id)

    # ── 多维离群过滤：尺寸 + 宽高比 + 笔画数 ────────────────────────
    def _metrics(s_list):
        pts = np.concatenate([s[:, :2] for s in s_list], axis=0)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        ar  = w / max(h, 1e-3)
        n_strokes = len(s_list)
        n_pts = sum(len(s) for s in s_list)
        return w, h, ar, n_strokes, n_pts

    filtered: dict[str, list] = {}
    for label, samples in index.items():
        if len(samples) < 3:
            filtered[label] = samples
            continue

        stroke_lists = [s for s, _ in samples]
        stats = [_metrics(s) for s in stroke_lists]
        ws, hs, ars, ns, npts = zip(*stats)

        med_size = float(np.median([max(w, h) for w, h in zip(ws, hs)]))
        med_ar   = float(np.median(ars))
        med_ns   = float(np.median(ns))

        keep = []
        for (s, wid), (w, h, ar, n_s, n_p) in zip(samples, stats):
            # 1. 尺寸：最长边不超过中位数 2.5 倍
            if max(w, h) > 2.5 * med_size + 1:
                continue
            # 2. 宽高比：不超过中位数 4 倍 / 不低于 1/4
            if med_ar > 0.05:   # 避免极端点状符号误过滤
                if ar > med_ar * 4 or ar < med_ar / 4:
                    continue
            # 3. 笔画数：不超过中位数 3 倍
            if n_s > max(med_ns * 3, med_ns + 4):
                continue
            # 4. 最少有 2 个点
            if n_p < 2:
                continue
            keep.append((s, wid))

        filtered[label] = keep if keep else samples  # 全过滤则保留原样

    total = sum(len(v) for v in filtered.values())
    removed = sum(len(v) for v in index.values()) - total
    print(f"[SymbolIndex] {len(filtered)} 种符号  {total} 个样例  (过滤离群 {removed} 个)")
    return filtered


# ── 核心：符号笔画 → LaTeX bbox ──────────────────────────────────── #

def place_symbol(
    strokes: list,      # list[ndarray(N,3)]，屏幕坐标
    bbox: dict,         # {"xMin":, "yMin":, "xMax":, "yMax":}（LaTeX坐标）
    t_offset: float,    # 时间偏移
    rng: random.Random,
    rng_np: np.random.Generator | None = None,
    local_noise: float = 0.02,   # 单字符柏林噪声幅度
    ref_height: float = 0.0,      # 公式参考高度（中位 bbox 高），用于小符号下限
    min_size_ratio: float = 0.80, # 小符号最小渲染尺寸 = ref_height * min_size_ratio
) -> tuple[list, float]:
    """
    将一个符号的笔画放置到目标 bbox 位置。
    返回 (transformed_strokes, new_t_offset)

    坐标变换：
      屏幕坐标 (Y↓) → 归一化 [0,1]² → Y翻转 → 缩放到 bbox (LaTeX Y↑)
    """
    if not strokes:
        return [], t_offset

    # 1. 合并所有点，计算源 bbox
    all_pts = np.concatenate([s[:, :2] for s in strokes], axis=0)
    xs, ys = all_pts[:, 0], all_pts[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    w_src = max(x_max - x_min, 1e-3)
    h_src = max(y_max - y_min, 1e-3)

    # 2. 目标 bbox 尺寸
    # synthetic 坐标系：x 向右，y 向下，baseline=0（上方为负，下方为正）
    # bbox 中 yMin < 0（字母顶部，画面上方），yMax >= 0（字母底部，画面下方）
    x_lo = bbox['xMin']
    x_hi = bbox['xMax']
    y_lo = bbox['yMin']   # 最负值 = 画面最上方
    y_hi = bbox['yMax']   # 最大值 = 画面最下方
    w_tgt = max(x_hi - x_lo, 1e-3)
    h_tgt = max(y_hi - y_lo, 1e-3)

    # 3. 微小随机扰动
    jitter_x = rng.uniform(-0.03, 0.03) * w_tgt
    jitter_y = rng.uniform(-0.03, 0.03) * h_tgt
    scale_noise = rng.uniform(0.95, 1.05)

    if rng_np is None:
        rng_np = np.random.default_rng(rng.randint(0, 2**31))

    # 需要拉伸填充 bbox 的结构性符号（布局决定性，形状不重要）
    _STRETCH = {
        r'\frac', r'\dfrac', r'\tfrac', r'\sqrt',
        r'\overline', r'\underline',   # bars must fill the full width of the bbox
        r'\int', r'\iint', r'\iiint', r'\oint',
        r'\sum', r'\prod', r'\coprod',
        '(', ')', '[', ']', r'\{', r'\}',
        r'\lvert', r'\rvert', r'\lVert', r'\rVert', '|',
        r'\left', r'\right', r'\big', r'\Big', r'\bigg', r'\Bigg',
        r'\uparrow', r'\downarrow', r'\updownarrow',
    }
    # 需要顶部对齐的修饰符号（accent 符号位于基字母上方，应紧靠 bbox 顶部）
    _TOP_ALIGN = frozenset({r'\tilde', r'\hat', r'\vec', r'\widehat', r'\widetilde'})

    token = bbox.get('token', '')
    stretch = token in _STRETCH
    top_align = token in _TOP_ALIGN

    # ── \sqrt 分段水平渲染预计算 ─────────────────────────────────── #
    # 钩形（checkmark）保持原比例，只横向拉伸横线（vinculum）部分。
    # 垂直方向仍均匀拉伸整体。
    #
    # 1. 找源笔画中 y 最小的点（屏幕坐标 y↓ 最小 = 画面最高处），
    #    该点的 x 坐标即为钩/横线分界 x_split_src。
    # 2. 钩形目标宽度 = (x_split_src - x_min) / h_src * h_tgt，
    #    即保持钩形的宽高比不变（只随目标高度缩放）。
    #    限制在 [5%, 45%] 目标宽度之间，避免极端情况。
    # 3. 水平分段线性映射：
    #    [x_min, x_split_src] → [x_lo, x_lo + hook_w_tgt]   (钩形)
    #    [x_split_src, x_max] → [x_lo + hook_w_tgt, x_hi]   (横线)
    if token == r'\sqrt':
        # 沿每条笔画的时序扫描，维护"到目前为止见过的最高点"（累积最小 y）。
        # 当累积最小 y 首次进入"全局最高 5% 以内"时，当前点的 x 就是钩→横线的交界：
        #   · 单笔画（钩 + 横线连续）：交界点刚好是斜线转横线的那刻           ✓
        #   · 多笔画横线笔：第一个点就达到最高 → x 取横线笔的最左端             ✓
        #   · 纯钩笔（不含横线）：累积最小 y 无法进入阈值 → 不贡献 junction_xs ✓
        # 取所有笔画给出的交界 x 的最小值作为最终分界。
        _all_xy   = np.concatenate([s[:, :2] for s in strokes], axis=0)
        _y_global = float(_all_xy[:, 1].min())         # 全局最高点 y（最小值）
        _thresh   = _y_global + 0.05 * h_src            # 进入"顶 5%"的门槛
        _junc_xs: list[float] = []
        for _s in strokes:
            _run_min = np.minimum.accumulate(_s[:, 1])  # 累积最小 y（时序）
            _near    = np.flatnonzero(_run_min <= _thresh)
            if len(_near):
                _junc_xs.append(float(_s[_near[0], 0]))
        if _junc_xs:
            x_split_src = float(min(_junc_xs))
        else:
            x_split_src = float(_all_xy[np.argmin(_all_xy[:, 1]), 0])
        # 限制在源宽度的 [5%, 70%]
        x_split_src = float(np.clip(x_split_src, x_min + 0.05 * w_src, x_min + 0.70 * w_src))
        hook_src_w = x_split_src - x_min
        hook_w_tgt = hook_src_w / h_src * h_tgt
        hook_w_tgt = float(np.clip(hook_w_tgt, 0.05 * w_tgt, 0.45 * w_tgt))
        x_split_tgt = x_lo + hook_w_tgt
        vin_src_w = max(x_max - x_split_src, 1e-6)
        vin_tgt_w = w_tgt - hook_w_tgt

    result = []
    for s in strokes:
        ns = s.copy()
        if token == r'\sqrt':
            # 垂直：均匀拉伸
            ns[:, 1] = y_lo + (ns[:, 1] - y_min) / h_src * h_tgt * scale_noise + jitter_y
            # 水平：分段线性（钩形不动宽度比例，横线自由拉伸）
            x = ns[:, 0].copy()
            left  = x <= x_split_src
            right = ~left
            if hook_src_w > 1e-6:
                ns[left,  0] = x_lo + (x[left] - x_min) / hook_src_w * hook_w_tgt + jitter_x
            else:
                ns[left,  0] = x_lo + jitter_x
            ns[right, 0] = x_split_tgt + (x[right] - x_split_src) / vin_src_w * vin_tgt_w * scale_noise + jitter_x
        elif stretch:
            # 拉伸填充：宽高各自独立缩放到 bbox（结构性符号）
            ns[:, 0] = (ns[:, 0] - x_min) / w_src
            ns[:, 1] = (ns[:, 1] - y_min) / h_src
            ns[:, 0] = x_lo + ns[:, 0] * w_tgt * scale_noise + jitter_x
            ns[:, 1] = y_lo + ns[:, 1] * h_tgt * scale_noise + jitter_y
        elif top_align:
            # 顶部对齐：accent 符号紧贴 bbox 顶部，x 方向居中
            eff_h = h_tgt
            eff_w = w_tgt
            if ref_height > 0:
                min_sz = ref_height * min_size_ratio
                if max(h_tgt, w_tgt) < min_sz:
                    boost = min_sz / max(max(h_tgt, w_tgt), 1e-3)
                    eff_h = h_tgt * boost
                    eff_w = w_tgt * boost
            scale = min(eff_w / w_src, eff_h / h_src) * scale_noise
            x_bbox_ctr = (x_lo + x_hi) / 2
            x_src_ctr  = (x_min + x_max) / 2
            # 顶部对齐：源笔画顶部 (y_min) 对齐到 bbox 顶部 (y_lo)
            ns[:, 0] = x_bbox_ctr + (ns[:, 0] - x_src_ctr) * scale + jitter_x
            ns[:, 1] = y_lo + (ns[:, 1] - y_min) * scale + jitter_y
        else:
            # 等比缩放：保持字形比例，居中放入 bbox（字母/数字/普通符号）
            # 对小符号（如 . \cdot \dot）强制最小渲染高度，避免太小不可见
            eff_h = h_tgt
            eff_w = w_tgt
            if ref_height > 0:
                min_sz = ref_height * min_size_ratio
                if max(h_tgt, w_tgt) < min_sz:
                    boost = min_sz / max(max(h_tgt, w_tgt), 1e-3)
                    eff_h = h_tgt * boost
                    eff_w = w_tgt * boost
            scale = min(eff_w / w_src, eff_h / h_src) * scale_noise
            x_bbox_ctr = (x_lo + x_hi) / 2
            y_bbox_ctr = (y_lo + y_hi) / 2
            x_src_ctr  = (x_min + x_max) / 2
            y_src_ctr  = (y_min + y_max) / 2
            ns[:, 0] = x_bbox_ctr + (ns[:, 0] - x_src_ctr) * scale + jitter_x
            ns[:, 1] = y_bbox_ctr + (ns[:, 1] - y_src_ctr) * scale + jitter_y

        # 时间：曲率驱动非线性分配（拐角减速，直线加速）
        # 用相邻点方向差估计曲率，speed ∝ 1/(1 + k*|Δθ|)，然后积分得时间轴。
        _dur = float(150 + rng.randint(0, 100))
        _n   = len(ns)
        if _n >= 3:
            _xy   = ns[:, :2].astype(np.float64)
            _seg  = np.diff(_xy, axis=0)                          # (N-1, 2) 段向量
            _len  = np.linalg.norm(_seg, axis=1).clip(1e-9)       # 段长
            _ang  = np.arctan2(_seg[:, 1], _seg[:, 0])            # 各段角度
            _dang = np.abs(np.diff(_ang))                         # (N-2,) 方向差
            _dang = np.minimum(_dang, 2 * np.pi - _dang)          # 折叠到 [0,π]
            # 每个内部点的"曲率" = 两侧方向差均值
            # _dang 长度 = N-2；每个内部点 k 对应 _dang[k-1]（左侧段对）
            _curv      = np.zeros(_n)
            if len(_dang) > 0:
                _curv[1:-1] = _dang          # 内部点 1..N-2 各对应一个方向差
                _curv[0]    = _dang[0]
                _curv[-1]   = _dang[-1]
            # 弧长（点到点距离），首点为 0
            _arc_seg   = np.concatenate([[0.0], _len])
            _arc       = np.cumsum(_arc_seg)
            # 速度反比于曲率（k=6 经验值，拐角约减速 3×）
            _speed     = 1.0 / (1.0 + 6.0 * _curv)
            # dt[i] = arc_seg[i] / speed[i]，积分得时刻
            _dt        = _arc_seg / _speed
            _t         = np.cumsum(_dt)
            _t_total   = _t[-1]
            if _t_total > 1e-9:
                _t = _t / _t_total * _dur
            else:
                _t = np.linspace(0.0, _dur, _n)
        else:
            _t = np.linspace(0.0, _dur, _n)
        ns[:, 2] = t_offset + _t
        result.append(ns)

    # 单字符局部柏林噪声（模拟每个符号书写时的手部抖动）
    if local_noise > 0:
        result = add_perlin_noise(result, rng_np,
                                  local_ratio=local_noise,
                                  n_anchors_range=(3, 6))

    new_t_offset = result[-1][-1, 2] + rng.uniform(100, 300)
    return result, new_t_offset


# ── 合成单条公式 ─────────────────────────────────────────────────── #

def apply_linear_transform(strokes: list, rng: np.random.Generator,
                           strength: float = 0.06) -> list:
    """
    对全局笔画施加轻微随机线性变换（仿射），模拟书写角度/倾斜的自然变化。

    变换内容：
      - 随机旋转：±strength × 15°
      - 随机水平剪切（斜体倾向）：±strength
      - 随机各向异性缩放：1 ± strength × 0.3

    以笔画包围盒中心为原点做变换，保持整体位置不变。
    """
    if not strokes or strength <= 0:
        return strokes

    all_pts = np.concatenate([s[:, :2] for s in strokes], axis=0)
    cx = float(all_pts[:, 0].mean())
    cy = float(all_pts[:, 1].mean())

    angle  = rng.uniform(-strength * 15, strength * 15) * (np.pi / 180)
    shear  = rng.uniform(-strength, strength)
    sx     = 1.0 + rng.uniform(-strength * 0.3, strength * 0.3)
    sy     = 1.0 + rng.uniform(-strength * 0.3, strength * 0.3)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # 旋转 + 剪切 + 各向异性缩放的合成矩阵
    M = np.array([
        [sx * cos_a - shear * sy * sin_a,  -sx * sin_a - shear * sy * cos_a],
        [             sy * sin_a,                         sy * cos_a        ],
    ], dtype=np.float64)

    result = []
    for s in strokes:
        ns = s.copy()
        xy = ns[:, :2].astype(np.float64)
        xy[:, 0] -= cx
        xy[:, 1] -= cy
        xy = (M @ xy.T).T
        xy[:, 0] += cx
        xy[:, 1] += cy
        ns[:, :2] = xy
        result.append(ns)
    return result


def synthesize_formula(
    bbox_record: dict,
    sym_index: dict,
    rng: random.Random,
    rng_np: np.random.Generator | None = None,
    fallback_tokens: list | None = None,
    local_noise: float = 0.02,
    global_noise: float = 0.005,
    affine_strength: float = 0.06,
) -> list | None:
    """
    给定 bbox_record（含 bboxes 列表），生成拼接好的笔画列表。
    返回 list[ndarray(N,3)]，失败返回 None。
    """
    if rng_np is None:
        rng_np = np.random.default_rng(rng.randint(0, 2**31))

    all_strokes = []
    t_offset = 0.0
    skip_count = 0

    # 计算公式参考高度（所有 bbox 高度的中位数），供小符号尺寸下限使用
    bbox_heights = [max(b['yMax'] - b['yMin'], 1e-3) for b in bbox_record['bboxes']]
    ref_height = float(np.median(bbox_heights)) if bbox_heights else 0.0

    # ── 一致笔迹风格采样 ──────────────────────────────────────────────
    # 同一公式内相同 token 优先选同一个书写者（writer_id = sourceSampleId），
    # 避免 x+x² 里两个 x 来自不同笔迹的人。
    # 策略：
    #   1. 预先收集公式内所有 token 的候选 writer_id 集合交集。
    #   2. 若存在交集，从中随机挑一个 preferred_writer；否则不限制。
    #   3. 每次采样时，优先从 preferred_writer 的样本中选；
    #      若该 writer 没有该 token 则 fallback 到全体候选。
    _token_list   = [b['token'] for b in bbox_record['bboxes']]
    _writer_sets  = []
    for _tok in set(_token_list):
        _cands = sym_index.get(_tok) or []
        if _cands:
            _writer_sets.append({wid for _, wid in _cands})
    if _writer_sets:
        _common = set.intersection(*_writer_sets)
    else:
        _common = set()
    preferred_writer: str | None = rng.choice(list(_common)) if _common else None

    for b in bbox_record['bboxes']:
        token = b['token']

        # 查找符号样例
        candidates = sym_index.get(token)
        if not candidates:
            # 尝试大小写互转 fallback
            alt = token.upper() if token.islower() else token.lower()
            candidates = sym_index.get(alt)
        if not candidates:
            # 跳过（空格、结构命令等）
            skip_count += 1
            continue

        # 优先选 preferred_writer 的样本
        if preferred_writer is not None:
            writer_cands = [(s, wid) for s, wid in candidates if wid == preferred_writer]
        else:
            writer_cands = []
        chosen_entry = rng.choice(writer_cands if writer_cands else candidates)
        strokes = chosen_entry[0]   # (strokes, writer_id)[0]

        placed, t_offset = place_symbol(
            strokes, b, t_offset, rng,
            rng_np=rng_np, local_noise=local_noise,
            ref_height=ref_height,
        )
        all_strokes.extend(placed)

    if not all_strokes:
        return None
    # 跳过太多 token 的公式（符号库覆盖不足）
    total_tokens = len(bbox_record['bboxes'])
    if skip_count > total_tokens * 0.5:
        return None

    # 整体柏林噪声（模拟手腕整体漂移）
    if global_noise > 0:
        all_strokes = add_global_perlin_noise(all_strokes, rng_np,
                                              global_ratio=global_noise)

    # 轻微线性变换（随机旋转 + 剪切 + 各向异性缩放）
    if affine_strength > 0:
        all_strokes = apply_linear_transform(all_strokes, rng_np,
                                             strength=affine_strength)

    return all_strokes


# ── 写出 InkML ───────────────────────────────────────────────────── #

def write_inkml(strokes: list, label: str, norm_label: str,
                out_path: Path, sample_id: str):
    ink = ET.Element('ink', xmlns=NS)

    def ann(type_, text):
        e = ET.SubElement(ink, 'annotation', type=type_)
        e.text = str(text)

    ann('label', label)
    ann('normalizedLabel', norm_label)
    ann('splitTagOriginal', 'targeted_synthetic')
    ann('inkCreationMethod', 'boundingBoxes')
    ann('sampleId', sample_id)

    fmt = ET.SubElement(ink, 'traceFormat')
    for ch_name, ch_type, units in [('X','decimal',None),('Y','decimal',None),('T','decimal','ms')]:
        kw = {}
        if units: kw['units'] = units
        ET.SubElement(fmt, 'channel', name=ch_name, type=ch_type, **kw)

    for i, s in enumerate(strokes):
        trace = ET.SubElement(ink, 'trace', id=str(i))
        trace.text = ','.join(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.1f}" for p in s)

    tree = ET.ElementTree(ink)
    ET.indent(tree, space='  ')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding='unicode', xml_declaration=False)


# ── 过滤规则 ─────────────────────────────────────────────────────── #

# 预定义：针对不同错误类型的目标 token 集合
ERROR_TYPE_TOKENS = {
    'case_mismatch': {
        # 含易混淆大小写字母的公式
        'include_any': ['x', 'X', 'p', 'P', 's', 'S', 'd', 'D',
                        '\\nu', '\\omega', '\\mu', '\\partial'],
        'min_tokens': 6,
    },
    'bracket_confusion': {
        # 含深层嵌套括号/上下标的公式
        'include_any': ['\\frac', '\\int', '\\sum', '\\prod'],
        'min_tokens': 8,
    },
    'visual_similar': {
        'include_any': ['\\tilde', '\\overline', '\\partial',
                        '\\gg', '\\ll', '\\cdot', '\\times'],
        'min_tokens': 6,
    },
    'insertion': {
        # 含 \left \right 配对括号的公式
        'include_any': ['\\left', '\\right', '(', ')'],
        'min_tokens': 6,
    },
    'all': {
        'include_any': None,  # 不过滤
    },
}


def passes_filter(record: dict, filter_cfg: dict) -> bool:
    tokens = {b['token'] for b in record['bboxes']}
    n = len(record['bboxes'])

    include_any = filter_cfg.get('include_any')
    if include_any is not None:
        if not any(t in tokens for t in include_any):
            return False

    min_tok = filter_cfg.get('min_tokens', 3)
    if n < min_tok:
        return False

    max_tok = filter_cfg.get('max_tokens', 40)
    if n > max_tok:
        return False

    return True


# ── 主流程 ──────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['filter', 'custom'], default='filter',
                        help='filter=从 synthetic-bboxes.jsonl 过滤；custom=用自定义 bbox jsonl')
    parser.add_argument('--bbox-file', default=None,
                        help='custom 模式下的 bbox jsonl 文件')
    parser.add_argument('--error-types', nargs='+',
                        default=['case_mismatch', 'bracket_confusion', 'visual_similar'],
                        choices=list(ERROR_TYPE_TOKENS.keys()))
    parser.add_argument('--n', type=int, default=5000,
                        help='目标生成总数')
    parser.add_argument('--augment', type=int, default=3,
                        help='每条公式增强几份（不同符号样本）')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local-noise', type=float, default=0.02,
                        help='单字符局部柏林噪声幅度（相对符号尺寸），0=关闭')
    parser.add_argument('--global-noise', type=float, default=0.005,
                        help='整体全局柏林噪声幅度（相对整体尺寸），0=关闭')
    parser.add_argument('--affine-strength', type=float, default=0.06,
                        help='线性变换强度（旋转±15°×s，剪切±s，缩放±0.3×s），0=关闭')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    # 构建符号索引
    sym_index = build_symbol_index(data_dir)

    # 读取 bbox 记录
    if args.mode == 'filter':
        bbox_src = data_dir / 'synthetic-bboxes.jsonl'
    else:
        assert args.bbox_file, '--bbox-file 必须提供（custom 模式）'
        bbox_src = Path(args.bbox_file)

    # 合并所有错误类型的过滤配置
    merged_tokens = set()
    for et in args.error_types:
        cfg = ERROR_TYPE_TOKENS.get(et, {})
        inc = cfg.get('include_any') or []
        merged_tokens.update(inc)

    combined_min = max(
        (ERROR_TYPE_TOKENS.get(et, {}).get('min_tokens', 3) for et in args.error_types),
        default=3,
    )
    combined_filter = {
        'include_any': list(merged_tokens) if merged_tokens else None,
        'min_tokens': combined_min,
        'max_tokens': 40,
    }

    # 读取并过滤
    records = []
    with open(bbox_src) as f:
        for line in f:
            d = json.loads(line)
            if passes_filter(d, combined_filter):
                records.append(d)

    print(f"[Filter] {len(records):,} / {args.n} 条符合条件")
    if not records:
        print("无匹配记录，退出")
        return

    # 生成
    n_per_record = args.augment
    target = args.n
    generated = skipped = 0

    # 打乱顺序
    rng.shuffle(records)

    for rec in records:
        if generated >= target:
            break
        for aug_i in range(n_per_record):
            if generated >= target:
                break
            strokes = synthesize_formula(
                rec, sym_index, rng, rng_np=rng_np,
                local_noise=args.local_noise,
                global_noise=args.global_noise,
                affine_strength=args.affine_strength,
            )
            if strokes is None:
                skipped += 1
                continue

            label = rec.get('label', rec['normalizedLabel'])
            norm  = rec['normalizedLabel']
            uid = hashlib.md5(
                f"{norm}_{aug_i}_{rng.random()}".encode()
            ).hexdigest()[:16]
            write_inkml(strokes, label, norm, out_dir / f"{uid}.inkml", uid)
            generated += 1

        if generated % 500 == 0:
            print(f"  进度: {generated}/{target} (跳过 {skipped})", flush=True)

    print(f"\n[完成] 生成 {generated} 个 inkml，跳过 {skipped} 条")
    print(f"输出: {out_dir}")


if __name__ == '__main__':
    main()
