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
联机手写识别 —— 序列特征提取器

将笔画轨迹（含时间戳）转换为时序特征矩阵，供 RNN 使用。

特征向量（FEATURE_DIM = 13 维/点）:
  [0]  x         归一化横坐标 [0,1]
  [1]  y         归一化纵坐标 [0,1]
  [2]  dx        横向位移增量（上一点→本点）
  [3]  dy        纵向位移增量
  [4]  dt        时间增量（秒）
  [5]  speed     瞬时速度 = |Δpos|/Δt
  [6]  sin_θ     方向角正弦（圆形编码，消除 ±π 不连续）
  [7]  cos_θ     方向角余弦
  [8]  κ         曲率 = Δθ/弧长
  [9]  arc_frac  累积弧长 / 总弧长 ∈ [0,1]
  [10] pen_up    笔画切换标志（0=绘制中，1=抬笔过渡帧）
  [11] h_ratio   笔画高度 / max(整体高度, 整体宽度)（全局大小信息）
  [12] w_ratio   笔画宽度 / max(整体高度, 整体宽度)（全局大小信息）
"""

import math
import numpy as np
from typing import List, Tuple, Optional

FEATURE_DIM = 13
_PEN_UP_DT  = 0.05   # 虚拟抬笔间隔（秒），影响切换帧的速度特征

# 各特征的裁剪范围（防止数值爆炸）
_CLIP_RANGES = [
    (0.0,  1.0),   # x
    (0.0,  1.0),   # y
    (-1.0, 1.0),   # dx
    (-1.0, 1.0),   # dy
    (0.0,  2.0),   # dt
    (0.0, 20.0),   # speed
    (-1.0, 1.0),   # sin_θ
    (-1.0, 1.0),   # cos_θ
    (-50., 50.),   # κ
    (0.0,  1.0),   # arc_frac
    (0.0,  1.0),   # pen_up
    (0.0,  1.0),   # h_ratio
    (0.0,  1.0),   # w_ratio
]


class SequenceFeatureExtractor:

    def __init__(self, max_len: int = 256):
        """
        Args:
            max_len: 序列超出此长度时等间隔下采样
        """
        self.max_len = max_len

    # ── 公开接口 ─────────────────────────────────────────────────── #

    def extract(
        self,
        strokes:    List[List[Tuple[float, float]]],
        timestamps: Optional[List[List[float]]] = None,
    ) -> np.ndarray:
        """
        输入：
          strokes    — 多笔画坐标列表 [ [(x,y),...], ... ]（任意绝对坐标）
          timestamps — 对应时间戳列表 [ [t0,t1,...], ... ]（秒/单调递增）
                       传 None 时按弧长自动合成（速度信息不准确）

        输出：
          np.ndarray, shape=(T, FEATURE_DIM), dtype=float32
          T = 总点数（含抬笔过渡帧），上限 max_len
        """
        if not strokes or all(len(s) < 1 for s in strokes):
            return np.zeros((1, FEATURE_DIM), dtype=np.float32)

        # 1. 坐标整体归一化（等比缩放，居中）+ 保留大小比例
        strokes, h_ratio, w_ratio = self._normalize(strokes)

        # 2. 补全或合成时间戳
        if timestamps is None or not any(len(t) for t in timestamps):
            timestamps = self._synth_timestamps(strokes)

        # 3. 展平为带 pen_up 标记的单一序列
        pts, ts, pen_up = self._flatten(strokes, timestamps)

        # 4. 逐点计算特征（前 11 维）
        feats = self._compute_features(pts, ts, pen_up)

        # 5. 填入全局大小特征（每个点都一样）
        feats[:, 11] = h_ratio
        feats[:, 12] = w_ratio

        # 6. 裁剪数值范围
        for fi, (lo, hi) in enumerate(_CLIP_RANGES):
            feats[:, fi] = np.clip(feats[:, fi], lo, hi)

        # 6. 超长时等间隔下采样
        if len(feats) > self.max_len:
            idx   = np.round(
                np.linspace(0, len(feats) - 1, self.max_len)
            ).astype(int)
            feats = feats[idx]

        return feats.astype(np.float32)

    # ── 内部方法 ─────────────────────────────────────────────────── #

    @staticmethod
    def _normalize(
        strokes: List[List[Tuple[float, float]]]
    ) -> tuple:
        """将所有笔画坐标整体归一化到 [0,1]（等比缩放，居中），同时返回大小比例"""
        all_pts = [p for s in strokes for p in s]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        span = max(xmax - xmin, ymax - ymin, 1e-9)
        h_ratio = (ymax - ymin) / span   # 高度占比 ∈ [0,1]
        w_ratio = (xmax - xmin) / span   # 宽度占比 ∈ [0,1]
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        normed = [
            [((x - cx) / span + .5, (y - cy) / span + .5) for x, y in s]
            for s in strokes
        ]
        return normed, h_ratio, w_ratio

    @staticmethod
    def _synth_timestamps(
        strokes: List[List[Tuple[float, float]]]
    ) -> List[List[float]]:
        """按弧长合成时间戳（匀速 ≈ 1 归一化单位/秒）"""
        t, result = 0.0, []
        for si, s in enumerate(strokes):
            ts = [t]
            for i in range(1, len(s)):
                dx = s[i][0] - s[i - 1][0]
                dy = s[i][1] - s[i - 1][1]
                t += max(math.hypot(dx, dy), 1e-4)
                ts.append(t)
            result.append(ts)
            t += _PEN_UP_DT
        return result

    @staticmethod
    def _flatten(
        strokes:    List[List[Tuple[float, float]]],
        timestamps: List[List[float]],
    ) -> Tuple[list, list, list]:
        """将多笔画展平，在非末尾笔画后插入 pen_up 过渡帧"""
        pts, ts, pu = [], [], []
        for si, (s, t_list) in enumerate(zip(strokes, timestamps)):
            if not s:
                continue
            for p, t in zip(s, t_list):
                pts.append(p)
                ts.append(t)
                pu.append(0)
            if si < len(strokes) - 1:
                pts.append(s[-1])
                ts.append(t_list[-1] + _PEN_UP_DT)
                pu.append(1)
        return pts, ts, pu

    @staticmethod
    def _compute_features(
        pts:    list,
        ts:     list,
        pen_up: list,
    ) -> np.ndarray:
        n    = len(pts)
        feat = np.zeros((n, FEATURE_DIM), dtype=np.float64)

        # 预计算累积弧长
        arcs = [0.0]
        for i in range(1, n):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            arcs.append(arcs[-1] + math.hypot(dx, dy))
        total_arc = max(arcs[-1], 1e-9)

        prev_dir = 0.0
        for i in range(n):
            x, y = pts[i]
            t    = ts[i]

            if i == 0:
                dx = dy = dt = 0.0
            else:
                dx = x - pts[i - 1][0]
                dy = y - pts[i - 1][1]
                dt = max(t - ts[i - 1], 1e-6)

            dist  = math.hypot(dx, dy)
            speed = dist / dt if dt > 1e-6 else 0.0

            # 方向角（圆形编码）
            cur_dir = math.atan2(dy, dx) if dist > 1e-6 else prev_dir

            # 曲率 κ = Δθ / 弧长
            if i > 0 and dist > 1e-6:
                ddir = (cur_dir - prev_dir + math.pi) % (2 * math.pi) - math.pi
                curv = ddir / dist
            else:
                curv = 0.0

            prev_dir = cur_dir

            feat[i, :11] = [
                x,                        # 0  x_norm
                y,                        # 1  y_norm
                dx,                       # 2  dx
                dy,                       # 3  dy
                dt,                       # 4  dt
                speed,                    # 5  speed
                math.sin(cur_dir),        # 6  sinθ
                math.cos(cur_dir),        # 7  cosθ
                curv,                     # 8  κ
                arcs[i] / total_arc,      # 9  arc_frac
                float(pen_up[i]),         # 10 pen_up
            ]
            # [11] h_ratio 和 [12] w_ratio 在 extract() 中统一填入

        return feat