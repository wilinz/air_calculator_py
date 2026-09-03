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
笔画序列 → 灰度图像渲染器

供双流模型使用：笔画特征序列（在线）+ 渲染图像（离线）联合识别。
"""

import numpy as np
from PIL import Image, ImageDraw

IMG_H = 64    # 渲染图像高度（像素）
IMG_W = 256   # 渲染图像宽度（像素）
LINE_W = 2    # 笔迹线宽


def render_strokes(
    strokes_xy,
    height: int   = IMG_H,
    width:  int   = IMG_W,
    line_width: int   = LINE_W,
    stretch_limit: float = 2.5,
) -> np.ndarray:
    """
    将笔画序列渲染为归一化灰度图像。

    Args:
        strokes_xy: list[list[tuple(x, y)]] 或 list[ndarray(N, >=2)]
            每条笔画为点列表，取前两维作为 (x, y)
        height, width: 输出图像尺寸
        line_width: 笔迹线宽（像素）

    Returns:
        np.ndarray (height, width) float32，0=背景，1=笔迹
    """
    all_pts = []
    for s in strokes_xy:
        for pt in s:
            all_pts.append((float(pt[0]), float(pt[1])))

    if not all_pts:
        return np.zeros((height, width), dtype=np.float32)

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    margin  = line_width + 3
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    # 先等比缩放（保持字形），再允许在较短方向上最多拉伸 stretch_limit 倍
    # stretch_limit=1.0 → 完全等比；stretch_limit=inf → 完全独立缩放
    base_scale = min((width  - 2 * margin) / x_range,
                     (height - 2 * margin) / y_range)
    x_scale = min((width  - 2 * margin) / x_range, base_scale * stretch_limit)
    y_scale = min((height - 2 * margin) / y_range, base_scale * stretch_limit)

    x_off = (width  - x_range * x_scale) / 2
    y_off = (height - y_range * y_scale) / 2

    def to_px(x, y):
        return (
            int((x - x_min) * x_scale + x_off),
            int((y - y_min) * y_scale + y_off),
        )

    img  = Image.new('L', (width, height), color=0)   # 黑色背景
    draw = ImageDraw.Draw(img)

    for s in strokes_xy:
        pts_px = [to_px(float(pt[0]), float(pt[1])) for pt in s]
        if len(pts_px) == 1:
            x, y = pts_px[0]
            r = max(line_width // 2, 1)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        elif len(pts_px) >= 2:
            draw.line(pts_px, fill=255, width=line_width)

    return np.array(img, dtype=np.float32) / 255.0