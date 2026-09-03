#!/usr/bin/env python3
"""
Android TFLite (LiteRT) 导出：prefix_enc.tflite + decoder.tflite，当前 Android
部署版（187 MB，encoder 与 decoder 都走 GPU，整图下沉）。

图结构按 Adreno OpenCL GpuDelegateV2 算子兼容性逐项改写（模型权重不变）：
  · CAST INT32→INT64           → 内部全程 int32；F.embedding 用 onehot @ W
  · GREATER_EQUAL 逻辑算子      → 因果 mask 改为「常量表 + onehot 行选取」
  · torch.where（KV 写入）      → onehot 浮点掩码混合：past*(1-m) + new*m
  · F.pad v4                   → torch.cat 拼零填充
  · SLICE v5                   → reshape/transpose/view 替代切片
  · ADD 广播 [1,N,D] vs [N,D]   → 显式 unsqueeze(0) / expand 补齐 batch 维
  · FULLY_CONNECTED 多 runtime  → attention KV 写入避开 runtime 矩阵乘

用法：
  cd air_calculator_py/export
  python3 export_tf_android_fp32.py \
    --ckpt ../../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
    --data-dir ../../dataset/mathwriting-2024 \
    --out ../../weights/exports/tflite_android \
    --max-decode 32
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))

import torch
import torch.nn as nn
import torch.nn.functional as F



def _layer_norm(x, ln):
    """手写 LayerNorm，避开 SQUARED_DIFFERENCE。

    `nn.LayerNorm` 经 torch.export 分解后会出现 SQUARED_DIFFERENCE，而 XNNPACK
    不支持它——prefill 子图里 15 个 LayerNorm 就把图劈成 35 个分区（decode 只有
    8 个），碎片间反复出入线程池，实测 prefill 74.8ms 只跑到 21.5 GFLOP/s，约为
    骁龙 870 四大核峰值的四分之一。

    用 `xc * xc` 代替平方差，其余照 LayerNorm 定义写，数值等价，算子全部是
    XNNPACK 认识的 MEAN / SUB / MUL / RSQRT / ADD。

    注：`xc` 实测最大到 6346，平方 4.0e7 已超过 fp16 上限 65504。试过先除以
    一个常数 C 再平方（数学上等价），C=32/64 都能让 CPU 结果不变，但 encoder
    跑在 GPU 上时反而崩——小值被压进 fp16 次正规数，GPU 刷成 0。所以这里保持
    原式：CPU 走 fp32 激活，不受影响。
    """
    # 平方前先按行内最大值归一，否则 fp16 会溢出。
    #
    # 实测中心化后 |xc| 最大到 6346，平方 4.0e7 远超 fp16 上限 65504，溢出成
    # inf 后 mean/rsqrt 一路塌成 0。这是 LiteRT 已知问题
    # （google-ai-edge/LiteRT#7693，EmbeddingGemma 在 GPU 上返回全零向量，
    # 原文定位到 RMSNorm 的 `MUL x*x` 溢出），官方给的绕法是强制 fp32，
    # 但那档比 CPU 还慢，只能在图上解决。
    #
    # 用固定常数缩放不行：常数按 decoder 的量级选，encoder 的小激活会被压进
    # fp16 次正规区（<6.1e-5），GPU 刷成 0 后同样崩——试过 C=32/64 都是这样。
    # 改成按行内最大值 m 归一，u ∈ [-1,1]，两头都安全：
    #     y = (xc/m) · rsqrt(mean((xc/m)²) + eps/m²) = xc · rsqrt(var + eps)
    # 逐位等价。m 取下限 1e-3 是为了全零行：那时 u=0、eps/m²=10，rsqrt 有限，
    # y=0，不会出 0/0。
    xc = x - x.mean(dim=-1, keepdim=True)

    # 定标路径：`calibrate_ln_scales` 量过这个 LayerNorm 的取值范围，确认一个
    # 编译期常数就能兜住所有行，于是 m 退化成常数。ABS / REDUCE_MAX / MAXIMUM
    # 三个算子连同 eps/(m*m) 的那次 DIV 全部消失（常数在图里直接折掉），
    # 运行时零开销。没标常数的照旧走行内最大值。
    c = getattr(ln, 'gpu_const_scale', None)
    if c is not None:
        u = xc * (1.0 / c)
        var = (u * u).mean(dim=-1, keepdim=True)
        y = u * torch.rsqrt(var + ln.eps / (c * c))
    else:
        m = xc.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)
        u = xc / m
        var = (u * u).mean(dim=-1, keepdim=True)
        y = u * torch.rsqrt(var + ln.eps / (m * m))
    if ln.weight is not None:
        y = y * ln.weight
    if ln.bias is not None:
        y = y + ln.bias
    return y


# 定标的安全窗口，约束的是喂给 rsqrt 的 var = mean(u²)，不是 u 的峰值——
# var 比峰值还小一个量级，按峰值定标会让幅度最小的行的 var 掉到 6e-5 以下，
# GPU 刷成 0 之后 rsqrt 出 inf，整行塌掉（实测过，输出直接变成 token 0）。
#
#   上界：sum(u²) 是 512 项累加，fp16 上限 65504 → var 不超过 128 才稳妥，
#         这里留一档取 32。
#   下界：fp16 正规数下限 6.104e-5，留两个数量级取 1e-2。
#
# 中间有三个半数量级，够放下大多数 LayerNorm 的跨行波动。
#
# 窗口开到 fp16 的真实边界，别再留冗余的保守量——留得越多，能给常数的活动
# 空间越小，裕度越薄。实测拿 40 条校准、窗口取 [1e-2, 32] 时，最险的
# `dec.layers.0.norm1` 在 500 条上只剩 1.1 倍裕度就顶到上界；顶出去就是
# sum(u²) 溢出 → rsqrt(inf) → 整行塌成 0，表现就是「特定输入固定错」。
_LN_VAR_HI = 96.0     # 硬限 128（512 项和不超过 65504），留 1.33 倍
_LN_VAR_LO = 1e-3     # 硬限 6.104e-5（fp16 正规下限），留 16 倍
# 常数选定后，观测范围到窗口两端的倍数低于这个值就不敢用，退回 row-max。
_LN_MIN_MARGIN = 4.0


def calibrate_ln_scales(decoder, prefix_encoder, samples, stroke_len, vocab,
                        max_decode=48, verbose=True):
    """量一遍 decoder 各 LayerNorm 的行内幅度，给范围够窄的挂上常数缩放。

    row-max 那套（ABS + REDUCE_MAX + MAXIMUM + DIV）是为 fp16 防溢出加的，
    17 个 LayerNorm 摊下来占 decode 子图约 15% 的算子，但它们不做任何数学。
    真正要防的是 sum(xc²) 在 fp16 里累加溢出——512 项各上万就爆了——而这
    用一个编译期常数把 xc 压到 [-1,1] 就够，不必每行去求最大值。

    只有跨行幅度差得太开的 LayerNorm 才留 row-max：常数按最大的行选，最小的
    那行会被压到 1/倍数，倍数过大时会掉进 fp16 次正规区被刷成 0。
    """
    import numpy as np

    stats = {}

    def hook(name):
        def f(mod, inp, _out):
            x = inp[0].detach()
            xc = x - x.mean(-1, keepdim=True)
            d = stats.setdefault(name, [[], []])
            d[0].extend((xc * xc).mean(-1).flatten().tolist())   # 每行 var
            d[1].extend(xc.abs().amax(-1).flatten().tolist())    # 每行峰值
        return f

    # decoder 和 encoder 的 LayerNorm 一起量。两边的激活量级差得远（decoder
    # 的 xc 到几百，encoder 只有个位数），所以常数必须一人一个——当年用一个
    # 全局常数同时压两边，decoder 那档的常数把 encoder 的小激活压进 fp16 次
    # 正规区，GPU 刷成 0，这就是注释里说的那次失败。
    handles, named = [], []
    for prefix, model in (('dec.', decoder), ('enc.', prefix_encoder)):
        for name, mod in model.named_modules():
            if isinstance(mod, nn.LayerNorm):
                full = prefix + name
                named.append((full, mod))
                handles.append(mod.register_forward_hook(hook(full)))

    from stroke_features import SequenceFeatureExtractor, FEATURE_DIM
    from stroke_renderer import IMG_H, IMG_W
    ext = SequenceFeatureExtractor(max_len=stroke_len)
    with torch.no_grad():
        for r in samples:
            st = [[(p[0], p[1]) for p in tr] for tr in r['traces']]
            ts = [[p[2] / 1000 for p in tr] for tr in r['traces']]
            f = ext.extract(st, ts)
            src = torch.zeros(1, stroke_len, FEATURE_DIM)
            src[0, :len(f)] = torch.tensor(f)
            pad = torch.ones(1, stroke_len, dtype=torch.bool)
            pad[0, :len(f)] = False
            prefix = prefix_encoder(torch.zeros(1, 1, IMG_H, IMG_W), src, pad)
            ids = [vocab.BOS]
            for _ in range(max_decode):
                logits = decoder(prefix, torch.tensor([ids]))
                nxt = int(logits[0, -1].argmax())
                if nxt == vocab.EOS:
                    break
                ids.append(nxt)
    for h in handles:
        h.remove()

    n_const = 0
    for name, mod in named:
        var = np.array(stats.get(name, [[], []])[0])
        peak = np.array(stats.get(name, [[], []])[1])
        var = var[var > 0]
        if var.size == 0:
            continue
        # C² 取 var 的几何中位数，缩放后的 var 在对数刻度上以 1 为中心，
        # 上下两头离窗口边界最远。
        c = float(np.sqrt(np.sqrt(var.min() * var.max())))
        lo, hi = float(var.min()) / (c * c), float(var.max()) / (c * c)
        # 峰值也要核一遍：单个 u² 同样不能冲破 fp16 上限。
        u_peak_sq = (float(peak.max()) / c) ** 2
        margin = min(lo / _LN_VAR_LO, _LN_VAR_HI / hi)
        if margin < _LN_MIN_MARGIN or u_peak_sq > 65504.0:
            if verbose:
                print(f'  {name:24s} var 缩放后 [{lo:.2e}, {hi:.2e}]，裕度只有 '
                      f'{margin:.1f}x → 保留 row-max')
            continue
        mod.gpu_const_scale = c
        n_const += 1
        if verbose:
            print(f'  {name:24s} C={c:8.2f}  var→[{lo:.2e}, {hi:.2e}]  '
                  f'裕度 {margin:5.1f}x')
    if verbose:
        print(f'  {n_const}/{len(named)} 个 LayerNorm 走常数定标，'
              f'每个省 4 个算子（窗口 [{_LN_VAR_LO:g}, {_LN_VAR_HI:g}]）')
    return n_const


def _manual_attention(q, k, v, attn_mask=None):
    """显式分解的 attention，规避 F.scaled_dot_product_attention 在 litert_torch
    下分解出的 `MUL: 2 const inputs` 模式（scale tensor × const fold 中间态）。

    q/k/v: (B, H, T_q, D_h) / (B, H, T_k, D_h)；attn_mask 形状广播到 (B,H,T_q,T_k)。
    """
    scale = 1.0 / math.sqrt(q.shape[-1])  # python float → 直接折成 lhs scalar，避免 const tensor
    # 把 scale 提前乘进 q（一次 mul，runtime × float），避免 SDPA 内部的链式 mul
    q_scaled = q * scale
    scores = q_scaled @ k.transpose(-2, -1)        # (B,H,T_q,T_k)
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = scores.softmax(dim=-1)
    return attn @ v                                # (B,H,T_q,D_h)


# ── GPU 友好的位置/词表 embedding：用 onehot @ W 替代 F.embedding ──────────
# F.embedding 在 TFLite 里通常 lower 为 GATHER，对应 INT64 索引；GPU delegate
# 对 CAST INT32→INT64 拒绝。改成 onehot(int32) @ embedding_weight 可保持纯
# matmul/add，全 fp32 路径，无 int64 介入。

def _make_onehot(idx: torch.Tensor, num_classes: int, dtype, device) -> torch.Tensor:
    """构造 (M, num_classes) onehot；M = idx 展平后元素个数。

    避开 torch.eye / F.one_hot（前者内部用 uint8 arange，后者要求 int64 索引）。

    idx 必须已经是浮点：早先这里收 int32 再 `.to(dtype)`，图里会留下 CAST。
    那两个 CAST 既进不了 XNNPACK，在 GPU 上也要跨委托边界，是 decode 子图在
    GPU fp16 下算错的头号嫌疑——索引一旦取错，onehot 会整片为 0，模型每步
    看到相同的零 embedding，于是同一个 token 无限重复。改由调用方直接喂
    float32，图里不再有 CAST。
    """
    rng = torch.arange(num_classes, dtype=dtype, device=device)
    flat = idx.reshape(-1).to(dtype)                       # 已是浮点，不产生 CAST
    diff = flat.unsqueeze(-1) - rng.unsqueeze(0)           # (M, C)
    # 用 relu(1-d)·relu(1+d) 而不是 1-|d| 再 clamp：整数 d 下两者完全一致
    # （d=0 得 1，|d|>=1 得 0），但前者只用 RELU 与 MUL，后者会生成 ABS 与
    # RELU_0_TO_1——这两个算子 XNNPACK 和 ml_drift 都接不了，每出现一次就把
    # 图劈开一次。
    return F.relu(1.0 - diff) * F.relu(1.0 + diff)         # 命中位=1 其余=0


# 试验开关：用 GATHER 取行，而不是构造 onehot 再做矩阵乘。当年 GpuDelegateV2
# 拒绝 F.embedding 是因为它带 INT32→INT64 的隐式转换，ml_drift 的算子集不同，
# 值得实测。置 False 回到 onehot 写法。
USE_GATHER_LOOKUP = False


def _onehot_lookup(idx: torch.Tensor, weight: torch.Tensor, num_classes: int) -> torch.Tensor:
    """idx: (...,) float32；weight: (num_classes, dim) → (..., dim)。"""
    if USE_GATHER_LOOKUP:
        flat = idx.reshape(-1).to(torch.int32)
        out = torch.index_select(weight, 0, flat)
        return out.reshape(*idx.shape, weight.shape[-1])
    onehot = _make_onehot(idx, num_classes, weight.dtype, weight.device)  # (M, C)
    out = onehot @ weight                                  # (M, dim)
    return out.reshape(*idx.shape, weight.shape[-1])


# ── PrefixEnc 展开版（stroke encoder 手术展开 TransformerEncoderLayer，
# 让 in_proj_weight 单消费，绕过 Core ML fp16 / PT2E XNNPACK int8 lower 的
# "双消费 → 图重写 mishandle" bug。详见
# docs_final/mobile/2026-05-05_prefix_enc展开修复.md。 ─────────────

def _stroke_layer_forward(layer: nn.Module, h: torch.Tensor,
                          nhead: int, head_dim: int, d_model: int,
                          key_pad_bias):
    """norm_first nn.TransformerEncoderLayer 全展开（self-attn + FF）。
    !! in_proj_weight / in_proj_bias 严禁加 .to() —— 单消费敏感。"""
    h_norm = _layer_norm(h, layer.norm1)
    qkv = F.linear(h_norm, layer.self_attn.in_proj_weight,
                   layer.self_attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    B, T = h.shape[0], h.shape[1]
    q_mh = q.unflatten(-1, (nhead, head_dim)).transpose(1, 2)
    k_mh = k.unflatten(-1, (nhead, head_dim)).transpose(1, 2)
    v_mh = v.unflatten(-1, (nhead, head_dim)).transpose(1, 2)

    # key_pad_bias 已经是 (B,1,1,T) 的浮点加性 mask，这里不再构造 BOOL、
    # 也不再用 masked_fill——那条路会在图里留下 GREATER_EQUAL / SELECT_V2 和
    # 一个 BOOL 张量，ml_drift 接不了，encoder 因此只能下沉 139/152、被切成
    # 2 个分区。改写见 PrefixEncUnrolledModule.forward。
    #
    # attention 也换成 _manual_attention：SDPA 在 litert_torch 下会分解出
    # 「两个常量相乘」的 MUL 中间态，这个文件为此专门写了手动版。
    out = _manual_attention(q_mh, k_mh, v_mh, attn_mask=key_pad_bias)
    out = out.transpose(1, 2).reshape(B, T, d_model)
    out = F.linear(out,
                   layer.self_attn.out_proj.weight.to(out.dtype),
                   layer.self_attn.out_proj.bias.to(out.dtype))
    h = h + out

    h_norm2 = _layer_norm(h, layer.norm2)
    ff = layer.activation(F.linear(h_norm2,
                                   layer.linear1.weight.to(h_norm2.dtype),
                                   layer.linear1.bias.to(h_norm2.dtype)))
    ff = F.linear(ff,
                  layer.linear2.weight.to(ff.dtype),
                  layer.linear2.bias.to(ff.dtype))
    h = h + ff
    return h


class PrefixEncUnrolledModule(nn.Module):
    """与 VisualPrefixEncoder 接口一致的展开版，stroke 分支手动展开
    TransformerEncoderLayer（绕过 in_proj_weight 双消费 bug）。
    输入：img (1,1,IMG_H,IMG_W) + stroke (1,STROKE_LEN,13) + stroke_real_len (1,)
    输出：prefix_embeds (1, n_img+n_stroke, d_model)"""
    def __init__(self, prefix_enc: nn.Module, stroke_len: int):
        super().__init__()
        pe = prefix_enc
        se = pe.stroke_encoder
        self.use_img = getattr(pe, 'use_img', True)
        if self.use_img:
            self.img_encoder = pe.img_encoder
            self.img_proj = pe.img_proj
            self.img_norm = pe.img_norm
        self.stroke_proj_in = se.proj
        self.stroke_layers = se.encoder.layers
        self.stroke_n_out = se.n_out
        self.stroke_pool_k = stroke_len // se.n_out
        self.stroke_proj_out = pe.stroke_proj
        self.stroke_norm = pe.stroke_norm
        self.stroke_len = stroke_len
        self.stroke_d = se.d_model
        attn0 = self.stroke_layers[0].self_attn
        self.stroke_nhead = attn0.num_heads
        self.stroke_head_dim = self.stroke_d // self.stroke_nhead
        self.register_buffer('stroke_positions',
                             torch.arange(stroke_len, dtype=torch.float32).view(1, -1))

    def forward(self, img: torch.Tensor,
                stroke: torch.Tensor,
                stroke_real_len: torch.Tensor) -> torch.Tensor:
        if self.use_img:
            img_tok = self.img_encoder(img)
            img_emb = _layer_norm(self.img_proj(img_tok), self.img_norm)

        x = self.stroke_proj_in(stroke)

        # padding mask 不走比较运算。令 d = pos - real_len（两边都是整数），
        # 铰链差 relu(d+1) - relu(d) 恰好等于 [pos >= real_len]：
        #   d <= -1 → 0 - 0 = 0        （有效位）
        #   d ==  0 → 1 - 0 = 1        （第一个 padding）
        #   d >=  1 → (d+1) - d = 1    （其余 padding）
        # 只用 SUB 和 RELU，图里不再出现 GREATER_EQUAL / SELECT_V2 / BOOL。
        #
        # real_len 收浮点而不是 int32：收整数会在图里留一个 CAST，同样交不出去
        # （decoder 那边的 token/pos 早就是这么处理的）。
        d = self.stroke_positions - stroke_real_len.reshape(1, 1)   # (1, T)
        ind = F.relu(d + 1.0) - F.relu(d)                           # 1=padding
        # -30000 而不是 -inf：这张表会被 fp16 覆盖，-inf 会让 softmax 出 NaN，
        # -30000 在 fp16 里精确可表示，softmax 之后同样压到 0。
        key_pad_bias = (ind * -30000.0).view(1, 1, 1, self.stroke_len)

        for layer in self.stroke_layers:
            x = _stroke_layer_forward(
                layer, x,
                self.stroke_nhead, self.stroke_head_dim, self.stroke_d,
                key_pad_bias)

        xt = x.transpose(1, 2).unsqueeze(2)
        xt = F.avg_pool2d(xt,
                          kernel_size=(1, self.stroke_pool_k),
                          stride=(1, self.stroke_pool_k))
        xt = xt.squeeze(2)
        stroke_tok = xt.transpose(1, 2)
        stroke_emb = _layer_norm(self.stroke_proj_out(stroke_tok), self.stroke_norm)
        if not self.use_img:
            return stroke_emb
        return torch.cat([img_emb, stroke_emb], dim=1)


# ── Prefill 模块（GPU 友好版）────────────────────────────────────────────

class DecoderPrefillModule(nn.Module):
    """Prefill：prefix_embeds (1, n_prefix, d_model) → kv (n_layers, 2, 1, max_kv, d_model)。"""

    def __init__(self, decoder, n_prefix, max_kv, n_layers, nhead, d_model,
                 bos_id=None, logit_bias=None):
        super().__init__()
        # bos_id / logit_bias 给出时，prefill 顺带把 BOS 那一步也算掉，直接返回
        # 第一个 logits。prefill 和一个解码步读的是同一份 49 MB 权重，成本几乎
        # 相同（实测 7.9 对 7.6 ms），合成一趟就省掉整整一次权重读。
        self.fuse_bos = bos_id is not None
        self.n_prefix = n_prefix
        self.max_kv = max_kv
        self.n_layers = n_layers
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        # 把 pos_embed 权重当成普通参数引用
        self.register_buffer('pos_embed_weight', decoder.pos_embed.weight.detach().clone())
        self.drop = decoder.drop
        self.layer_list = decoder.layers.layers
        # 常量化的 prefix 位置 embedding：trace 时直接折叠为常量，无 GATHER
        with torch.no_grad():
            pe_const = decoder.pos_embed.weight[:n_prefix].detach().clone()  # (n_prefix, d_model)
        self.register_buffer('prefix_pos_embed', pe_const.unsqueeze(0))  # (1, n_prefix, d_model)
        # 单层用的 (1, max_kv - n_prefix, d_model) 零尾巴；按层 cat 后再 stack，
        # 避免对 5D 张量做 CONCATENATION（GPU delegate 拒绝高维）
        # 融合 BOS 后序列长 33，是这张图独有的非对齐宽度（decode 的 softmax
        # 宽 80、非融合 prefill 宽 32，都是 8 的倍数）。补到 8 的倍数，多出来
        # 的位置全程 mask 掉、k/v 也切掉，只为让 kernel 走对齐路径。
        n_real = n_prefix + (1 if self.fuse_bos else 0)
        n_seq = ((n_real + 7) // 8) * 8 if self.fuse_bos else n_real
        self.n_real = n_real
        self.n_seq = n_seq
        zeros_one = torch.zeros(1, max_kv - n_real, d_model)
        self.register_buffer('kv_pad_zeros_one', zeros_one)

        if self.fuse_bos:
            self.norm = decoder.norm
            self.head = decoder.head
            self.register_buffer('logit_bias', logit_bias)
            # BOS 的输入向量是常量：token embedding + 第 n_prefix 位的位置编码。
            with torch.no_grad():
                bos_vec = (decoder.tok_embed.weight[bos_id]
                           + decoder.pos_embed.weight[n_prefix])
            self.register_buffer('bos_embed', bos_vec.view(1, 1, d_model).clone())
            # 整条序列的常量加项：0..n_prefix-1 是 prefix 的位置编码，第 n_prefix
            # 行是 BOS 的 (token embedding + 位置编码)，补齐行为 0。
            # 这样就不用 `+ 0 * x` 去物化常量了——那个技巧在 GPU 上不可靠：
            # 乘的是整块张量时没事，乘一个切片时 BOS 那行会被 prefix 首个 token
            # 污染，logits 变成随输入漂移（CPU 正确、GPU 错，实测）。
            seq_const = torch.zeros(1, n_seq, d_model)
            seq_const[0, :n_prefix] = pe_const
            seq_const[0, n_prefix] = bos_vec
            self.register_buffer('seq_const', seq_const)
            self.register_buffer('seq_tail_zeros',
                                 torch.zeros(1, n_seq - n_prefix, d_model))
            # 取 BOS 那一行用矩阵乘，不用切片。这个文件本来就在系统性回避
            # SLICE/GATHER；排除法走到最后，这是唯一还没回避的一处。
            row_sel = torch.zeros(1, 1, n_seq)
            row_sel[0, 0, n_prefix] = 1.0
            self.register_buffer('bos_row_sel', row_sel)
            # (n_seq, n_seq) 常量 mask：prefix 内部全连（训练时就是双向），
            # prefix 看不到 BOS，BOS 能看到全部。用 -30000 而不是 -inf：取行走
            # 的是加法，-inf 会在 fp16 里溢出，且和 0 系数相乘出 NaN。
            MASK_NEG = -30000.0
            mask = torch.zeros(n_seq, n_seq)
            mask[:n_prefix, n_prefix:] = MASK_NEG   # prefix 看不到 BOS 及以后
            mask[:, n_real:] = MASK_NEG             # 补齐位对谁都不可见
            # 显式展开到 nhead 份，不让运行时在 head 维上做广播。
            # decode 那边的 mask 是 (1,1,1,max_kv)，只在 query 维广播，实测没问题；
            # 这里是 (1,1,n_seq,n_seq) 要在 head 维广播，是这张图独有的新模式，
            # 而 GPU 上的表现和 CPU 不一致。常量多占 nhead×n_seq² 个 fp16，17 KB。
            self.register_buffer(
                'attn_mask_const',
                mask.view(1, 1, n_seq, n_seq).expand(1, nhead, n_seq, n_seq).contiguous())

    def forward(self, prefix_embeds):
        # 通过 +0*prefix_embeds 强制把常量物化为 runtime tensor，阻止 converter
        # 把 (1, N, D) 折回 (N, D) 触发 ADD broadcast 拒绝
        if self.fuse_bos:
            # 先把 prefix 补零到 n_seq，再整体加常量。cat(运行时张量, 常量) 与
            # (运行时张量 + 常量) 都是这张图里已经在用、GPU 上验证过的模式。
            h = torch.cat([prefix_embeds, self.seq_tail_zeros], dim=1)
            h = self.drop(h + self.seq_const)
            attn_mask = self.attn_mask_const
        else:
            pe = self.prefix_pos_embed + 0 * prefix_embeds
            h = self.drop(prefix_embeds + pe)
            attn_mask = None

        # 每层独立 pad k/v 到 max_kv：3D cat（decoder 注定落 XNNPACK，不再为 GPU 做 4D 升降维）
        k_padded_list = []
        v_padded_list = []
        for layer in self.layer_list:
            h_norm = _layer_norm(h, layer.norm1)
            qkv = F.linear(h_norm, layer.self_attn.in_proj_weight,
                           layer.self_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            # 只有前 n_real 个位置是真的，补齐位不能进 KV cache——decode 的因果
            # mask 只按位置截断，看不出它们是补齐来的。
            k_full = torch.cat([k[:, :self.n_real, :], self.kv_pad_zeros_one], dim=1)
            v_full = torch.cat([v[:, :self.n_real, :], self.kv_pad_zeros_one], dim=1)
            k_padded_list.append(k_full)
            v_padded_list.append(v_full)

            q_mh = q.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
            k_mh = k.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
            v_mh = v.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
            out = _manual_attention(q_mh, k_mh, v_mh, attn_mask=attn_mask)
            out = out.transpose(1, 2).reshape(1, self.n_seq, self.d_model)
            out = F.linear(out, layer.self_attn.out_proj.weight,
                           layer.self_attn.out_proj.bias)
            h = h + out

            h_norm2 = _layer_norm(h, layer.norm2)
            ff = layer.activation(F.linear(h_norm2, layer.linear1.weight,
                                           layer.linear1.bias))
            ff = F.linear(ff, layer.linear2.weight, layer.linear2.bias)
            h = h + ff

        # 逐层分开返回 k、v，不 stack 成一个大张量。
        #
        # 早先是 stack 成 (n_layers, 2, 1, max_kv, d_model) 一个张量，接口是干净
        # 了，但 step 每步都要在图里把它拆回来——实测 decode 子图里 48 个 SLICE
        # 全是这么来的，而 SLICE 不在 GPU 加速器的支持之列，63 个交不出去的算子
        # 基本就是它们，图因此被切成多个分区。分开传之后图里不再有拆分，每个
        # 张量也能各自做输出回灌。
        if not self.fuse_bos:
            return tuple(k_padded_list) + tuple(v_padded_list)
        # BOS 那一位的输出就是第一个 token 的 logits，logits 排在最前面，
        # 后面依旧是逐层的 k、v。
        last = torch.matmul(self.bos_row_sel, h).reshape(1, self.d_model)
        logits = self.head(_layer_norm(last, self.norm)) + self.logit_bias
        return (logits,) + tuple(k_padded_list) + tuple(v_padded_list)


# ── Decode Step 模块（GPU 友好版）────────────────────────────────────────

class DecoderStepModule(nn.Module):
    """Decode 单步。输入全 int32；KV 写入用 onehot 浮点混合；causal mask 用查表。"""

    def __init__(self, decoder, logit_bias, n_layers, d_model, nhead, max_kv):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_kv = max_kv

        # 词表大小、位置表长度（onehot lookup 需要常量类别数）
        self.vocab_size = decoder.tok_embed.num_embeddings
        self.max_pos = decoder.pos_embed.num_embeddings

        self.register_buffer('tok_embed_weight',
                             decoder.tok_embed.weight.detach().clone())  # (vocab, d_model)
        self.register_buffer('pos_embed_weight',
                             decoder.pos_embed.weight.detach().clone())  # (max_pos, d_model)
        self.drop = decoder.drop
        self.layer_list = decoder.layers.layers
        self.norm = decoder.norm
        self.head = decoder.head
        self.register_buffer('logit_bias', logit_bias)

        # 因果 mask 查表：对每个绝对位置 p，行 mask[p, k] = 0 if k<=p else 大负数。
        # 形状 (max_pos, max_kv)，trace 时是常量；运行时通过 onehot(pos) @ table 取行。
        #
        # 用有限大负数而不是 -inf：取行走的是矩阵乘法，没被选中的那些行会以
        # 系数 0 参与求和，而 0 × (-inf) = NaN——整行 mask 就成了 NaN，softmax
        # 之后 logits 全是 NaN。
        #
        # 取 -30000 而不是 -1e9：这张表会被 fp16 权重量化覆盖到，而 fp16 的上限
        # 是 65504，-1e9 会溢出成 -inf，等于把上面那个坑又踩回来。-30000 在
        # fp16 里精确可表示，softmax 里同样压到 0。
        MASK_NEG = -30000.0
        idx = torch.arange(max_kv).unsqueeze(0)            # (1, max_kv)
        pos_range = torch.arange(self.max_pos).unsqueeze(1)  # (max_pos, 1)
        causal_mask = torch.where(idx <= pos_range,
                                  torch.zeros(1),
                                  torch.full((1,), MASK_NEG))
        self.register_buffer('causal_mask_table', causal_mask)  # (max_pos, max_kv)

        # 槽位 onehot 查表：行 slot_table[p] = onehot(p, max_kv)，形状 (max_pos, max_kv)。
        slot_table = torch.zeros(self.max_pos, max_kv)
        for p in range(self.max_pos):
            if p < max_kv:
                slot_table[p, p] = 1.0
        self.register_buffer('slot_onehot_table', slot_table)  # (max_pos, max_kv)

    def _layer_step(self, x, past_k, past_v, slot_onehot, attn_mask, layer):
        """slot_onehot: (1, max_kv, 1) fp32；attn_mask: (1, 1, 1, max_kv) fp32。"""
        x_norm = _layer_norm(x, layer.norm1)
        qkv = F.linear(x_norm, layer.self_attn.in_proj_weight,
                       layer.self_attn.in_proj_bias)
        q, k_new, v_new = qkv.chunk(3, dim=-1)  # 各 (1, 1, d)

        # KV 槽位写入：直接叠加，不必先把原位清零。
        #
        # 目标槽位写入前必定是 0：prefill 每轮把整块 cache 重填一遍（有效位加
        # 零填充），decode 每步写一个全新位置，一轮之内同一个槽位只写一次。
        # 所以 `past*(1-onehot)` 这项恒等于 past，省掉一次 SUB 和一次全尺寸
        # MUL——每层 k/v 各少一个算子，decode 子图少 32 个。
        # past_k: (1, max_kv, d)；k_new: (1, 1, d) 乘 onehot 后自然广播。
        k_full = past_k + k_new * slot_onehot                   # (1, max_kv, d)
        v_full = past_v + v_new * slot_onehot

        q_mh = q.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
        k_mh = k_full.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)
        v_mh = v_full.unflatten(-1, (self.nhead, self.head_dim)).transpose(1, 2)

        out = _manual_attention(q_mh, k_mh, v_mh, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(1, 1, self.d_model)
        out = F.linear(out, layer.self_attn.out_proj.weight,
                       layer.self_attn.out_proj.bias)
        x = x + out

        x_norm2 = _layer_norm(x, layer.norm2)
        ff = layer.activation(F.linear(x_norm2, layer.linear1.weight,
                                       layer.linear1.bias))
        ff = F.linear(ff, layer.linear2.weight, layer.linear2.bias)
        x = x + ff
        return x, k_full, v_full

    def forward(self, token_id, pos, *past_kv):
        """
        token_id (1, 1) fp32；pos (1,) fp32；past_kv 为 2·n_layers 个
        (1, max_kv, d_model) fp32 张量，先 n_layers 个 k，再 n_layers 个 v。

        token 与 pos 用 fp32 而不是 int32：整数值在 fp32/fp16 里都精确，而收
        int32 会在图里留下 CAST，见 _make_onehot 的说明。
        """
        # ── 1. token / pos embedding：onehot @ weight，避免 GATHER + INT64 cast
        tok_vec = _onehot_lookup(token_id, self.tok_embed_weight, self.vocab_size)  # (1,1,d)
        pos_vec = _onehot_lookup(pos.unsqueeze(-1), self.pos_embed_weight, self.max_pos)  # (1,1,d)
        x = self.drop(tok_vec + pos_vec)

        # ── 2. 因果 mask & 槽位 onehot：用 onehot(pos) @ table 取行（matmul，无 GREATER_EQUAL）
        pos_onehot = _make_onehot(pos, self.max_pos, x.dtype, x.device)  # (1, max_pos)
        causal_row = pos_onehot @ self.causal_mask_table        # (1, max_kv)
        attn_mask = causal_row.view(1, 1, 1, self.max_kv)       # 显式 4D，避免广播报错

        slot_row = pos_onehot @ self.slot_onehot_table          # (1, max_kv)
        slot_onehot = slot_row.view(1, self.max_kv, 1)          # (1, max_kv, 1)

        # ── 3. 逐层
        n = self.n_layers
        new_k, new_v = [], []
        for i in range(self.n_layers):
            past_k = past_kv[i]
            past_v = past_kv[n + i]
            x, k_full, v_full = self._layer_step(
                x, past_k, past_v, slot_onehot, attn_mask, self.layer_list[i])
            new_k.append(k_full)
            new_v.append(v_full)


        # head 取最后一个 token：固定形状 (1,1,d) → (1,d)，用 reshape 而非切片
        last = x.reshape(1, self.d_model)
        logits = self.head(_layer_norm(last, self.norm)) + self.logit_bias
        return (logits,) + tuple(new_k) + tuple(new_v)


# ── logit bias（与原脚本一致）────────────────────────────────────────────

def _build_logit_bias(tokens):
    bias = torch.zeros(len(tokens))
    boost = set('0123456789+-=.,;()[]{}/')
    penalize = {'^': -2.0, '_': -1.0}
    for i, tok in enumerate(tokens):
        if tok in boost:
            bias[i] = 0.3
        elif tok in penalize:
            bias[i] = penalize[tok]
    return bias


# ── 主流程 ─────────────────────────────────────────────────────────────────

def _patch_quantizer_tensor_names():
    """让 ai_edge_quantizer 能处理 tensor 名是 str 的模型。

    它的多处变换都写成 `tensor.name + b'_dequant'` 这类形式，默认名字是
    bytes（从 flatbuffer 解析出来时确实如此）。但 litert_torch 的多签名转换
    产出的模型里名字是 str，于是 decoder 的 fp16 量化在 dequant_insert 处抛
    `can only concatenate str (not "bytes") to str`；单签名的 prefix_enc 不走
    这条路，所以只有 decoder 会炸。0.6.0 与 0.9.0 都是这样，是上游 bug。

    补在每个变换函数的入口：把它这次要用的那个张量的名字转成 bytes，再交给
    原函数。不能提前批量改整张图——出问题的张量是在变换过程中新建的。
    """
    from ai_edge_quantizer.transformations import dequant_insert, quant_insert

    for module, fn_name in ((dequant_insert, 'insert_dequant'),
                            (quant_insert, 'insert_quant')):
        original = getattr(module, fn_name)
        if getattr(original, '_mwh_name_patch', False):
            continue

        def make(original):
            def patched(transformation_input):
                tensors = transformation_input.subgraph.tensors
                tensor = tensors[transformation_input.tensor_id]
                if isinstance(tensor.name, str):
                    tensor.name = tensor.name.encode('utf-8')
                return original(transformation_input)
            patched._mwh_name_patch = True
            return patched

        setattr(module, fn_name, make(original))


def _move_layernorm_scale_to_activation(path):
    """把 `权重 × γ` 改写成 `激活 × γ`，让权重能被共享、也能被量化。

    litert_torch 把 LayerNorm 的 γ 折进了后一层权重，但这次乘法留在运行时：

        MUL(W[out,in], γ[in]) -> Wγ ;  FULLY_CONNECTED(x, Wγ, b)

    两个坏处：prefill 每次推理都要重算十几个大矩阵；而且多出来的这个 MUL
    消费者会挡住 fp16 权重量化——量化器只能改 FULLY_CONNECTED 那条边，fp32
    原件还被 MUL 引用着，只能留下，于是 fp16 副本是加上去而不是换上去，模型
    反而变大（实测 97.7 MB → 133 MB）。

    直接把 MUL 折成常量也不行：W 本来是 prefill 与 decode 共享的，折出来的
    Wγ 是 prefill 独有的，会多复制一份（97.7 → 75 MB，只省了一半该省的）。

    按 (W·γ)[o,i] = W[o,i]·γ[i]：

        x @ (W·γ)^T = Σ_i x[i]·W[o,i]·γ[i] = (x·γ) @ W^T

    所以把 γ 乘到激活上完全等价。W 继续共享，量化比这才到 0.50；乘法规模也
    从 [out,in] 降到激活大小（1536×512 → 32×512）。

    `lightweight_conversion` 与 `runtime_constant_folding` 都折不掉这个 MUL，
    所以在 flatbuffer 上改。
    """
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema

    raw = path.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))
    opcode = [c.builtinCode for c in model.operatorCodes]
    MUL = schema.BuiltinOperator.MUL
    FC = schema.BuiltinOperator.FULLY_CONNECTED

    def is_const(t):
        if not t.buffer:
            return False
        b = model.buffers[t.buffer]
        if b.data is not None:
            return len(b.data) > 0
        return bool(getattr(b, 'offset', 0) and getattr(b, 'size', 0))

    moved = 0
    for sg in model.subgraphs:
        produced = {op.outputs[0]: i for i, op in enumerate(sg.operators)
                    if opcode[op.opcodeIndex] == MUL and len(op.inputs) == 2}

        for op in sg.operators:
            if opcode[op.opcodeIndex] != FC or len(op.inputs) < 2:
                continue
            if op.inputs[1] not in produced:
                continue
            mul_idx = produced[op.inputs[1]]
            mul = sg.operators[mul_idx]
            ta, tb = sg.tensors[mul.inputs[0]], sg.tensors[mul.inputs[1]]
            if not (is_const(ta) and is_const(tb)):
                continue
            sa = [int(x) for x in (ta.shape if ta.shape is not None else [])]
            sb = [int(x) for x in (tb.shape if tb.shape is not None else [])]
            if len(sa) == 2 and len(sb) == 1:
                w, gamma = mul.inputs[0], mul.inputs[1]
            elif len(sb) == 2 and len(sa) == 1:
                w, gamma = mul.inputs[1], mul.inputs[0]
            else:
                continue

            act = sg.tensors[op.inputs[0]]
            scaled = schema.TensorT()
            scaled.shape, scaled.type, scaled.buffer = act.shape, act.type, 0
            nm = act.name if isinstance(act.name, bytes) else str(act.name).encode()
            scaled.name = nm + b'_gamma'
            sg.tensors.append(scaled)

            new_mul = schema.OperatorT()
            new_mul.opcodeIndex = mul.opcodeIndex
            new_mul.inputs = [int(op.inputs[0]), int(gamma)]
            new_mul.outputs = [len(sg.tensors) - 1]
            new_mul.builtinOptionsType = mul.builtinOptionsType
            new_mul.builtinOptions = mul.builtinOptions

            # flatbuffer 解出来的 inputs 是只读 ndarray，换成 list 再改。
            ins = [int(x) for x in op.inputs]
            ins[0], ins[1] = len(sg.tensors) - 1, int(w)
            op.inputs = ins
            sg.operators[mul_idx] = new_mul
            moved += 1

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b'TFL3')
    path.write_bytes(bytes(builder.Output()))
    return moved


def _quantize_fp16(path):
    """把权重量化成 fp16。

    用 ai_edge_quantizer 的 float_casting 算法覆盖全部支持的算子
    （FULLY_CONNECTED / CONV_2D / DEPTHWISE_CONV_2D / CONV_2D_TRANSPOSE /
    EMBEDDING_LOOKUP）。注意因果 mask 表也会被覆盖到，所以它的填充值必须在
    fp16 范围内——见上面 MASK_NEG 的注释。
    """
    import shutil, tempfile
    from ai_edge_quantizer import quantizer

    _patch_quantizer_tensor_names()
    recipe = [{
        'regex': '.*', 'operation': '*', 'algorithm_key': 'float_casting',
        'op_config': {
            'weight_tensor_config': {'num_bits': 16, 'symmetric': True,
                                     'granularity': 'TENSORWISE', 'dtype': 'FLOAT'},
            'compute_precision': 'FLOAT', 'explicit_dequantize': False,
            'skip_checks': False, 'min_weight_elements': 0,
        }}]
    qt = quantizer.Quantizer(str(path))
    qt.load_quantization_recipe(recipe)
    with tempfile.TemporaryDirectory() as tmp:
        qt.quantize().export_model(f'{tmp}/{path.name}')
        shutil.move(f'{tmp}/{path.name}', str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',     required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out',      default='export_v3_tflite_android')
    ap.add_argument('--max-decode', type=int, default=32)
    ap.add_argument('--fuse-bos', action='store_true',
                    help='让 prefill 顺带算 BOS 首步、直接出第一个 logits。'
                         '默认关：这条路在 Adreno/ml_drift 上算不对——同一个 '
                         'flatbuffer，CPU 与 PyTorch 逐位一致，GPU 的 BOS 行却'
                         '随输入漂移，KV 也跟着错。已排除 mask 形状、序列对齐、'
                         'fp16 量化、γ 移位改写、常量物化技巧、SLICE 取行六项，'
                         '未定位到根因。')
    ap.add_argument('--ln-calib-samples',
                    default='air_calculator/assets/benchmark_samples.jsonl',
                    help='用于 LayerNorm 定标校准的 jsonl；置空则全部保留 row-max')
    ap.add_argument('--ln-calib-n', type=int, default=500,
                    help='校准样本条数。样本越多，观测到的 var 范围越接近真实'
                         '分布，常数才能摆在窗口正中间。用 40 条时最险的那个'
                         'LayerNorm 在 500 条上只剩 1.1 倍裕度。')
    ap.add_argument('--fp16', action='store_true',
                    help='权重导出为 fp16：三件套 187 MB → 52 MB，输出逐 token 不变')
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'加载 checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {})

    d_model      = ckpt_args.get('d_model',      512)
    n_layers     = ckpt_args.get('n_layers',     8)
    nhead        = ckpt_args.get('nhead',        8)
    max_src      = ckpt_args.get('max_src',      512)
    n_stroke_tok = ckpt_args.get('n_stroke_tok', 32)
    modality     = ckpt_args.get('modality', 'both')
    n_img_tok    = 64 if modality in ('both', 'image_only') else 0
    n_stroke_tok = n_stroke_tok if modality in ('both', 'stroke_only') else 0
    n_prefix     = n_img_tok + n_stroke_tok
    max_kv       = n_prefix + args.max_decode

    print(f'[Config] d_model={d_model} n_layers={n_layers} nhead={nhead} '
          f'n_prefix={n_prefix} max_src={max_src} max_kv={max_kv}')

    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import Vocabulary
    vocab = Vocabulary.load(data_dir / 'vocab.json')
    vocab_size = len(vocab)
    max_seq = ckpt['decoder']['pos_embed.weight'].shape[0]
    print(f'[Vocab] size={vocab_size} max_seq={max_seq}')

    from models import CausalTransformerDecoder
    decoder = CausalTransformerDecoder(
        vocab_size=vocab_size, d_model=d_model,
        nhead=nhead, n_layers=n_layers, max_seq=max_seq,
    ).eval()
    decoder.load_state_dict(ckpt['decoder'])
    logit_bias = _build_logit_bias(vocab.idx2token)

    import litert_torch

    # ================================================================
    # 1. prefix_enc.tflite  (encoder 不动，原脚本本身就能在 GPU 跑 35/790
    #    算子。如未来想优化可以同样 onehot 化 pos_embed)
    # ================================================================
    print('\n' + '=' * 60)
    print('1/2 导出 prefix_enc')
    print('=' * 60)

    from models import VisualPrefixEncoder
    from stroke_renderer import IMG_H, IMG_W

    prefix_enc_base = VisualPrefixEncoder(
        d_model=d_model, n_stroke=n_stroke_tok, modality=modality,
    ).eval()
    prefix_enc_base.load_state_dict(ckpt['prefix_enc'])
    # 校准必须赶在任何一次 convert 之前：常数挂在 nn.LayerNorm 模块上，
    # 而 encoder 在这里就要导出，decoder 的两个子模块也共用同一批 LayerNorm。
    if args.ln_calib_samples:
        print('\n[LayerNorm 定标] 校准中...')
        calib_rows = [json.loads(l) for l
                      in open(args.ln_calib_samples, encoding='utf-8') if l.strip()]
        calib_rows = [r for r in calib_rows if not r.get('__meta__')][:args.ln_calib_n]
        calibrate_ln_scales(decoder, prefix_enc_base, calib_rows,
                            stroke_len=max_src, vocab=vocab,
                            max_decode=args.max_decode)
    else:
        print('\n[LayerNorm 定标] 未提供校准样本，全部保留 row-max')

    prefix_mod = PrefixEncUnrolledModule(prefix_enc_base, stroke_len=max_src).eval()

    dummy_img    = torch.zeros(1, 1, IMG_H, IMG_W)
    dummy_stroke = torch.zeros(1, max_src, 13)
    # real_len 走 f32，见 PrefixEncUnrolledModule.forward 里的说明
    dummy_len    = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        out = prefix_mod(dummy_img, dummy_stroke, dummy_len)
    print(f'  prefix_enc output: {out.shape}  expected: (1, {n_prefix}, {d_model})  ✓')

    def _fp16_recipe():
        from litert_torch.generative.quantize import quant_recipes
        _patch_quantizer_tensor_names()
        return quant_recipes.full_fp16_recipe()

    enc_quant = _fp16_recipe() if args.fp16 else None
    print(f'  转换中...（encoder 权重 {"fp16" if args.fp16 else "fp32"}）')
    prefix_edge = litert_torch.convert(
        prefix_mod,
        sample_args=(dummy_img, dummy_stroke, dummy_len),
        quant_config=enc_quant,
    )
    prefix_path = out_dir / 'prefix_enc.tflite'
    prefix_edge.export(str(prefix_path))
    prefix_size_mb = prefix_path.stat().st_size / (1024 * 1024)
    print(f'  已保存: {prefix_path}  ({prefix_size_mb:.1f} MB)')

    # ================================================================
    # 2. decoder.tflite（GPU 友好双签名）
    # ================================================================
    print('\n' + '=' * 60)
    print('2/2 导出 decoder（Android GPU 友好版）')
    print('=' * 60)

    prefill_mod = DecoderPrefillModule(
        decoder, n_prefix, max_kv, n_layers, nhead, d_model,
        bos_id=(vocab.BOS if args.fuse_bos else None),
        logit_bias=(logit_bias if args.fuse_bos else None),
    ).eval()
    step_mod    = DecoderStepModule(decoder, logit_bias, n_layers, d_model, nhead, max_kv).eval()

    prefix_ex  = torch.zeros(1, n_prefix, d_model)
    token_ex   = torch.zeros(1, 1, dtype=torch.float32)
    pos_ex     = torch.tensor([float(n_prefix)], dtype=torch.float32)
    # KV 逐层分开：先 n_layers 个 k，再 n_layers 个 v，每个 (1, max_kv, d_model)。
    past_kv_ex = tuple(torch.zeros(1, max_kv, d_model) for _ in range(2 * n_layers))

    with torch.no_grad():
        kv0 = prefill_mod(prefix_ex)
        step_out = step_mod(token_ex, pos_ex, *past_kv_ex)
    logits0, kv1 = step_out[0], step_out[1:]
    # 融合了 BOS 的 prefill 第一个输出是 logits，KV 从第二个起。
    pre_off = 1 if args.fuse_bos else 0
    if pre_off:
        assert kv0[0].shape == (1, vocab_size), kv0[0].shape
        print(f'  prefill logits: {tuple(kv0[0].shape)}  ✓（已融合 BOS 首步）')
        kv0 = kv0[1:]
    assert len(kv0) == 2 * n_layers and kv0[0].shape == (1, max_kv, d_model)
    assert logits0.shape == (1, vocab_size)
    assert len(kv1) == 2 * n_layers and kv1[0].shape == (1, max_kv, d_model)
    print(f'  prefill output: {len(kv0)} 个 {tuple(kv0[0].shape)}  ✓')
    print(f'  step    logits: {tuple(logits0.shape)}  ✓')
    print(f'  step    new_kv: {len(kv1)} 个 {tuple(kv1[0].shape)}  ✓')

    # ── Sanity check：与原始 decoder 数值是否一致（防止改写引入精度漂移）
    print('  对照原始 decoder 检查数值等价性...')
    with torch.no_grad():
        ref_x = decoder.tok_embed(token_ex.long()) + decoder.pos_embed(pos_ex.long().view(1, 1))
        # 仅做 embedding 层快速比对（mask/KV 路径无解析参考）
        rebuilt_x = (
            _onehot_lookup(token_ex, step_mod.tok_embed_weight, step_mod.vocab_size) +
            _onehot_lookup(pos_ex.unsqueeze(-1), step_mod.pos_embed_weight, step_mod.max_pos)
        )
        max_err = (ref_x - rebuilt_x).abs().max().item()
        print(f'  embedding 最大误差: {max_err:.2e}  (应 < 1e-4)')
        assert max_err < 1e-4, f'embedding 重写后数值漂移 {max_err}'

    print('  转换中（litert_torch 多签名）...')
    converter = litert_torch.signature(
        'prefill', prefill_mod, sample_args=(prefix_ex,),
    )
    converter.signature(
        'decode', step_mod, sample_args=(token_ex, pos_ex) + past_kv_ex,
    )
    # 下面这些算子改写做完之后，decoder 两个签名都能整图下沉到 GPU
    # （prefill 455/455、decode 484/484，各 1 个分区）；改之前只有不到一成算子
    # 接得住，跨边界 sync 反而更慢。重新打开 lightweight_conversion 以折叠常量、
    # 压缩文件体积。
    decoder_edge = converter.convert(
        lightweight_conversion=True,
        strict_export='auto',
    )
    decoder_path = out_dir / 'decoder.tflite'
    decoder_edge.export(str(decoder_path))
    if args.fp16:
        # decoder 不在转换期量化：litert_torch 那条 generative 配方按 LLM 的层
        # 角色匹配，这张图上只认出 29 个张量。走导出后处理，见下面两个函数。
        n = _move_layernorm_scale_to_activation(decoder_path)
        print(f'  γ 移到激活：改写 {n} 处')
        _quantize_fp16(decoder_path)

    decoder_size_mb = decoder_path.stat().st_size / (1024 * 1024)
    print(f'  已保存: {decoder_path}  ({decoder_size_mb:.1f} MB)')

    # ================================================================
    # 3. vocab.json
    # ================================================================
    out_vocab = out_dir / 'vocab.json'
    meta = {
        'tokens':  vocab.idx2token,
        'bos_idx': vocab.BOS,
        'eos_idx': vocab.EOS,
        'pad_idx': vocab.PAD,
        'max_kv':  max_kv,
        'n_prefix': n_prefix,
    }
    out_vocab.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[Vocab] 已保存: {out_vocab}  ({vocab_size} tokens, max_kv={max_kv})')

    print('\n' + '=' * 60)
    print('Android GPU 友好版导出完成！')
    total_mb = prefix_size_mb + decoder_size_mb
    print(f'  {prefix_path.name}  ({prefix_size_mb:.1f} MB)')
    print(f'  {decoder_path.name}  ({decoder_size_mb:.1f} MB)')
    print(f'  vocab.json')
    print(f'  合计: {total_mb:.1f} MB')
    print('=' * 60)
    print()
    print('部署：把 export_v3_tflite_android/*.tflite 拷到 platform_models/android/，')
    print('     重新跑 tool/copy_platform_models.sh，安卓 backend 选 gpu 即可。')
    print()
    print('警告：onehot @ weight 替代 embedding，会增加 max_pos×d_model 的常量乘加；')
    print('     vocab=230 / max_pos=128 时常量开销很小，但首层 token onehot 会膨胀')
    print('     vocab_size 维度，CPU 上反而更慢。该脚本仅用于 Android GPU 路径。')


if __name__ == '__main__':
    main()
