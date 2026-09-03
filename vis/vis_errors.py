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
错误可视化 Streamlit App (v3 checkpoint)

运行:
  cd air_calculator_py/vis
  MATHWRITING_DIR=../../dataset/mathwriting-2024 \
  streamlit run vis_errors.py --server.port 8501 --server.address 0.0.0.0
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))

import numpy as np
import torch
import streamlit as st
from PIL import Image

from dataset import MathWritingDataset, get_vocabulary, DATA_DIR, collate_fn
from models import CausalTransformerDecoder, VisualPrefixEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DEFAULT = str(Path(__file__).resolve().parent.parent.parent / 'weights' / 'v3_final' / 'v3_best_epoch104_acc0.7720.pt')
DATA_DIR_DEFAULT = os.environ.get('MATHWRITING_DIR', str(DATA_DIR))
STRUCT_TOKENS = set('{}_^')


# ── edit distance ─────────────────────────────────────────────── #

def edit_ops(gt_toks, pred_toks):
    m, n = len(gt_toks), len(pred_toks)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if gt_toks[i-1] == pred_toks[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt_toks[i-1] == pred_toks[j-1]:
            ops.append(('match', gt_toks[i-1], pred_toks[j-1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1]+1:
            ops.append(('sub', gt_toks[i-1], pred_toks[j-1])); i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1]+1:
            ops.append(('ins', None, pred_toks[j-1])); j -= 1
        else:
            ops.append(('del', gt_toks[i-1], None)); i -= 1
    ops.reverse()
    return ops


def error_category(gt_toks, pred_toks):
    if gt_toks == pred_toks:
        return 'correct'
    if [t.lower() for t in gt_toks] == [t.lower() for t in pred_toks]:
        return 'case_mismatch'
    ops = edit_ops(gt_toks, pred_toks)
    subs = [(g,p) for op,g,p in ops if op=='sub']
    dels = [g for op,g,p in ops if op=='del']
    ins  = [p for op,g,p in ops if op=='ins']
    all_err = set(subs and [x for pair in subs for x in pair]) | set(dels) | set(ins)
    n_struct = (sum(1 for g,p in subs if g in STRUCT_TOKENS or p in STRUCT_TOKENS)
              + sum(1 for g in dels if g in STRUCT_TOKENS)
              + sum(1 for p in ins  if p in STRUCT_TOKENS))
    n_total = len(subs)+len(dels)+len(ins)
    if n_total > 0 and all_err <= STRUCT_TOKENS:
        return 'pure_struct'
    if n_struct > 0:
        return 'mixed_struct'
    visual = {('\\nu','v'),('v','\\nu'),('d','\\partial'),('\\partial','d'),
              ('\\tilde','\\overline'),('\\overline','\\tilde'),
              ('\\omega','w'),('w','\\omega'),('\\phi','\\Phi'),('\\Phi','\\phi')}
    if any((g,p) in visual for g,p in subs):
        return 'visual_similar'
    decorators = {'\\hat','\\dot','\\tilde','\\overline','\\bar','\\prime','\\vec'}
    if any(g in decorators for g in dels):
        return 'decorator_lost'
    if any(g.lower()==p.lower() and g!=p for g,p in subs if g not in STRUCT_TOKENS):
        return 'case_error'
    if dels and not ins and not subs:
        return 'pure_deletion'
    if ins and not dels and not subs:
        return 'pure_insertion'
    return 'other'


def format_diff(ops):
    parts = []
    for op, g, p in ops:
        if op == 'match':
            parts.append(f'<span style="color:gray">{g}</span>')
        elif op == 'sub':
            parts.append(f'<span style="color:red;text-decoration:line-through">{g}</span>'
                         f'<span style="color:green">{p}</span>')
        elif op == 'del':
            parts.append(f'<span style="color:red;text-decoration:line-through">{g}</span>')
        elif op == 'ins':
            parts.append(f'<span style="color:blue">[+{p}]</span>')
    return ' '.join(parts)


# ── 加载模型（缓存）────────────────────────────────────────────── #

@st.cache_resource
def load_model(ckpt_path, data_dir):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    ca = ckpt.get('args', {})
    d_model  = ca.get('d_model', 512)
    n_layers = ca.get('n_layers', 8)
    nhead    = ca.get('nhead', 8)
    max_tgt  = ca.get('max_tgt', 64)
    max_src  = ca.get('max_src', 512)
    max_seq  = ckpt['decoder']['pos_embed.weight'].shape[0]

    vocab = get_vocabulary(Path(data_dir))
    prefix_enc = VisualPrefixEncoder(d_model=d_model).to(DEVICE)
    decoder    = CausalTransformerDecoder(
        vocab_size=len(vocab), d_model=d_model,
        nhead=nhead, n_layers=n_layers, max_seq=max_seq,
    ).to(DEVICE)
    prefix_enc.load_state_dict(ckpt['prefix_enc'])
    decoder.load_state_dict(ckpt['decoder'])
    prefix_enc.eval(); decoder.eval()
    return vocab, prefix_enc, decoder, max_tgt, max_src


@st.cache_data
def load_dataset(data_dir, max_src, max_tgt, n_samples):
    vocab, _, _, _, _ = load_model(CKPT_DEFAULT, data_dir)
    ds = MathWritingDataset(split='valid', vocab=vocab,
                            max_src=max_src, max_tgt=max_tgt,
                            data_dir=Path(data_dir))
    n = min(n_samples, len(ds))
    return torch.utils.data.Subset(ds, list(range(n)))


# ── 推理（缓存）──────────────────────────────────────────────────── #

def run_inference(data_dir, n_samples, beam):
    vocab, prefix_enc, decoder, max_tgt, max_src = load_model(CKPT_DEFAULT, data_dir)
    from torch.utils.data import DataLoader
    ds = MathWritingDataset(split='valid', vocab=vocab,
                            max_src=max_src, max_tgt=max_tgt,
                            data_dir=Path(data_dir))
    n = min(n_samples, len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n)))
    loader = DataLoader(subset, batch_size=32, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    results = []  # (img_np, gt_str, pred_str, category)
    with torch.no_grad():
        for stroke, stroke_mask, imgs, ids_list in loader:
            stroke = stroke.to(DEVICE)
            stroke_mask = stroke_mask.to(DEVICE)
            imgs = imgs.to(DEVICE)
            prefix = prefix_enc(imgs, stroke, stroke_mask)
            for i in range(prefix.size(0)):
                pfx = prefix[i:i+1]
                if beam > 1:
                    pred_ids = decoder.generate_beam(pfx, vocab.BOS, vocab.EOS,
                                                     max_len=max_tgt, beam_width=beam)
                else:
                    pred_ids = decoder.generate(pfx, vocab.BOS, vocab.EOS, max_len=max_tgt)
                gt_toks   = vocab.decode(ids_list[i].tolist())
                pred_toks = vocab.decode(pred_ids)
                gt_str    = ''.join(gt_toks)
                pred_str  = ''.join(pred_toks)
                cat = error_category(gt_toks, pred_toks)
                # 保存图像
                img_np = imgs[i].cpu().numpy()[0]  # (H, W)
                results.append({
                    'img': img_np,
                    'gt': gt_str,
                    'pred': pred_str,
                    'gt_toks': gt_toks,
                    'pred_toks': pred_toks,
                    'category': cat,
                })
    return results


# ── Streamlit UI ──────────────────────────────────────────────── #

st.set_page_config(page_title='错误可视化', layout='wide')
st.title('手写公式识别错误可视化')

with st.sidebar:
    st.header('设置')
    data_dir = st.text_input('数据目录', DATA_DIR_DEFAULT)
    n_samples = st.slider('评估样本数', 100, 3000, 500, step=100)
    beam = st.radio('Beam size', [1, 4], index=0)
    run_btn = st.button('运行推理', type='primary')

    categories = ['all_errors', 'correct', 'pure_struct', 'mixed_struct',
                  'visual_similar', 'case_mismatch', 'case_error',
                  'decorator_lost', 'pure_deletion', 'pure_insertion', 'other']
    sel_cat = st.selectbox('筛选类别', categories)
    page_size = st.slider('每页显示', 5, 50, 20)

cache_key = f'{data_dir}_{n_samples}_{beam}'
if run_btn or 'results' not in st.session_state or st.session_state.get('cache_key') != cache_key:
    with st.spinner('推理中...'):
        st.session_state.results = run_inference(data_dir, n_samples, beam)
        st.session_state.cache_key = cache_key

results = st.session_state.results
if not results:
    st.warning('还没有结果，点击"运行推理"')
    st.stop()

# 统计
total = len(results)
correct = sum(1 for r in results if r['category'] == 'correct')
st.metric('准确率', f'{correct/total:.1%}', f'{correct}/{total}')

cat_counts = {}
for r in results:
    cat_counts[r['category']] = cat_counts.get(r['category'], 0) + 1
cols = st.columns(4)
for i, (cat, cnt) in enumerate(sorted(cat_counts.items(), key=lambda x: -x[1])):
    cols[i%4].metric(cat, cnt, f'{cnt/(total-correct)*100:.1f}%' if cat != 'correct' else '')

st.divider()

# 筛选
if sel_cat == 'all_errors':
    filtered = [r for r in results if r['category'] != 'correct']
else:
    filtered = [r for r in results if r['category'] == sel_cat]

st.write(f'**{sel_cat}**: {len(filtered)} 条')

# 翻页
total_pages = max(1, (len(filtered) - 1) // page_size + 1)
page = st.number_input('页码', 1, total_pages, 1) - 1
page_items = filtered[page*page_size:(page+1)*page_size]

for r in page_items:
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            # 显示笔画图像
            img = (r['img'] * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(img, mode='L').resize((512, 128), Image.NEAREST)
            st.image(pil, use_container_width=False)

        with c2:
            cat = r['category']
            color = {'correct':'green','pure_struct':'orange','mixed_struct':'orange',
                     'visual_similar':'purple','case_mismatch':'blue','case_error':'blue',
                     'decorator_lost':'red','pure_deletion':'red','pure_insertion':'red',
                     'other':'gray'}.get(cat, 'gray')
            st.markdown(f'**类别**: :{color}[{cat}]')
            st.markdown(f'**GT**: `{r["gt"]}`')
            st.markdown(f'**PRED**: `{r["pred"]}`')
            if r['category'] != 'correct':
                ops = edit_ops(r['gt_toks'], r['pred_toks'])
                diff_html = format_diff(ops)
                st.markdown(f'**DIFF**: {diff_html}', unsafe_allow_html=True)
