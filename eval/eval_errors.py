#!/usr/bin/env python3
"""
v3 checkpoint 错误分析（edit distance 版）

用法:
  cd air_calculator_py/eval
  MATHWRITING_DIR=../../dataset/mathwriting-2024 \
  python3 eval_errors.py \
    --ckpt ../../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
    --n-samples 0 --beam 1 --batch 64
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))

import torch
from torch.utils.data import DataLoader

from dataset import MathWritingDataset, get_vocabulary, DATA_DIR, collate_fn
from models import CausalTransformerDecoder, VisualPrefixEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STRUCT_TOKENS = set('{}_^')


# ── edit distance ─────────────────────────────────────────────── #

def edit_ops(gt_toks, pred_toks):
    m, n = len(gt_toks), len(pred_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_toks[i-1] == pred_toks[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt_toks[i-1] == pred_toks[j-1]:
            ops.append(('match', gt_toks[i-1], pred_toks[j-1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('sub', gt_toks[i-1], pred_toks[j-1])); i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(('ins', None, pred_toks[j-1])); j -= 1
        else:
            ops.append(('del', gt_toks[i-1], None)); i -= 1
    ops.reverse()
    return ops


def classify_ops(ops):
    subs = [(gt, pred) for op, gt, pred in ops if op == 'sub']
    dels = [(gt,)      for op, gt, pred in ops if op == 'del']
    ins  = [(pred,)    for op, gt, pred in ops if op == 'ins']
    return subs, dels, ins


def categorize_error(gt_toks, pred_toks, ops):
    if gt_toks == pred_toks:
        return ['correct']
    if [t.lower() for t in gt_toks] == [t.lower() for t in pred_toks]:
        return ['case_mismatch']

    subs, dels, ins = classify_ops(ops)
    cats = []

    all_error_toks = set()
    for gt, pred in subs: all_error_toks.add(gt); all_error_toks.add(pred)
    for (gt,) in dels: all_error_toks.add(gt)
    for (pred,) in ins: all_error_toks.add(pred)

    n_errors = len(subs) + len(dels) + len(ins)
    n_struct  = (sum(1 for gt, pred in subs if gt in STRUCT_TOKENS or pred in STRUCT_TOKENS)
               + sum(1 for (gt,) in dels if gt in STRUCT_TOKENS)
               + sum(1 for (pred,) in ins if pred in STRUCT_TOKENS))

    if n_errors > 0 and all_error_toks <= STRUCT_TOKENS:
        cats.append('pure_struct')
    elif n_struct > 0:
        cats.append('mixed_struct')

    visual_pairs = {
        ('\\nu','v'),('v','\\nu'),('\\nu','V'),('V','\\nu'),
        ('\\omega','w'),('w','\\omega'),('\\omega','W'),('W','\\omega'),
        ('d','\\partial'),('\\partial','d'),
        ('\\tilde','\\overline'),('\\overline','\\tilde'),
        ('\\varphi','\\rho'),('\\rho','\\varphi'),
        ('o','\\theta'),('\\theta','o'),
        ('.','\\cdot'),('\\cdot','.'),
        ('\\phi','\\Phi'),('\\Phi','\\phi'),
    }
    if any((gt, pred) in visual_pairs for gt, pred in subs):
        cats.append('visual_similar')

    n_case = sum(1 for gt, pred in subs
                 if gt.lower() == pred.lower() and gt != pred and gt not in STRUCT_TOKENS)
    if n_case > 0:
        cats.append('case_error')

    decorators = {'\\hat','\\dot','\\tilde','\\overline','\\bar',
                  '\\prime','\\vec','\\ddot','\\check','\\breve'}
    if any(gt in decorators for (gt,) in dels):
        cats.append('decorator_lost')

    if len(dels) > 0 and len(ins) == 0 and len(subs) == 0:
        cats.append('pure_deletion')
    elif len(ins) > 0 and len(dels) == 0 and len(subs) == 0:
        cats.append('pure_insertion')

    return cats or ['other']


# ── 主评估 ───────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n-samples', type=int, default=2000)
    ap.add_argument('--beam', type=int, default=1)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--data-dir', default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    ckpt_path = Path(args.ckpt)

    print(f'[Load] {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    ckpt_args = ckpt.get('args', {})
    d_model  = ckpt_args.get('d_model',  512)
    n_layers = ckpt_args.get('n_layers', 8)
    nhead    = ckpt_args.get('nhead',    8)
    max_tgt  = ckpt_args.get('max_tgt',  64)
    max_src  = ckpt_args.get('max_src',  512)
    n_stroke = ckpt_args.get('n_stroke_tok', 32)
    modality = ckpt_args.get('modality', 'both')
    print(f'[Config] d_model={d_model} n_layers={n_layers} nhead={nhead} max_tgt={max_tgt} modality={modality}')

    vocab = get_vocabulary(data_dir)
    print(f'[Vocab] size={len(vocab)}')

    val_ds = MathWritingDataset(split='valid', vocab=vocab,
                                max_src=max_src, max_tgt=max_tgt,
                                data_dir=data_dir)
    n = len(val_ds) if args.n_samples == 0 else min(args.n_samples, len(val_ds))
    subset = torch.utils.data.Subset(val_ds, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    print(f'[Eval] {n} 样本, beam={args.beam}')

    max_seq = ckpt['decoder']['pos_embed.weight'].shape[0]
    print(f'[Config] max_seq={max_seq}')

    prefix_enc = VisualPrefixEncoder(d_model=d_model, modality=modality).to(DEVICE)
    decoder    = CausalTransformerDecoder(
        vocab_size=len(vocab), d_model=d_model,
        nhead=nhead, n_layers=n_layers, max_seq=max_seq,
    ).to(DEVICE)
    prefix_enc.load_state_dict(ckpt['prefix_enc'])
    decoder.load_state_dict(ckpt['decoder'])
    prefix_enc.eval(); decoder.eval()

    total = correct = 0
    gt_total_toks = 0
    cat_counter = collections.Counter()
    sub_counter = collections.Counter()
    del_counter = collections.Counter()
    ins_counter = collections.Counter()
    gt_len_bins = collections.Counter()
    cat_samples = collections.defaultdict(list)
    MAX_PER_CAT = 20

    with torch.no_grad():
        for stroke, stroke_mask, imgs, ids_list in loader:
            stroke      = stroke.to(DEVICE)
            stroke_mask = stroke_mask.to(DEVICE)
            imgs        = imgs.to(DEVICE)
            prefix = prefix_enc(imgs, stroke, stroke_mask)

            for i in range(prefix.size(0)):
                pfx = prefix[i:i+1]
                if args.beam > 1:
                    pred_ids = decoder.generate_beam(
                        pfx, vocab.BOS, vocab.EOS,
                        max_len=max_tgt, beam_width=args.beam)
                else:
                    pred_ids = decoder.generate(
                        pfx, vocab.BOS, vocab.EOS, max_len=max_tgt)

                gt_ids    = ids_list[i].tolist()
                gt_toks   = vocab.decode(gt_ids)
                pred_toks = vocab.decode(pred_ids)
                gt_str    = ''.join(gt_toks)
                pred_str  = ''.join(pred_toks)

                total += 1
                gt_total_toks += len(gt_toks)
                gt_len_bins[(len(gt_toks) // 5) * 5] += 1

                if pred_str == gt_str:
                    correct += 1
                    continue

                ops = edit_ops(gt_toks, pred_toks)
                subs, dels, ins = classify_ops(ops)
                for gt, pred in subs: sub_counter[(gt, pred)] += 1
                for (gt,) in dels:   del_counter[gt] += 1
                for (pred,) in ins:  ins_counter[pred] += 1

                cats = categorize_error(gt_toks, pred_toks, ops)
                diff_str = ', '.join(
                    (f'{gt}→{pred}' if op=='sub' else f'-{gt}' if op=='del' else f'+{pred}')
                    for op, gt, pred in ops if op != 'match'
                )
                for c in cats:
                    cat_counter[c] += 1
                    if len(cat_samples[c]) < MAX_PER_CAT:
                        cat_samples[c].append((gt_str, pred_str, diff_str))

            print(f'\r  {total}/{n}  acc={correct/total:.4f}', end='', flush=True)

    print()
    n_err = total - correct
    acc   = correct / total

    total_edit_ops = sum(sub_counter.values()) + sum(del_counter.values()) + sum(ins_counter.values())
    cer = total_edit_ops / max(gt_total_toks, 1) * 100

    print(f'\n{"="*60}')
    print(f'评估样本: {total}  |  val_acc = {acc:.4f} ({correct}/{total})')
    print(f'GT token总数: {gt_total_toks}  |  总edit ops: {total_edit_ops}')
    print(f'CER = {cer:.2f}%  (edit_ops / gt_tokens)')
    print(f'{"="*60}')

    print(f'\n── 错误类别分布（共 {n_err} 条，edit distance）──')
    for cat, cnt in cat_counter.most_common(20):
        print(f'  {cat:<25s}  {cnt:5d}  ({cnt/n_err*100:.1f}%)')

    print(f'\n── Top-20 substitution ──')
    for (g, p), cnt in sub_counter.most_common(20):
        tag = '  [struct]' if g in STRUCT_TOKENS or p in STRUCT_TOKENS else (
              '  [case]'   if g.lower()==p.lower() else '')
        print(f'  {repr(g):<20s} → {repr(p):<20s}  {cnt:4d}{tag}')

    print(f'\n── Top-20 deletion ──')
    for tok, cnt in del_counter.most_common(20):
        print(f'  {repr(tok):<20s}  {cnt:4d}{"  [struct]" if tok in STRUCT_TOKENS else ""}')

    print(f'\n── Top-20 insertion ──')
    for tok, cnt in ins_counter.most_common(20):
        print(f'  {repr(tok):<20s}  {cnt:4d}{"  [struct]" if tok in STRUCT_TOKENS else ""}')

    struct_ops = (sum(cnt for (g,p),cnt in sub_counter.items() if g in STRUCT_TOKENS or p in STRUCT_TOKENS)
                + sum(cnt for tok,cnt in del_counter.items() if tok in STRUCT_TOKENS)
                + sum(cnt for tok,cnt in ins_counter.items() if tok in STRUCT_TOKENS))
    print(f'\n── 结构 {{_^}} vs 非结构 ──')
    print(f'  总 edit ops: {total_edit_ops}  结构: {struct_ops} ({struct_ops/total_edit_ops*100:.1f}%)  非结构: {total_edit_ops-struct_ops} ({(total_edit_ops-struct_ops)/total_edit_ops*100:.1f}%)')

    print(f'\n── GT 长度分布 ──')
    for b in sorted(gt_len_bins):
        bar = '█' * (gt_len_bins[b] * 40 // max(gt_len_bins.values()))
        print(f'  [{b:3d}-{b+4:3d}]  {gt_len_bins[b]:5d} ({gt_len_bins[b]/total*100:4.1f}%)  {bar}')

    for cat in ['pure_struct','mixed_struct','visual_similar','case_error',
                'case_mismatch','decorator_lost','pure_deletion','pure_insertion','other']:
        samples = cat_samples.get(cat, [])
        if not samples: continue
        print(f'\n── {cat} 样例（共 {cat_counter[cat]} 条，展示 {len(samples)} 条）──')
        for gt, pred, diff in samples:
            print(f'  GT  : {gt[:120]}')
            print(f'  PRED: {pred[:120]}')
            print(f'  EDIT: {diff[:160]}')
            print()


if __name__ == '__main__':
    main()
