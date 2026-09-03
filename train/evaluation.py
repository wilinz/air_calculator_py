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

"""验证集评估：贪心/beam 生成 → EM + CER。"""

import torch


@torch.no_grad()
def _edit_distance(a, b):
    """token 级 Levenshtein 距离。"""
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def evaluate(prefix_enc, decoder, vocab, loader, device,
             max_new_tokens=64, max_samples=None, beam=1):
    prefix_enc.eval()
    decoder.eval()

    correct = total = 0
    edit_ops_total = 0
    gt_tok_total = 0
    for stroke, stroke_mask, imgs, ids_list in loader:
        if max_samples is not None and total >= max_samples:
            break
        stroke = stroke.to(device)
        stroke_mask = stroke_mask.to(device)
        imgs = imgs.to(device)

        prefix = prefix_enc(imgs, stroke, stroke_mask)
        B = prefix.size(0)

        for i in range(B):
            if max_samples is not None and total >= max_samples:
                break
            p = prefix[i:i+1]

            if beam > 1:
                pred_ids = decoder.generate_beam(
                    p, vocab.BOS, vocab.EOS,
                    max_len=max_new_tokens, beam_width=beam)
            else:
                pred_ids = decoder.generate(
                    p, vocab.BOS, vocab.EOS,
                    max_len=max_new_tokens)

            pred_toks = vocab.decode(pred_ids)
            gt_ids = ids_list[i].tolist()
            gt_toks = vocab.decode(gt_ids)
            pred_str = ''.join(pred_toks)
            gt_str = ''.join(gt_toks)

            if total < 5:
                print(f'  [Sample {total}] pred={repr(pred_str[:80])}  gt={repr(gt_str[:80])}  match={pred_str == gt_str}')

            if pred_str == gt_str:
                correct += 1
            else:
                edit_ops_total += _edit_distance(gt_toks, pred_toks)
            gt_tok_total += len(gt_toks)
            total += 1

    prefix_enc.train()
    decoder.train()
    em = correct / total if total > 0 else 0.0
    cer = edit_ops_total / gt_tok_total if gt_tok_total > 0 else 0.0
    return em, cer
