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
从头训练 Decoder-only 模型：视觉前缀 + 因果 Transformer

架构:
  1. DeiT-Small 图像编码器 → 64 个 image tokens (384-dim)
  2. 笔画特征 Transformer 编码器 → 32 个 stroke tokens (256-dim)
  3. Projection → d_model (384)
  4. [img_tokens | stroke_tokens | BOS | latex_tok1 | ... | EOS]
  5. 从头训练的 Causal Transformer Decoder (6~8 层, d=384)

与 LLM 方案的关键区别:
  - 无预训练 LM 先验 → 模型无法忽略视觉输入，必须依赖 prefix 才能生成
  - 更小 (~40M)，训练快，可直接部署到移动端

运行示例:
  # 快速验证
  cd air_calculator_py/train
  MATHWRITING_DIR=../../dataset/mathwriting-2024 \
  python3 train.py --out ./decoder_only --max-samples 1000 --epochs 3

  # 正式训练（双卡 V100）
  MATHWRITING_DIR=../../dataset/mathwriting-2024 \
  nohup python3 train.py --out ./decoder_only \
    --epochs 30 --batch 64 --grad-accum 2 --lr 3e-4 \
    >> ../../logs/train_decoder_only.log 2>&1 &
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# 多 worker 下用 file_system 共享策略，避免文件描述符耗尽（Errno 24）
mp.set_sharing_strategy('file_system')

from dataset import MathWritingDataset, get_vocabulary, DATA_DIR, collate_fn
from models import VisualPrefixEncoder, CausalTransformerDecoder
from evaluation import evaluate


# ── 主训练流程 ───────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./decoder_only')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d-model', type=int, default=512)
    ap.add_argument('--n-layers', type=int, default=8)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--use-synthetic', action='store_true',
                    help='合并 synthetic/ 数据（229K+396K=625K）')
    ap.add_argument('--extra-data', nargs='*', default=[],
                    help='额外数据目录名列表（相对于 data_dir），如 targeted_round1 targeted_round2')
    ap.add_argument('--n-stroke-tok', type=int, default=32)
    ap.add_argument('--max-src', type=int, default=512)
    ap.add_argument('--max-tgt', type=int, default=64)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--eval-every', type=int, default=2,
                    help='粗 eval 频率（每 N epoch 一次小子集 eval）')
    ap.add_argument('--eval-max-samples', type=int, default=500,
                    help='粗 eval 用的样本数')
    ap.add_argument('--eval-every-large', type=int, default=10,
                    help='精 eval 频率（每 N epoch 一次大子集 eval；只有精 eval 才更新 best）')
    ap.add_argument('--eval-large-samples', type=int, default=2000,
                    help='精 eval 用的样本数')
    ap.add_argument('--eval-beam', type=int, default=1,
                    help='评估时 beam width，1=greedy')
    ap.add_argument('--save-steps', type=int, default=500)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--vocab-file', default=None,
                    help='强制指定词表 json 路径（不传则自动检测 bpe_vocab.json > vocab.json）')
    ap.add_argument('--modality', choices=['both', 'image_only', 'stroke_only'],
                    default='both',
                    help='消融用：both=双流；image_only=去笔画分支；stroke_only=去图像分支')
    ap.add_argument('--no-human-data', action='store_true',
                    help='跳过人工 train 集，只用 --extra-data 指定的数据训练')
    ap.add_argument('--num-workers', type=int, default=4,
                    help='DataLoader 进程数（数据加载是瓶颈时调大）')
    ap.add_argument('--label-smoothing', type=float, default=0.1)
    ap.add_argument('--reset-optimizer', action='store_true',
                    help='续训时只加载模型权重，重置 optimizer/scheduler（换 batch 时用）')
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = (
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else
        'cpu'
    )

    # ── 词表 ─────────────────────────────────────────────────────── #
    if args.vocab_file:
        from dataset import Vocabulary
        vocab = Vocabulary.load(Path(args.vocab_file))
        print(f'[Vocab] 强制加载: {args.vocab_file}')
    else:
        vocab = get_vocabulary(data_dir)
    vocab_size = len(vocab)
    print(f'[Vocab] size={vocab_size}  BOS={vocab.BOS}  EOS={vocab.EOS}  PAD={vocab.PAD}')

    # ── 模型 ─────────────────────────────────────────────────────── #
    n_img_tok = 64 if args.modality in ('both', 'image_only') else 0
    n_stroke_used = args.n_stroke_tok if args.modality in ('both', 'stroke_only') else 0
    n_prefix = n_img_tok + n_stroke_used
    print(f'[Modality] {args.modality}: n_prefix={n_prefix} (img={n_img_tok} stroke={n_stroke_used})')

    prefix_enc = VisualPrefixEncoder(
        d_model=args.d_model,
        d_img=384,  # DeiT-Small 原始维度
        n_img=64,
        d_stroke=256,
        n_stroke=args.n_stroke_tok,
        modality=args.modality,
    ).to(DEVICE)

    decoder = CausalTransformerDecoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        n_layers=args.n_layers,
        dropout=0.1,
        max_seq=n_prefix + args.max_tgt + 2,  # prefix + BOS + tgt + EOS
    ).to(DEVICE)

    prefix_params = sum(p.numel() for p in prefix_enc.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = prefix_params + decoder_params
    print(f'[Model] prefix_enc: {prefix_params:,}  decoder: {decoder_params:,}  total: {total_params:,}')
    print(f'[Model] d_model={args.d_model}  n_layers={args.n_layers}  nhead={args.nhead}')

    # ── 数据集 ───────────────────────────────────────────────────── #
    from torch.utils.data import ConcatDataset

    if args.no_human_data:
        print('[Data] 跳过人工数据（--no-human-data）')
        train_ds = None
    else:
        print('[Data] 加载训练集（人工数据）...')
        train_ds = MathWritingDataset(
            split='train', vocab=vocab,
            max_src=args.max_src, max_tgt=args.max_tgt,
            data_dir=data_dir, max_samples=args.max_samples,
            augment=True, oversample_hard=1,
        )

    if args.use_synthetic:
        print('[Data] 加载合成数据...')
        synth_ds = MathWritingDataset(
            split='synthetic', vocab=vocab,
            max_src=args.max_src, max_tgt=args.max_tgt,
            data_dir=data_dir, max_samples=args.max_samples,
            augment=True, oversample_hard=1,
        )
        train_ds = ConcatDataset([train_ds, synth_ds]) if train_ds else synth_ds
        print(f'[Data] 合并后总样本: {len(train_ds):,}')

    # 靶向增量数据（每轮针对错误类型生成的 inkml）
    for extra_split in (args.extra_data or []):
        parquet_path = data_dir / f'{extra_split}.parquet'
        extra_dir = data_dir / extra_split
        if not parquet_path.exists() and not extra_dir.exists():
            print(f'[Data] 警告：{extra_split} 不存在，跳过')
            continue
        print(f'[Data] 加载额外数据: {extra_split} ...')
        extra_ds = MathWritingDataset(
            split=extra_split, vocab=vocab,
            max_src=args.max_src, max_tgt=args.max_tgt,
            data_dir=data_dir, max_samples=None,
            augment=True, oversample_hard=1,
        )
        train_ds = ConcatDataset([train_ds, extra_ds]) if train_ds else extra_ds
        print(f'[Data] 加入 {len(extra_ds):,} 个样本，总计: {len(train_ds):,}')

    val_ds = MathWritingDataset(
        split='valid', vocab=vocab,
        max_src=args.max_src, max_tgt=args.max_tgt,
        data_dir=data_dir,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(DEVICE == 'cuda'),
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(2, args.num_workers // 4), pin_memory=(DEVICE == 'cuda'),
        collate_fn=collate_fn,
    )

    # ── 优化器 ───────────────────────────────────────────────────── #
    all_params = list(prefix_enc.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    total_steps = max(len(train_loader) * args.epochs // args.grad_accum, 1)
    warmup_steps = min(total_steps // 10, 1000)

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=vocab.PAD,
        label_smoothing=args.label_smoothing,
    )

    # ── Resume ───────────────────────────────────────────────────── #
    start_epoch = 0
    resume_step = 0
    best_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        prefix_enc.load_state_dict(ckpt['prefix_enc'])
        decoder.load_state_dict(ckpt['decoder'])
        if not args.reset_optimizer:
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = 0 if args.reset_optimizer else ckpt.get('epoch', 0)
        resume_step = 0 if args.reset_optimizer else ckpt.get('global_step', 0)
        best_acc = ckpt.get('val_acc', ckpt.get('best_acc', 0.0))
        print(f'[Resume] epoch={start_epoch}  step={resume_step}  best_acc={best_acc:.4f}'
              + ('  [optimizer reset]' if args.reset_optimizer else ''))

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    # ── 训练循环 ─────────────────────────────────────────────────── #
    global_step = resume_step
    print(f'\n[Train] epochs={args.epochs}  batch={args.batch}×{args.grad_accum}'
          f'  steps/epoch={len(train_loader)}  n_prefix={n_prefix}')

    for epoch in range(1, args.epochs + 1):
        if epoch <= start_epoch:
            continue

        prefix_enc.train()
        decoder.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (stroke, stroke_mask, imgs, ids_list) in enumerate(train_loader, 1):
            stroke = stroke.to(DEVICE)
            stroke_mask = stroke_mask.to(DEVICE)
            imgs = imgs.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda'), dtype=torch.bfloat16):
                # 1. 视觉前缀
                prefix = prefix_enc(imgs, stroke, stroke_mask)  # (B, n_prefix, d_model)
                B = prefix.size(0)

                # 2. 构建 input 和 target
                #    input:  [BOS, tok1, ..., tokN]
                #    target: [tok1, ..., tokN, EOS]
                max_tgt_len = max(len(ids) for ids in ids_list)
                input_ids = torch.full((B, max_tgt_len + 1), vocab.PAD,
                                       dtype=torch.long, device=DEVICE)
                target_ids = torch.full((B, max_tgt_len + 1), vocab.PAD,
                                        dtype=torch.long, device=DEVICE)

                for i in range(B):
                    ids = ids_list[i].tolist() if hasattr(ids_list[i], 'tolist') else list(ids_list[i])
                    L = len(ids)
                    input_ids[i, 0] = vocab.BOS
                    input_ids[i, 1:L+1] = torch.tensor(ids, dtype=torch.long)
                    target_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
                    target_ids[i, L] = vocab.EOS

                # 3. Forward
                logits = decoder(prefix, input_ids)  # (B, T_text, vocab_size)

                # 4. Loss
                loss = loss_fn(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_latest(out_dir, epoch, step, global_step, prefix_enc,
                                 decoder, optimizer, scheduler, scaler,
                                 epoch_loss / step, best_acc)
                    print(f'  [Ckpt] step={global_step} saved', flush=True)

            if step % 50 == 0:
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]['lr']
                print(f'  Epoch {epoch} [{step}/{len(train_loader)}]'
                      f'  loss={epoch_loss/step:.4f}'
                      f'  lr={lr_now:.2e}'
                      f'  {elapsed:.0f}s', flush=True)

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch}  avg_loss={avg_loss:.4f}  {time.time()-t0:.0f}s')

        # ── 保存 latest ──────────────────────────────────────────── #
        _save_latest(out_dir, epoch, step, global_step, prefix_enc,
                     decoder, optimizer, scheduler, scaler,
                     avg_loss, best_acc)

        # ── 评估 ─────────────────────────────────────────────────── #
        # 二档评估：eval_every_large（默认 10）次粗 eval（eval_max_samples 500） + 1 次精 eval（eval_large_samples 2000）
        do_small = args.eval_every > 0 and (epoch % args.eval_every == 0)
        do_large = args.eval_every_large > 0 and (epoch % args.eval_every_large == 0)
        if do_large:
            n_eval = args.eval_large_samples
            tag = 'large'
        elif do_small:
            n_eval = args.eval_max_samples
            tag = 'small'
        else:
            n_eval = 0
        if n_eval > 0:
            print(f'[Eval-{tag}] Epoch {epoch} ({n_eval} samples, greedy)...')
            val_acc, val_cer = evaluate(prefix_enc, decoder, vocab, val_loader,
                                        DEVICE, max_new_tokens=args.max_tgt,
                                        max_samples=n_eval, beam=args.eval_beam)
            print(f'[Eval-{tag}] Epoch {epoch}  EM={val_acc:.4f}  CER={val_cer:.4f}  (n={n_eval})')

            # best ckpt 仅在 large eval 上更新（避免被 500 子集噪声误选）
            if tag == 'large' and val_acc > best_acc:
                best_acc = val_acc
                # 保存最佳模型
                best_dir = out_dir / f'best_epoch{epoch:03d}_acc{val_acc:.4f}'
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'prefix_enc': prefix_enc.state_dict(),
                    'decoder': decoder.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'args': vars(args),
                }, best_dir / 'model.pt')
                print(f'[Save] 新最佳: {best_dir}')

                # 更新 latest 中的 best_acc
                _save_latest(out_dir, epoch, step, global_step, prefix_enc,
                             decoder, optimizer, scheduler, scaler,
                             avg_loss, best_acc)

    print(f'\n训练完成  best_val_acc={best_acc:.4f}')


def _save_latest(out_dir, epoch, step, global_step, prefix_enc,
                 decoder, optimizer, scheduler, scaler, loss, best_acc):
    # 原子写：先写临时文件再 rename，防止 scp/rsync 读到写了一半的文件
    tmp = out_dir / 'latest.pt.tmp'
    torch.save({
        'prefix_enc': prefix_enc.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'step': step,
        'global_step': global_step,
        'loss': loss,
        'best_acc': best_acc,
    }, tmp)
    tmp.replace(out_dir / 'latest.pt')


if __name__ == '__main__':
    main()
