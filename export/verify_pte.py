#!/usr/bin/env python3
"""对比 PyTorch 原模型与导出 PTE 的 prefix_enc 输出（纯轨迹）。"""
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path('/Users/wilinz/Documents/大四上/毕设/project/air_calculator_py/export')))
from models import VisualPrefixEncoder
from dataset import Vocabulary

CKPT='/Users/wilinz/Documents/大四上/毕设/project/experiments/2026-09-01_stroke_only_full_pipeline/stroke_v4_best_ep040_acc0.7650.pt'
PTE='/Users/wilinz/Documents/大四上/毕设/project/experiments/2026-09-01_stroke_only_full_pipeline/export_ios/prefix_enc.pte'

ckpt=torch.load(CKPT,map_location='cpu',weights_only=False)
ca=ckpt.get('args',{})
print('modality =',ca.get('modality'),' d_model =',ca.get('d_model'),' n_stroke_tok =',ca.get('n_stroke_tok'))

pe=VisualPrefixEncoder(d_model=ca.get('d_model',512),n_stroke=ca.get('n_stroke_tok',32),
                       modality=ca.get('modality','both')).eval()
pe.load_state_dict(ckpt['prefix_enc'])

sys.path.insert(0,str(Path(__file__).parent))
from export_et_hybrid_fp16 import PrefixEncUnrolledModule
mod=PrefixEncUnrolledModule(pe,stroke_len=ca.get('max_src',512)).eval()

torch.manual_seed(0)
img=torch.zeros(1,1,64,256)
stroke=torch.randn(1,ca.get('max_src',512),13)
rlen=torch.tensor([128],dtype=torch.int32)
with torch.no_grad():
    ref=mod(img,stroke,rlen)
print('PyTorch 输出 shape:',tuple(ref.shape))

from executorch.runtime import Runtime
rt=Runtime.get()
prog=rt.load_program(Path(PTE))
m=prog.load_method('forward')
out=m.execute([img,stroke,rlen])[0]
print('PTE     输出 shape:',tuple(out.shape))
d=(out-ref).abs()
print(f'最大绝对误差 = {d.max().item():.6f}   平均绝对误差 = {d.mean().item():.6f}')
cos=torch.nn.functional.cosine_similarity(out.flatten(),ref.flatten(),dim=0)
print(f'余弦相似度 = {cos.item():.6f}')
