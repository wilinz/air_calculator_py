import sys, torch
from pathlib import Path
from executorch.runtime import Runtime

PTE='/Users/wilinz/Documents/大四上/毕设/project/experiments/2026-09-01_stroke_only_full_pipeline/export_ios_merged/decoder.pte'
rt=Runtime.get(); prog=rt.load_program(Path(PTE))
print('方法列表:', list(prog.method_names))

pre=prog.load_method('prefill')
kv=pre.execute([torch.zeros(1,32,512)])[0]
print('prefill 输出 kv shape:', tuple(kv.shape))

st=prog.load_method('step')
out=st.execute([torch.zeros(1,1,dtype=torch.int32),
                torch.tensor([32],dtype=torch.int32), kv])
print('step 输出:', [tuple(o.shape) for o in out])
