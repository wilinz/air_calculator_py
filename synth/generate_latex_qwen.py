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
用本地 Qwen2.5-Math-Instruct 针对错误类型批量生成 LaTeX 公式。
与 generate_latex.py 完全兼容（相同输出格式、相同 ERROR_PROMPTS）。

用法:
  python3 generate_latex_qwen.py \
    --model /path/to/Qwen2.5-Math-1.5B-Instruct \
    --error-types upper_lower_mix greek_lookalike symbol_confusion \
    --n 5000 \
    --out ../../data/generated_latex/qwen_round1.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── ERROR_PROMPTS（与 generate_latex.py 完全一致，直接内联避免 anthropic 依赖）── #

_SYSTEM = (
    "You are generating LaTeX math expressions for ML training data. "
    "Output only LaTeX, one per line, no numbering or explanation. "
    "Generate expressions (sub-expressions, terms, fragments) — NOT only complete equations. "
    "Mathematically valid is enough; no need to be a full formula or have an = sign. "
    "IMPORTANT: include concrete numbers (2, 3, 5, \\frac{1}{2}, \\frac{3}{4}, \\pi, e) as actual terms, "
    "coefficients, or standalone numeric fractions in +/-/×/÷ operations — NOT only as exponents or subscripts. "
    "Examples: 3x^2 + 2x - 5, \\frac{2}{x+1}, 5\\sin(x) - 3, \\frac{1}{2} + \\frac{1}{3}, 2\\pi r + 7. "
    "Length: 8-25 tokens. "
    "FORBIDDEN commands (do NOT use): \\text{}, \\mathbf{}, \\mathbb{}, \\vec{}, \\boldsymbol{}, "
    "\\mathrm{}, \\mathcal{}, \\mathit{}, \\operatorname{}, \\,, \\;, \\!, \\quad, \\qquad — these are not supported. "
    "Use only standard math symbols and greek letters."
)

ERROR_PROMPTS = {
    "upper_lower_mix": {
        "desc": "same handwriting-similar letter in both upper and lowercase within one expression",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions where EACH line contains the SAME letter in BOTH "
            "uppercase AND lowercase form. "
            "Pick from: C/c, K/k, M/m, N/n, O/o, P/p, S/s, U/u, V/v, W/w, X/x, Y/y, Z/z. "
            "Use a DIFFERENT pair each line. "
            "Patterns (adapt freely, do NOT copy verbatim): "
            "K_n = \\sum_{{i=1}}^n k_i, "
            "N(0,1) \\text{{ with }} n \\text{{ samples}}, "
            "Y = y_0 e^{{rt}}, "
            "C_k = \\binom{{n}}{{k}} c^k, "
            "Z = z_1 + z_2, "
            "U = \\int_0^1 u(x)\\,dx, "
            "W(w) = w^2 + 1. "
            "Mix domains: probability, calculus, combinatorics, linear algebra.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "bracket_confusion": {
        "desc": "deeply nested brackets and subscripts/superscripts with balanced symbol coverage",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions with DEEP nesting (2-3 levels). "
            "Distribute evenly across ALL these nesting patterns — roughly equal count for each: "
            "\\frac inside \\frac, "
            "\\frac inside \\sqrt, "
            "\\sqrt inside \\frac, "
            "\\sum inside \\frac, "
            "\\frac inside \\int body, "
            "deeply nested parens \\left(\\left(expr\\right)^2 + c\\right), "
            "\\left[\\right] wrapping sum or fraction, "
            "\\left\\{{ ... \\right\\}} curly braces wrapping a condition or fraction, "
            "subscript/superscript chains x_{{i_j}}^{{k^2}}, "
            "all three bracket types in one meaningful expression, e.g. "
            "\\left\\{{x \\in \\left[0,1\\right] : \\left(x - \\frac{{1}}{{2}}\\right)^2 < \\frac{{1}}{{4}}\\right\\}}, "
            "\\left[\\sum_{{k=1}}^n \\left(a_k + \\frac{{1}}{{k}}\\right)\\right] \\cap \\left\\{{x > 0\\right\\}}. "
            "About half should have concrete numeric bounds or coefficients. "
            "Each line must use a DIFFERENT nesting pattern.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "mixed_brackets": {
        "desc": "single deep nesting and pairwise/triple combinations of (), [], {} brackets",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions involving bracket nesting. "
            "Distribute evenly across these patterns: "
            "deep single () nesting like \\left(\\left(\\left(expr\\right)+1\\right)^2\\right), "
            "deep single [] nesting like \\left[\\left[expr\\right]^2 + c\\right], "
            "deep single \\{{\\}} nesting like \\left\\{{\\left\\{{expr\\right\\}} + 1\\right\\}}, "
            "() inside [], [] inside (), "
            "() inside \\{{\\}}, \\{{\\}} wrapping (), "
            "[] inside \\{{\\}}, \\{{\\}} wrapping [], "
            "all three together. "
            "Include concrete numbers. Each line structurally different.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "insertion": {
        "desc": "expressions with brackets and nested structure",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions from different areas (algebra, calculus, geometry, probability). "
            "Mix equations (with =) and standalone expressions freely — both are fine. "
            "Include \\left(\\right), fractions, integrals, sums, and nested structures. "
            "About half should contain concrete numbers as coefficients or bounds. "
            "Each line from a different domain with a different structure.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "deletion": {
        "desc": "nested structure expressions",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions with natural nesting (sqrt inside fraction, fraction inside integral, etc.). "
            "Use concrete numbers for coefficients and bounds where natural. "
            "Cover algebra, analysis, combinatorics, and applied math. Vary the nesting style each line.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "greek_lookalike": {
        "desc": "Greek letters that look like plain letters: nu/v, mu/u, omega/w, partial/d, rho/p, tau/t",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions that each contain at least one of: "
            "\\nu, \\mu, \\omega, \\partial, \\rho, \\tau. "
            "Each expression should have at least one of: \\frac, \\int, \\sum, \\sqrt, \\lim, or nested brackets. "
            "About half should include concrete numbers. Cover calculus, PDE, probability — each line structurally different.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
        "user_control": (
            "Generate {n} math expressions using plain letters v, u, w, d, p, t as variables — NOT \\nu \\mu \\omega \\partial \\rho \\tau. "
            "Each expression should have at least one of: \\frac, \\int, \\sum, \\sqrt, \\lim, or nested brackets. "
            "About half should include concrete numbers. Cover algebra, calculus, probability — each line structurally different.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "symbol_confusion": {
        "desc": "visually similar operator/decorator symbols: tilde, cdot, times, approx, sim, gg, ll, overline",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions that each contain at least one of: "
            "\\tilde, \\overline, \\cdot, \\times, \\gg, \\ll, \\sim, \\approx. "
            "Each expression should have at least one of: \\frac, \\int, \\sum, \\sqrt, \\lim, or nested brackets. "
            "About half should include concrete numbers. Cover calculus, algebra, probability — each line structurally different.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "digit_letter": {
        "desc": "digits mixed with letter variables",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions where digits and letter variables coexist. "
            "Every expression must contain at least one concrete number (2, 3, \\frac{{1}}{{2}}, \\pi, e) "
            "as a coefficient, exponent, or bound — not just in a subscript. "
            "Cover polynomials, sequences, combinatorics, probability. Each line different.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "mixed_hard": {
        "desc": "mixed multi-domain expressions",
        "system": _SYSTEM,
        "user": (
            "Generate {n} math expressions. "
            "Cover algebra, calculus, probability, geometry, number theory — each line from a different domain. "
            "Mix standalone expressions and equations freely. About half should contain concrete numbers.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },
}

# ── 错误类型验证规则（后验过滤，确保公式真的含目标符号）─────────────── #

_REQUIRED_ANY: dict[str, list[str]] = {
    "upper_lower_mix": [],   # 结构约束，不做 token 过滤
    "greek_lookalike": [r'\nu', r'\mu', r'\omega', r'\partial', r'\rho', r'\tau'],
    "symbol_confusion": [r'\tilde', r'\overline', r'\cdot', r'\times',
                         r'\gg', r'\ll', r'\sim', r'\approx'],
    "bracket_confusion": [],
    "mixed_brackets":   [],
    "insertion":        [],
    "deletion":         [],
    "digit_letter":     [],
    "mixed_hard":       [],
}

def _passes_required(line: str, error_type: str) -> bool:
    """检查 line 是否包含该错误类型要求的目标符号（至少一个）。"""
    required = _REQUIRED_ANY.get(error_type, [])
    if not required:
        return True
    return any(sym in line for sym in required)


# ── 模型加载 ─────────────────────────────────────────────────────────── #

_model = None
_tokenizer = None


def load_model(model_path: str):
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"[Qwen] 加载模型: {model_path}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    dtype = torch.float16 if device != 'cpu' else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    _model.eval()
    print(f"[Qwen] 加载完成，device={device}", flush=True)


# ── 生成单批 ─────────────────────────────────────────────────────────── #

def generate_batch(
    error_type: str,
    n_per_call: int,
    use_control: bool = False,
    temperature: float = 0.9,
    max_new_tokens: int = 1024,
    recent: list[str] | None = None,   # 最近生成的公式（滑动窗口，≤15条）
) -> list[str]:
    prompt_cfg = ERROR_PROMPTS[error_type]
    user_key = "user_control" if (use_control and "user_control" in prompt_cfg) else "user"
    user_msg = prompt_cfg[user_key].format(n=n_per_call)

    # 把最近生成的样本附加到 prompt，让模型主动避开
    if recent:
        avoid_str = '\n'.join(recent[-15:])
        user_msg += f"\n\nDo NOT repeat or closely imitate these already-generated expressions:\n{avoid_str}"

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer([text], return_tensors='pt').to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=_tokenizer.eos_token_id,
        )
    response = _tokenizer.decode(
        out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    )

    def _clean(line: str) -> str:
        """清理单行：去掉包装符和前缀，返回纯 LaTeX 或空串。"""
        line = line.strip()
        if not line:
            return ''
        # 去掉编号前缀、markdown 列表前缀、"Expression:" 等标签
        line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
        line = re.sub(r'^[-\*]\s+', '', line).strip()
        line = re.sub(r'^(Expression|Formula|Result|Answer)\s*\d*\s*[:：]\s*', '', line, flags=re.IGNORECASE).strip()
        # 去掉行尾 LaTeX 换行符 \\ 和多余标点
        line = re.sub(r'\s*\\\\$', '', line).strip()
        line = line.rstrip('.,;').strip()
        # 去掉行尾的领域注释，如 "(calculus)" "(combinatorics)"
        line = re.sub(r'\s*\([a-zA-Z\s]+\)\s*$', '', line).strip()
        # 去掉 \boxed{...} 包装
        line = re.sub(r'^\\boxed\{(.*)\}$', r'\1', line).strip()
        # 去掉各种数学模式包装符（可能在去掉前缀后才暴露）
        for start, end in [('$$', '$$'), ('$', '$'), (r'\[', r'\]'), (r'\(', r'\)')]:
            if line.startswith(start) and line.endswith(end) and len(line) > len(start) + len(end):
                line = line[len(start):-len(end)].strip()
                break
        return line

    lines = []
    for raw in response.splitlines():
        line = _clean(raw)
        if not line:
            continue
        # 过滤对齐环境行（& 开头）和 \begin/\end 行
        if line.startswith('&') or re.match(r'^\\(begin|end)\{', line):
            continue
        # 过滤孤立/不完整的结构
        if line in (r'\[', r'\]', r'\(', r'\)', r'\boxed{'):
            continue
        # 过滤 markdown 粗体行 **...**
        if re.match(r'^\*\*.*\*\*[:\.]?\s*$', line):
            continue
        # 过滤含太多英文单词的说明行（≥5 个长单词 + 含冒号/逗号）
        long_words = re.findall(r'[a-zA-Z]{5,}', line)
        if len(long_words) >= 5 and (':' in line or ',' in line):
            continue
        # 过滤纯英文（无任何 LaTeX/数学字符）
        has_math = bool(re.search(r'\\[a-zA-Z]|[\^_]|\{|\}|\d+[a-zA-Z]|[a-zA-Z]\d+', line))
        if not has_math:
            continue
        # 过滤含禁用命令的行
        _FORBIDDEN = (r'\text{', r'\mathbf{', r'\mathbb{', r'\vec{', r'\boldsymbol{',
                      r'\mathrm{', r'\mathcal{', r'\mathit{', r'\operatorname{')
        if any(cmd in line for cmd in _FORBIDDEN):
            continue
        # 过滤含英文句子的行（含 "if", "where", "and", "is", "are" 等连接词）
        if re.search(r'\b(if|where|and|is|are|not|the|an|a)\b', line, re.IGNORECASE):
            english_words = re.findall(r'[a-zA-Z]{3,}', line)
            if len(english_words) >= 3:
                continue
        # 过滤太短或太长
        if len(line) < 3 or len(line) > 300:
            continue
        lines.append(line)
    return lines


# ── 主流程 ───────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None,
                    help='Qwen 模型路径或 HuggingFace model ID')
    ap.add_argument('--error-types', nargs='+',
                    default=list(ERROR_PROMPTS.keys()),
                    choices=list(ERROR_PROMPTS.keys()))
    ap.add_argument('--n', type=int, default=5000, help='每种错误类型生成总数')
    ap.add_argument('--n-per-call', type=int, default=30, help='每次推理生成条数')
    ap.add_argument('--out', required=True)
    ap.add_argument('--temperature', type=float, default=0.9)
    args = ap.parse_args()

    # 自动查找本地 HuggingFace 缓存
    model_path = args.model
    if model_path is None:
        candidates = [
            Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct',
            Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct',
        ]
        for c in candidates:
            snaps = list((c / 'snapshots').glob('*')) if (c / 'snapshots').exists() else []
            if snaps:
                model_path = str(snaps[0])
                break
        if model_path is None:
            ap.error('找不到本地 Qwen 模型，请用 --model 指定路径')

    load_model(model_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(out_path, 'a') as fout:
        for et in args.error_types:
            need = args.n
            written = 0
            seen = set()
            print(f'\n[{et}] 目标 {need} 条', flush=True)

            use_control = False
            recent_window: list[str] = []   # 滑动窗口，最近生成的公式
            while written < need:
                batch_n = min(args.n_per_call, need - written)
                lines = generate_batch(et, batch_n, use_control=use_control,
                                       temperature=args.temperature,
                                       recent=recent_window)

                for line in lines:
                    if not _passes_required(line, et):
                        continue
                    uid = hashlib.md5(line.encode()).hexdigest()[:8]
                    if uid in seen:
                        continue
                    seen.add(uid)
                    rec = {'formula': line, 'error_type': et, 'source': 'qwen'}
                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    written += 1
                    total_written += 1
                    recent_window.append(line)
                    if len(recent_window) > 15:
                        recent_window.pop(0)
                    if written >= need:
                        break

                # greek_lookalike 交替生成正反两面
                if et == 'greek_lookalike':
                    use_control = not use_control

                print(f'  {written}/{need}', end='\r', flush=True)

            print(f'  {written}/{need} ✓')

    print(f'\n[完成] 共写入 {total_written} 条 → {out_path}')


if __name__ == '__main__':
    main()
