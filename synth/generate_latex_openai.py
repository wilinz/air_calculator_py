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
用 OpenAI 兼容 API（DeepSeek V3 / 中转站等）批量生成 LaTeX 公式。
与 generate_latex.py 完全兼容（相同输出格式、相同 ERROR_PROMPTS）。

用法:
  python3 generate_latex_openai.py \
    --error-types upper_lower_mix bracket_confusion \
    --n 4000 \
    --out ../data/generated_latex/deepseek_round1.jsonl

配置文件（openai_api.ini，与脚本同目录）:
  [openai]
  api_key  = YOUR_KEY
  base_url = https://your-relay.com/v1
  model    = deepseek-chat
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import random
import configparser
from pathlib import Path

from openai import AsyncOpenAI

# ── ERROR_PROMPTS & _SYSTEM（与 generate_latex.py 完全一致）─────── #

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
        "user": (
            "Generate {n} math expressions. "
            "EACH line MUST contain the SAME letter in both uppercase and lowercase, "
            "chosen from: C/c, K/k, M/m, N/n, O/o, P/p, S/s, U/u, V/v, W/w, X/x, Y/y, Z/z. "
            "Distribute evenly across these pairs — do NOT always use X/x or S/s. "
            "Good examples: P(X = x), S_n = \\sum s_i, V v = \\lambda v, W(w) = w^2, N(n) = n!, "
            "M = \\sum m_i, Y = y_1 + y_2, C(n,k) \\cdot c, O(n) vs o(n), U u^T = I, Z(z) = z^2 + c. "
            "Cover probability, linear algebra, analysis, combinatorics. Each line structurally different.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "bracket_confusion": {
        "desc": "deeply nested brackets and subscripts/superscripts with balanced symbol coverage",
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
            "all three bracket types in one meaningful expression. "
            "About half should have concrete numeric bounds or coefficients. "
            "Each line must use a DIFFERENT nesting pattern.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "mixed_brackets": {
        "desc": "single deep nesting and pairwise/triple combinations of (), [], {} brackets",
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
        "user": (
            "Generate {n} math expressions with natural nesting (sqrt inside fraction, fraction inside integral, etc.). "
            "Use concrete numbers for coefficients and bounds where natural. "
            "Cover algebra, analysis, combinatorics, and applied math. Vary the nesting style each line.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "greek_lookalike": {
        "desc": "Greek letters that look like plain letters: nu/v, mu/u, omega/w, partial/d, rho/p, tau/t",
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
        "user": (
            "Generate {n} math expressions. "
            "Cover algebra, calculus, probability, geometry, number theory — each line from a different domain. "
            "Mix standalone expressions and equations freely. About half should contain concrete numbers.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },

    "subscript_brace": {
        "desc": "heavy use of { } _ ^ with nested subscripts/superscripts and brace grouping",
        "user": (
            "Generate {n} math expressions with HEAVY use of subscripts (_), superscripts (^), and brace grouping ({{}}). "
            "Distribute evenly across these patterns — roughly equal count for each: "
            "nested subscripts like x_{{i_j}} or a_{{n_k}}, "
            "nested superscripts like e^{{x^2}} or f^{{(n)}}, "
            "mixed sub+super on same symbol like x_{{i}}^{{(k)}}, f_{{n}}^{{2}}, \\sigma_{{ij}}^{{2}}, "
            "chains like x_{{i_{{j+1}}}}^{{k^{{2}}+1}}, "
            "sums/products with multi-char bounds \\sum_{{i=1}}^{{n}} and \\prod_{{k=0}}^{{N-1}}, "
            "integrals with limits \\int_{{a}}^{{b}} and \\int_{{-\\infty}}^{{+\\infty}}, "
            "fractions with brace-grouped numerator/denominator \\frac{{a^2+b^2}}{{c_{{ij}}+1}}, "
            "matrix-style subscripts A_{{mn}}, T_{{ij}}^{{k}}, \\lambda_{{\\max}}, "
            "multi-level braces like (x^{{a_{{bc}}}})^{{2}} or \\left(x_{{i}}^{{j}}\\right)^{{n_k}}. "
            "Every expression MUST contain at least 3 uses of _ or ^ combined, and at least 2 brace groups. "
            "Include concrete numbers in about half. Each line structurally different.\n"
            "Output LaTeX only, one per line, {n} lines total."
        ),
    },
}

# ── 后验验证：确保公式含目标符号 ─────────────────────────────────── #

_REQUIRED_ANY: dict[str, list[str]] = {
    "greek_lookalike": [r'\nu', r'\mu', r'\omega', r'\partial', r'\rho', r'\tau'],
    "symbol_confusion": [r'\tilde', r'\overline', r'\cdot', r'\times',
                         r'\gg', r'\ll', r'\sim', r'\approx'],
}

def _passes_required(line: str, error_type: str) -> bool:
    required = _REQUIRED_ANY.get(error_type, [])
    if not required:
        return True
    return any(sym in line for sym in required)


# ── 清洗 / 过滤 ──────────────────────────────────────────────────── #

_BANNED_CMDS = re.compile(
    r'\\(?:text|mathbf|mathbb|vec|boldsymbol|mathrm|mathcal|mathit|operatorname)\{'
    r'|\\[,;!]'
    r'|\\(?:quad|qquad)\b'
)

def clean_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^```[a-z]*\n?', '', s)
    s = re.sub(r'\n?```$', '', s)
    s = re.sub(r'^\d+[\.\)]\s*', '', s)
    s = re.sub(r'^[-\*]\s+', '', s)
    for start, end in [('$$', '$$'), ('$', '$'), (r'\[', r'\]'), (r'\(', r'\)')]:
        if s.startswith(start) and s.endswith(end) and len(s) > len(start) + len(end):
            s = s[len(start):-len(end)].strip()
            break
    s = s.rstrip('.,;').strip()
    return s

def filter_formula(s: str, vocab_tokens: set,
                   min_tok: int = 4, max_tok: int = 35) -> bool:
    if not s:
        return False
    if _BANNED_CMDS.search(s):
        return False
    toks = re.findall(r'\\[a-zA-Z]+|\{|\}|\^|_|\d+|[a-zA-Z]|[^\s]', s)
    if not (min_tok <= len(toks) <= max_tok):
        return False
    if vocab_tokens:
        known = sum(1 for t in toks if t in vocab_tokens)
        if known < len(toks) * 0.3:
            return False
    depth = 0
    for c in s:
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        if depth < 0:
            return False
    return depth == 0

def dedup(formulas: list[str]) -> list[str]:
    seen = set()
    result = []
    for f in formulas:
        h = hashlib.md5(f.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(f)
    return result


# ── API 调用 ─────────────────────────────────────────────────────── #

# 每次随机注入的领域/约束片段，增加生成多样性
_DOMAIN_HINTS = [
    "Focus on calculus and real analysis.",
    "Focus on linear algebra and matrices.",
    "Focus on probability and statistics.",
    "Focus on number theory and combinatorics.",
    "Focus on differential equations and physics.",
    "Focus on complex analysis.",
    "Focus on discrete mathematics and sequences.",
    "Focus on geometry and trigonometry.",
    "Mix all mathematical domains freely.",
    "Emphasize expressions with fractions and radicals.",
    "Emphasize expressions with sums and products.",
    "Emphasize expressions with integrals and limits.",
]

_LENGTH_HINTS = [
    "Keep expressions SHORT: 5-10 tokens each.",
    "Keep expressions MEDIUM length: 10-18 tokens each.",
    "Keep expressions LONG and complex: 18-30 tokens each.",
    "Mix short and long expressions freely.",
]


async def generate_batch(
    client: AsyncOpenAI,
    error_type: str,
    n_per_call: int,
    semaphore: asyncio.Semaphore,
    model: str = 'deepseek-chat',
    use_control: bool = False,
    temperature: float = 1.2,
) -> list[str]:
    prompt_cfg = ERROR_PROMPTS[error_type]
    user_key = "user_control" if (use_control and "user_control" in prompt_cfg) else "user"
    # 每次随机注入领域 + 长度约束，打破重复
    domain_hint = random.choice(_DOMAIN_HINTS)
    length_hint = random.choice(_LENGTH_HINTS)
    user_msg = prompt_cfg[user_key].format(n=n_per_call) + f"\n{domain_hint} {length_hint}"

    for attempt in range(4):
        try:
            async with semaphore:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        max_tokens=4096,
                        temperature=temperature,
                        messages=[
                            {"role": "system", "content": _SYSTEM},
                            {"role": "user",   "content": user_msg},
                        ],
                    ),
                    timeout=120.0,
                )
            text = resp.choices[0].message.content.strip()
            return [l.strip() for l in text.splitlines() if l.strip()]
        except asyncio.TimeoutError:
            print(f"  [Timeout] 超过 60s，重试 {attempt+1}/4 ...", flush=True)
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  [Error] {e}, 等待 {wait:.1f}s ...", flush=True)
            await asyncio.sleep(wait)

    return []


# ── 配置加载 ─────────────────────────────────────────────────────── #

def load_config(config_path: str | None) -> dict:
    cfg = {}
    candidates = [
        config_path,
        str(Path(__file__).parent.parent / 'openai_api.ini'),
        str(Path(__file__).parent / 'openai_api.ini'),
        str(Path.home() / '.openai_api.ini'),
    ]
    for path in candidates:
        if path and Path(path).exists():
            parser = configparser.ConfigParser()
            parser.read(path)
            sec = parser['openai'] if 'openai' in parser else parser.defaults()
            cfg['api_key']  = sec.get('api_key',  '').strip() or None
            cfg['base_url'] = sec.get('base_url', '').strip() or None
            cfg['model']    = sec.get('model',    '').strip() or None
            print(f"[Config] 从 {path} 加载配置")
            break
    return cfg


# ── 主流程 ───────────────────────────────────────────────────────── #

async def main_async(args):
    file_cfg = load_config(args.config)
    api_key  = args.api_key  or file_cfg.get('api_key')  or os.environ.get('OPENAI_API_KEY')
    base_url = args.base_url or file_cfg.get('base_url')
    model    = args.model    or file_cfg.get('model') or 'deepseek-chat'

    client = AsyncOpenAI(
        api_key=api_key or 'sk-placeholder',
        base_url=base_url,
    )

    # 加载 vocab
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'train'))
    try:
        from mathwriting_dataset import get_vocabulary
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'mathwriting-2024'
        vocab = get_vocabulary(data_dir)
        vocab_tokens = set(vocab.idx2token)
        if len(vocab_tokens) <= 4:
            raise ValueError("vocab 为空")
        print(f"[Config] vocab size={len(vocab_tokens)}")
    except Exception as e:
        vocab_tokens = set()
        print(f"[Warning] 无法加载 vocab（{e}），跳过 vocab 过滤")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    existing_per_type: dict[str, int] = {}
    if out_path.exists() and not args.overwrite:
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                existing.add(d['formula'])
                et = d.get('error_type', '')
                existing_per_type[et] = existing_per_type.get(et, 0) + 1
        print(f"[已有] {len(existing)} 条，增量追加模式")
        for et, cnt in sorted(existing_per_type.items()):
            print(f"  {et:<22} {cnt} 条")

    error_types = args.error_types
    n_per_type  = args.n // len(error_types)
    n_per_call  = args.batch
    semaphore   = asyncio.Semaphore(args.concurrency)
    # 用已有数量初始化，避免重复生成
    all_results: dict[str, list[str]] = {
        et: [''] * existing_per_type.get(et, 0) for et in error_types
    }

    out_mode = 'w' if args.overwrite or not out_path.exists() else 'a'
    out_file = open(out_path, out_mode)
    total_written = 0

    try:
        for error_type in error_types:
            if error_type not in ERROR_PROMPTS:
                print(f"[跳过] 未知: {error_type}")
                continue

            print(f"\n[{error_type}] 目标 {n_per_type} 条 ...")
            has_control = "user_control" in ERROR_PROMPTS[error_type]
            call_counter = 0

            while len(all_results[error_type]) < n_per_type:
                need   = n_per_type - len(all_results[error_type])
                n_calls = min(max(1, (need + n_per_call - 1) // n_per_call),
                              args.concurrency * 2)

                tasks = [
                    generate_batch(
                        client, error_type, n_per_call, semaphore, model,
                        use_control=(has_control and (call_counter + i) % 2 == 1),
                        temperature=args.temperature,
                    )
                    for i in range(n_calls)
                ]
                call_counter += n_calls
                batches = await asyncio.gather(*tasks)

                new_batch = []
                for batch in batches:
                    for raw in batch:
                        formula = clean_latex(raw)
                        if (formula
                                and formula not in existing
                                and _passes_required(formula, error_type)
                                and filter_formula(formula, vocab_tokens,
                                                   args.min_tokens, args.max_tokens)):
                            new_batch.append(formula)
                            existing.add(formula)

                new_batch = dedup(new_batch)
                all_results[error_type].extend(new_batch)

                if new_batch:
                    for formula in new_batch:
                        out_file.write(json.dumps({
                            "formula":    formula,
                            "error_type": error_type,
                            "source":     "openai",
                        }, ensure_ascii=False) + '\n')
                    out_file.flush()
                    total_written += len(new_batch)

                print(f"  → 累计 {len(all_results[error_type])}/{n_per_type}  "
                      f"(本轮 +{len(new_batch)}, 已保存 {total_written})", flush=True)

                if not new_batch:
                    print("  [Warning] 本轮无新增，稍等后重试...")
                    await asyncio.sleep(3)
    finally:
        out_file.close()

    print(f"\n[完成] 写出 {total_written} 条 → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="用 OpenAI 兼容 API（DeepSeek V3 等）批量生成 LaTeX 公式"
    )
    parser.add_argument('--error-types', nargs='+',
                        default=list(ERROR_PROMPTS.keys()))
    parser.add_argument('--n', type=int, default=3000,
                        help='总目标数量（按类型平均分配）')
    parser.add_argument('--out', default='../data/generated_latex/deepseek_round1.jsonl')
    parser.add_argument('--batch', type=int, default=30,
                        help='每次 API 调用请求的公式数')
    parser.add_argument('--concurrency', type=int, default=8,
                        help='并发 API 调用数')
    parser.add_argument('--min-tokens', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=35)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--config', default=None,
                        help='配置文件路径（INI），默认查找 openai_api.ini')
    parser.add_argument('--api-key',  default=None)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--model',    default=None,
                        help='模型 ID，默认 deepseek-chat')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='生成温度，越高越随机（默认 1.2）')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
