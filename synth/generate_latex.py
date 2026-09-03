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
用 Claude API (Haiku) 针对错误类型批量生成 LaTeX 公式。

用法:
  python3 generate_latex_claude.py \
    --error-types case_mismatch bracket_confusion \
    --n 4000 \
    --out ../data/generated_latex/round2.jsonl

错误类型对应 prompt 模板，见 ERROR_PROMPTS。
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import configparser
import os

import anthropic

# ── 错误类型 → 提示词模板 ────────────────────────────────────────── #

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


# ── Claude API 调用 ──────────────────────────────────────────────── #

async def generate_batch(
    client: anthropic.AsyncAnthropic,
    error_type: str,
    n_per_call: int,
    semaphore: asyncio.Semaphore,
    model: str = 'claude-haiku-4-5-20251001',
    use_control: bool = False,
) -> list[str]:
    """一次 API 调用生成一批公式，带重试。"""
    prompt_cfg = ERROR_PROMPTS[error_type]
    user_key = "user_control" if (use_control and "user_control" in prompt_cfg) else "user"
    user_msg = prompt_cfg[user_key].format(n=n_per_call)

    for attempt in range(4):
        try:
            async with semaphore:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=prompt_cfg["system"],
                    messages=[{"role": "user", "content": user_msg}],
                )
            text = resp.content[0].text.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return lines
        except anthropic.RateLimitError:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  [RateLimit] 等待 {wait:.1f}s ...", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  [Error] {e}, attempt={attempt}", flush=True)
            await asyncio.sleep(2 ** attempt)

    return []


def clean_latex(s: str) -> str:
    """清洗单条 LaTeX：去掉 markdown 代码块标记等。"""
    s = s.strip()
    # 去掉 ```latex ... ``` 包裹
    s = re.sub(r'^```[a-z]*\n?', '', s)
    s = re.sub(r'\n?```$', '', s)
    # 去掉行首的数字序号 "1. " "1) "
    s = re.sub(r'^\d+[\.\)]\s*', '', s)
    # 去掉 $ 符号
    s = s.replace('$', '').strip()
    return s


_BANNED_CMDS = re.compile(
    r'\\(?:text|mathbf|mathbb|vec|boldsymbol|mathrm|mathcal|mathit|operatorname)\{'
    r'|\\[,;!]'
    r'|\\(?:quad|qquad|hspace|vspace|medspace|thickspace|thinspace)\b'
)

def filter_formula(s: str, vocab_tokens: set,
                   min_tok: int = 4, max_tok: int = 35) -> bool:
    """过滤不合格公式。"""
    if not s:
        return False
    if _BANNED_CMDS.search(s):
        return False
    # 长度过滤（按空格粗略估计 token 数）
    # 先做简单 tokenize：按空格 + 单字符切
    toks = re.findall(r'\\[a-zA-Z]+|\{|\}|\^|_|\d+|[a-zA-Z]|[^\s]', s)
    if not (min_tok <= len(toks) <= max_tok):
        return False
    # 必须含至少一个已知 vocab token（否则可能是乱码）
    if vocab_tokens:  # vocab 为空时跳过此检查
        known = sum(1 for t in toks if t in vocab_tokens)
        if known < len(toks) * 0.3:
            return False
    # 括号平衡检查
    depth = 0
    for c in s:
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        if depth < 0:
            return False
    if depth != 0:
        return False
    return True


def dedup(formulas: list[str]) -> list[str]:
    """去重（按内容 hash）。"""
    seen = set()
    result = []
    for f in formulas:
        h = hashlib.md5(f.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(f)
    return result


# ── 主流程 ──────────────────────────────────────────────────────── #

def load_config(config_path: str | None) -> dict:
    """从 INI 配置文件读取 API 设置。"""
    cfg = {}
    # 默认查找路径：脚本同目录下的 claude_api.ini
    candidates = [
        config_path,
        str(Path(__file__).parent / 'claude_api.ini'),
        str(Path.home() / '.claude_api.ini'),
    ]
    for path in candidates:
        if path and Path(path).exists():
            parser = configparser.ConfigParser()
            parser.read(path)
            sec = parser['claude'] if 'claude' in parser else parser.defaults()
            cfg['api_key']  = sec.get('api_key', '').strip() or None
            cfg['base_url'] = sec.get('base_url', '').strip() or None
            cfg['model']    = sec.get('model', '').strip() or None
            print(f"[Config] 从 {path} 加载配置")
            break
    return cfg


async def main_async(args):
    # 优先级：命令行参数 > 配置文件 > 环境变量
    file_cfg = load_config(args.config)
    api_key  = args.api_key  or file_cfg.get('api_key')  or os.environ.get('ANTHROPIC_API_KEY')
    base_url = args.base_url or file_cfg.get('base_url')
    model    = args.model    or file_cfg.get('model') or 'claude-haiku-4-5-20251001'

    client_kwargs = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    if base_url:
        client_kwargs['base_url'] = base_url
    client = anthropic.AsyncAnthropic(**client_kwargs)

    # 加载 vocab token 集合
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'train'))
    try:
        from mathwriting_dataset import get_vocabulary
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'mathwriting-2024'
        vocab = get_vocabulary(data_dir)
        vocab_tokens = set(vocab.idx2token)
        if len(vocab_tokens) <= 4:  # 只有特殊 token，说明加载到了空 vocab
            raise ValueError("vocab 为空，跳过过滤")
        print(f"[Config] vocab size={len(vocab_tokens)}")
    except Exception as e:
        vocab_tokens = set()
        print(f"[Warning] 无法加载 vocab（{e}），跳过 vocab 过滤")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有去重集合（增量模式）
    existing = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path) as f:
            for line in f:
                d = json.loads(line)
                existing.add(d['formula'])
        print(f"[已有] {len(existing)} 条，增量追加模式")

    error_types = args.error_types
    n_total = args.n
    n_per_type = n_total // len(error_types)
    n_per_call = min(args.batch, 50)

    semaphore = asyncio.Semaphore(args.concurrency)
    all_results: dict[str, list[str]] = {et: [] for et in error_types}

    # 增量写入：每轮 API 批次完成后立即追加，断电/崩溃只丢失当前未完成的批次
    out_mode = 'w' if args.overwrite or not out_path.exists() else 'a'
    out_file = open(out_path, out_mode)
    total_written = 0

    try:
        for error_type in error_types:
            if error_type not in ERROR_PROMPTS:
                print(f"[跳过] 未知错误类型: {error_type}")
                continue

            print(f"\n[{error_type}] 目标 {n_per_type} 条新公式 ...")
            has_control = "user_control" in ERROR_PROMPTS[error_type]
            call_counter = 0

            while len(all_results[error_type]) < n_per_type:
                need = n_per_type - len(all_results[error_type])
                n_calls = max(1, (need + n_per_call - 1) // n_per_call)
                n_calls = min(n_calls, args.concurrency * 2)

                tasks = [
                    generate_batch(client, error_type, n_per_call, semaphore, model,
                                   use_control=(has_control and (call_counter + i) % 2 == 1))
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
                                and filter_formula(formula, vocab_tokens,
                                                  args.min_tokens, args.max_tokens)):
                            new_batch.append(formula)
                            existing.add(formula)

                new_batch = dedup(new_batch)
                all_results[error_type].extend(new_batch)

                # 本轮新增立即写入文件
                if new_batch:
                    for formula in new_batch:
                        out_file.write(json.dumps({
                            "formula": formula,
                            "error_type": error_type,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--error-types', nargs='+',
                        default=['upper_lower_mix', 'bracket_confusion', 'mixed_brackets',
                                 'greek_lookalike', 'symbol_confusion',
                                 'insertion', 'digit_letter', 'mixed_hard'],
                        choices=list(ERROR_PROMPTS.keys()),
                        help='要生成的错误类型')
    parser.add_argument('--n', type=int, default=3000,
                        help='总目标生成数量（按类型平均分配）')
    parser.add_argument('--out', default='../data/generated_latex/round1.jsonl')
    parser.add_argument('--batch', type=int, default=200,
                        help='每次 API 调用请求的公式数')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='并发 API 调用数')
    parser.add_argument('--min-tokens', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=35)
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已有输出（否则追加）')
    parser.add_argument('--config', default=None,
                        help='配置文件路径（INI 格式），默认查找 claude_api.ini')
    parser.add_argument('--api-key', default=None,
                        help='Anthropic API key（覆盖配置文件）')
    parser.add_argument('--base-url', default=None,
                        help='自定义 API base URL（走代理/中转时使用，覆盖配置文件）')
    parser.add_argument('--model', default=None,
                        help='使用的模型 ID（覆盖配置文件）')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
