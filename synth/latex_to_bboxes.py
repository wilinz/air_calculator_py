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

# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── (font_prefix4, char_code) → symbol library token ─────────────── #
# font_prefix: first 4 chars of font name (cmmi, cmsy, cmex, cmr, ...)
# Determined empirically with lualatex node walk

_FC2T: dict = {
    # ── cmmi (CM Math Italic = OML) ──
    # Uppercase Greek (0-10)
    ('cmmi', 0):  r'\Gamma',   ('cmmi', 1):  r'\Delta',  ('cmmi', 2):  r'\Theta',
    ('cmmi', 3):  r'\Lambda',  ('cmmi', 4):  r'\Xi',     ('cmmi', 5):  r'\Pi',
    ('cmmi', 6):  r'\Sigma',   ('cmmi', 7):  r'\Upsilon',('cmmi', 8):  r'\Phi',
    ('cmmi', 9):  r'\Psi',     ('cmmi', 10): r'\Omega',
    # Lowercase Greek (11-33)
    ('cmmi', 11): r'\alpha',   ('cmmi', 12): r'\beta',   ('cmmi', 13): r'\gamma',
    ('cmmi', 14): r'\delta',   ('cmmi', 15): r'\epsilon',('cmmi', 16): r'\zeta',
    ('cmmi', 17): r'\eta',     ('cmmi', 18): r'\theta',  ('cmmi', 19): r'\iota',
    ('cmmi', 20): r'\kappa',   ('cmmi', 21): r'\lambda', ('cmmi', 22): r'\mu',
    ('cmmi', 23): r'\nu',      ('cmmi', 24): r'\xi',     ('cmmi', 25): r'\pi',
    ('cmmi', 26): r'\rho',     ('cmmi', 27): r'\sigma',  ('cmmi', 28): r'\tau',
    ('cmmi', 29): r'\upsilon', ('cmmi', 30): r'\phi',    ('cmmi', 31): r'\chi',
    ('cmmi', 32): r'\psi',     ('cmmi', 33): r'\omega',
    # Variant Greek
    ('cmmi', 35): r'\vartheta',('cmmi', 36): r'\varpi',
    ('cmmi', 38): r'\varsigma',('cmmi', 39): r'\varphi',
    # Other cmmi
    ('cmmi', 64): r'\partial',
    ('cmmi', 126): r'\vec',    # vector arrow accent in cmmi
    # Letters a-z, A-Z (same as ASCII)
    **{('cmmi', c): chr(c) for c in range(65, 91)},
    **{('cmmi', c): chr(c) for c in range(97, 123)},

    # ── cmsy (CM Math Symbol = OMS) ──
    ('cmsy', 0):  '-',         # minus (OMS)
    ('cmsy', 1):  r'\cdot',    ('cmsy', 2):  r'\times',  ('cmsy', 4):  r'\div',
    ('cmsy', 6):  r'\pm',      ('cmsy', 7):  r'\mp',
    ('cmsy', 8):  r'\oplus',   ('cmsy', 9):  r'\ominus', ('cmsy', 10): r'\otimes',
    ('cmsy', 11): r'\emptyset',('cmsy', 12): r'\odot',
    ('cmsy', 14): r'\circ',    ('cmsy', 15): r'\bullet',
    ('cmsy', 17): r'\equiv',
    ('cmsy', 18): r'\subseteq',('cmsy', 19): r'\supseteq',
    ('cmsy', 20): r'\le',      ('cmsy', 21): r'\ge',
    ('cmsy', 24): r'\sim',     ('cmsy', 25): r'\approx',
    ('cmsy', 26): r'\subset',  ('cmsy', 27): r'\supset',
    ('cmsy', 28): r'\ll',      ('cmsy', 29): r'\gg',
    ('cmsy', 32): r'\leftarrow',('cmsy', 33): r'\rightarrow',
    ('cmsy', 36): r'\leftrightarrow',
    ('cmsy', 39): r'\simeq',
    ('cmsy', 41): r'\Rightarrow',
    ('cmsy', 44): r'\Leftrightarrow',
    ('cmsy', 47): r'\propto',
    ('cmsy', 48): r'\prime',
    ('cmsy', 49): r'\infty',
    ('cmsy', 50): r'\in',
    ('cmsy', 51): r'\ni',
    ('cmsy', 54): r'\ne',      # first glyph of \ne composite (skip second)
    ('cmsy', 56): r'\forall',  ('cmsy', 57): r'\exists',
    ('cmsy', 58): r'\cap',     ('cmsy', 59): r'\cup',
    ('cmsy', 62): r'\perp',    ('cmsy', 63): r'\angle',
    ('cmsy', 64): r'\aleph',
    ('cmsy', 91): r'\wedge',   ('cmsy', 92): r'\vee',
    ('cmsy', 94): r'\wedge',   ('cmsy', 95): r'\vee',
    ('cmsy', 96): r'\vdash',   ('cmsy', 106): r'\models',
    ('cmsy', 98): r'\lfloor',  ('cmsy', 99): r'\rfloor',
    ('cmsy', 100): r'\lceil',  ('cmsy', 101): r'\rceil',
    ('cmsy', 104): r'\langle', ('cmsy', 105): r'\rangle',
    ('cmsy', 112): r'\sqrt',   # radical sign glyph
    ('cmsy', 114): r'\nabla',
    ('cmsy', 102): r'\{',    ('cmsy', 103): r'\}',
    ('cmsy', 106): '|',      ('cmsy', 107): r'\|',
    ('cmsy', 121): r'\dagger',

    # ── cmex (CM Math Extension = CMEX, large operators & delimiters) ──
    # Radical sign glyphs (single large char or extensible multi-piece):
    #   cmsy 112 = small \sqrt (already in cmsy section)
    #   cmex 114 = medium \sqrt (single glyph, confirmed empirically)
    #   cmex 116 = extensible radical foot (bottom checkmark piece)
    #   cmex 117 = extensible radical extension (vertical fill piece, may repeat)
    #   cmex 118 = extensible radical top (top piece that touches the bar)
    # Consecutive \sqrt tokens are merged in _nodes_to_bboxes.
    ('cmex', 114): r'\sqrt',
    ('cmex', 115): r'\sqrt',  # potential intermediate size
    ('cmex', 116): r'\sqrt',
    ('cmex', 117): r'\sqrt',
    ('cmex', 118): r'\sqrt',
    # Large parentheses (size 1–3, char codes 0/1, 16/17, 18/19, 20/21)
    ('cmex', 0):  '(',  ('cmex', 1):  ')',
    ('cmex', 16): '(',  ('cmex', 17): ')',
    ('cmex', 18): '(',  ('cmex', 19): ')',
    ('cmex', 20): '(',  ('cmex', 21): ')',
    # Large square brackets (104/105, 106/107)
    ('cmex', 104): '[', ('cmex', 105): ']',
    ('cmex', 106): '[', ('cmex', 107): ']',
    # Large curly braces: 110/111 confirmed empirically (\left\{\frac{a}{b}\right\})
    ('cmex', 110): r'\{', ('cmex', 111): r'\}',
    # cmex 112/113 are radical signs for medium-sized \sqrt expressions.
    # Confirmed empirically:
    #   cmex 112: \left(\sqrt{x^2+1}\right) → G cmex 112 (height=0.4, depth=11.6)
    #   cmex 113: \sqrt{\tilde{a}^2+\overline{b}^2+3} → G cmex 113 (height=0.4, depth=17.6)
    #             \sqrt{x+\sqrt{y+\sqrt{z}}} inner radicals also use cmex 113
    ('cmex', 112): r'\sqrt',
    ('cmex', 113): r'\sqrt',
    # Extensible bracket parts (top/mid/bottom) — map to closest bracket
    ('cmex', 8):  '(',  ('cmex', 9):  ')',   # top/bot parts of paren
    ('cmex', 10): '[',  ('cmex', 11): ']',
    # cmex 12 = extensible | piece (confirmed: \bigg| uses these)
    # cmex 13 = extensible \| (double bar) piece
    ('cmex', 12): '|',  ('cmex', 13): r'\|',
    # Vertical bars (large single-glyph forms)
    ('cmex', 55): '|',  ('cmex', 56): r'\|',
    # text-size operators
    ('cmex', 80): r'\sum',     ('cmex', 81): r'\prod',   ('cmex', 82): r'\int',
    ('cmex', 83): r'\bigoplus',('cmex', 84): r'\bigcup', ('cmex', 72): r'\bigcap',
    ('cmex', 76): r'\bigvee',  ('cmex', 87): r'\bigwedge',
    ('cmex', 73): r'\iint',    ('cmex', 78): r'\oint',
    # display-size (char code + 8)
    ('cmex', 88): r'\sum',     ('cmex', 89): r'\prod',   ('cmex', 90): r'\int',
    ('cmex', 91): r'\bigoplus',

    # ── cmr (CM Roman = OT1) ──
    # Digits (ASCII codes in math roman font)
    **{('cmr', c): chr(c) for c in range(48, 58)},
    # Punctuation and operators
    ('cmr', 33): '!',   ('cmr', 40): '(',  ('cmr', 41): ')',
    ('cmr', 42): '*',   ('cmr', 43): '+',  ('cmr', 44): ',',
    ('cmr', 45): '-',   ('cmr', 46): '.',  ('cmr', 47): '/',
    ('cmr', 58): ':',   ('cmr', 59): ';',  ('cmr', 60): '<',
    ('cmr', 61): '=',   ('cmr', 62): '>',  ('cmr', 63): '?',
    ('cmr', 91): '[',   ('cmr', 93): ']',  ('cmr', 124): '|',
    # Accent glyphs in CM Roman
    ('cmr', 94):  r'\hat',    # hat accent (^)
    ('cmr', 126): r'\tilde',  # tilde accent (~)
    # Letters (italic variant in some contexts)
    **{('cmr', c): chr(c) for c in range(65, 91)},
    **{('cmr', c): chr(c) for c in range(97, 123)},

    # ── msam/msbm (AMS fonts) ── skip most, keep a few
    # ('msam', 44): r'\triangleq',  # too rare to worry about
}

# Track which (fp,char) pairs follow immediately after a \ne first glyph,
# so we can skip the '=' of \ne without skipping standalone '='.
# Handled dynamically in _nodes_to_bboxes via prev_token tracking.
_NE_SECOND = ('cmr', 61)   # second glyph of \ne is '=' from cmr

_SCALE = 100.0  # lualatex pt → output units


# ── lualatex document template ────────────────────────────────────── #
# Batch template: reuse \fmlbox for each formula, prefix NODE output with formula index.

_TEX_HEADER = r"""
\documentclass{standalone}
\usepackage{amsmath,amssymb,luacode}
\begin{luacode*}
local sp = function(v) return v / 65536.0 end

local function fp(fid)
    local f = font.getfont(fid)
    local n = (f and f.name) or 'unkn'
    if n:sub(1,4) == 'cmmi' then return 'cmmi'
    elseif n:sub(1,4) == 'cmsy' then return 'cmsy'
    elseif n:sub(1,4) == 'cmex' then return 'cmex'
    elseif n:sub(1,3) == 'cmr'  then return 'cmr'
    elseif n:sub(1,4) == 'msam' then return 'msam'
    elseif n:sub(1,4) == 'msbm' then return 'msbm'
    else return n:sub(1,4) end
end

-- Compute the actual set width of a glue node inside a box.
-- gs/gset/go: glue_sign, glue_set, glue_order of the enclosing hlist.
local function glue_width(n, gs, gset, go)
    local w = n.width or 0
    if gs == 1 and go == (n.stretch_order or 0) then
        w = w + gset * (n.stretch or 0)
    elseif gs == 2 and go == (n.shrink_order or 0) then
        w = w - gset * (n.shrink or 0)
    end
    return w
end

local GLUE_ID = node.id('glue')
local KERN_ID = node.id('kern')
local GLYPH_ID = node.id('glyph')
local HLIST_ID = node.id('hlist')
local VLIST_ID = node.id('vlist')
local RULE_ID  = node.id('rule')

-- Forward declarations for mutual recursion between walk and walk_vlist_children.
local walk
local walk_vlist_children

-- Walk the children of a vlist node 'vn' starting at vertical position vy_top.
-- Rules are tagged 'RF' (fraction bar: an hlist was seen before the rule in this
-- vlist scope) or 'RO' (overline/vinculum: rule appears before any hlist).
-- Handles nested VLIST_ID children recursively (e.g. \overline inside \sqrt).
walk_vlist_children = function(vn, x, vy, out, debug)
    local m = vn.head
    local saw_hlist = false
    while m do
        if m.id == HLIST_ID then
            saw_hlist = true
            walk(m.head, x, vy - m.height, out, debug,
                 m.glue_sign or 0, m.glue_set or 0.0, m.glue_order or 0)
            vy = vy - m.height - m.depth
        elseif m.id == VLIST_ID then
            -- Nested vlist (e.g. \overline{x^2} inside a \sqrt vlist).
            -- Recurse: the nested vlist's top is at vy - shift + height.
            local inner_vshift = m.shift or 0
            local inner_vy = vy - inner_vshift + m.height
            walk_vlist_children(m, x, inner_vy, out, debug)
            vy = vy - m.height - m.depth
        elseif m.id == KERN_ID then
            vy = vy - m.kern
        elseif m.id == GLUE_ID then
            -- Vertical glue: use the parent vlist's stretch/shrink context.
            local vw = m.width or 0
            local vgs = vn.glue_sign or 0
            local vgo = vn.glue_order or 0
            local vgset = vn.glue_set or 0.0
            if vgs == 1 and vgo == (m.stretch_order or 0) then
                vw = vw + vgset * (m.stretch or 0)
            elseif vgs == 2 and vgo == (m.shrink_order or 0) then
                vw = vw - vgset * (m.shrink or 0)
            end
            vy = vy - vw
        elseif m.id == RULE_ID then
            -- Tag: 'RF' = fraction bar (hlist seen before this rule in scope),
            --       'RO' = overline/vinculum (rule before any hlist in scope).
            -- Use the parent vlist's width for the rule (visual width of bar).
            local rtag = saw_hlist and 'RF' or 'RO'
            out[#out+1] = string.format(
                rtag .. ' %.4f %.4f %.4f %.4f %.4f',
                sp(x), sp(vy - m.height),
                sp(vn.width), sp(m.height), sp(m.depth))
            vy = vy - m.height - m.depth
        end
        m = m.next
    end
end

-- gs/gset/go: glue context of the enclosing hlist (for \hfil centering glue).
walk = function(n, x, y, out, debug, gs, gset, go)
    gs, gset, go = gs or 0, gset or 0.0, go or 0
    while n do
        local id = n.id
        if id == GLYPH_ID then
            out[#out+1] = string.format(
                'G %s %d %.4f %.4f %.4f %.4f %.4f',
                fp(n.font), n.char, sp(x), sp(y),
                sp(n.width), sp(n.height), sp(n.depth))
            x = x + n.width
        elseif id == GLUE_ID then
            -- Advance x by the glue's actual set width (handles \hfil centering).
            x = x + glue_width(n, gs, gset, go)
        elseif id == HLIST_ID then
            if debug then
                out[#out+1] = string.format(
                    'DH %.4f %.4f %.4f %.4f %.4f',
                    sp(x), sp(y), sp(n.width), sp(n.height), sp(n.shift))
            end
            -- Pass the hlist's own glue context when descending into it.
            walk(n.head, x, y - n.shift, out, debug,
                 n.glue_sign or 0, n.glue_set or 0.0, n.glue_order or 0)
            x = x + n.width
        elseif id == VLIST_ID then
            local vshift = n.shift or 0
            -- vshift: positive = lower. Top of vlist = y - vshift + n.height.
            local vy = y - vshift + n.height
            if debug then
                out[#out+1] = string.format(
                    'DV %.4f %.4f %.4f %.4f %.4f shift=%.4f vy=%.4f',
                    sp(x), sp(y), sp(n.width), sp(n.height), sp(n.depth),
                    sp(vshift), sp(vy * 65536))
            end
            walk_vlist_children(n, x, vy, out, debug)
            x = x + n.width
        elseif id == KERN_ID then
            x = x + n.kern
        elseif id == RULE_ID then
            -- Inline rule inside an hlist (rare in math mode); keep old 'R' tag.
            out[#out+1] = string.format(
                'R %.4f %.4f %.4f %.4f %.4f',
                sp(x), sp(y), sp(n.width), sp(n.height), sp(n.depth))
            x = x + n.width
        end
        n = n.next
    end
end

function WALK_BOX(reg, fml_id)
    local out = {}
    local ok, err = pcall(function()
        local b = tex.box[reg]
        if b and b.head then walk(b.head, 0, 0, out, false) end
    end)
    local prefix = 'NODE:' .. fml_id .. ':'
    if ok then
        for _,r in ipairs(out) do
            tex.print('\\typeout{' .. prefix .. r .. '}')
        end
    end
end

function WALK_BOX_DEBUG(reg, fml_id)
    local out = {}
    local ok, err = pcall(function()
        local b = tex.box[reg]
        if b and b.head then walk(b.head, 0, 0, out, true) end
    end)
    local prefix = 'NODE:' .. fml_id .. ':'
    if ok then
        for _,r in ipairs(out) do
            tex.print('\\typeout{' .. prefix .. r .. '}')
        end
    else
        tex.print('\\typeout{NODE:' .. fml_id .. ':ERROR ' .. tostring(err) .. '}')
    end
end
\end{luacode*}

\newsavebox{\fmlbox}
\begin{document}
"""

_TEX_FOOTER = r"\end{document}" + "\n"

# One formula block — placeholder FML_ID and FML_FORMULA are replaced at runtime
_TEX_FORMULA_BLOCK = (
    r"\savebox{\fmlbox}{$\displaystyle FML_FORMULA$}" + "\n"
    r"\directlua{WALK_BOX(\the\fmlbox, FML_ID)}" + "\n"
)

# Debug block: uses WALK_BOX_DEBUG to also emit DH/DV lines
_TEX_FORMULA_BLOCK_DEBUG = (
    r"\savebox{\fmlbox}{$\displaystyle FML_FORMULA$}" + "\n"
    r"\directlua{WALK_BOX_DEBUG(\the\fmlbox, FML_ID)}" + "\n"
)


def _build_batch_tex(formulas: list) -> str:
    """Build a lualatex document that processes all formulas in one run."""
    parts = [_TEX_HEADER]
    for i, fml in enumerate(formulas):
        block = _TEX_FORMULA_BLOCK.replace('FML_ID', str(i)).replace('FML_FORMULA', fml)
        parts.append(block)
    # Minimal visible output: render the first formula (standalone requires output)
    parts.append(r"\usebox{\fmlbox}" + "\n")
    parts.append(_TEX_FOOTER)
    return ''.join(parts)


def _run_lualatex_batch(formulas: list, workdir: Path) -> dict:
    """Run lualatex once for all formulas. Returns {idx: [raw_node_lines]}."""
    tex = _build_batch_tex(formulas)
    tex_file = workdir / 'fml.tex'
    tex_file.write_text(tex, encoding='utf-8')

    result = subprocess.run(
        ['lualatex', '-interaction=nonstopmode', str(tex_file)],
        cwd=workdir, capture_output=True, text=True, timeout=120,
    )
    lines = (result.stdout + result.stderr).splitlines()

    # Parse "NODE:ID:TYPE data" lines
    by_id: dict = {}
    for line in lines:
        if not line.startswith('NODE:'):
            continue
        rest = line[5:]          # strip "NODE:"
        colon = rest.find(':')
        if colon < 0:
            continue
        fml_id = int(rest[:colon])
        node_line = rest[colon + 1:]
        by_id.setdefault(fml_id, []).append(node_line)
    return by_id


def _parse_nodes(lines: list) -> list:
    nodes = []
    for l in lines:
        parts = l.split()
        if not parts:
            continue
        if parts[0] == 'G' and len(parts) >= 8:
            nodes.append({
                'kind': 'G',
                'fp': parts[1][:4],       # font prefix (4 chars)
                'char': int(parts[2]),
                'x': float(parts[3]), 'y': float(parts[4]),
                'w': float(parts[5]), 'h': float(parts[6]), 'd': float(parts[7]),
            })
        elif parts[0] in ('R', 'RF', 'RO') and len(parts) >= 6:
            nodes.append({
                'kind': 'R',
                'rule_type': parts[0],   # 'R'=inline, 'RF'=fraction bar, 'RO'=overline
                'x': float(parts[1]), 'y': float(parts[2]),
                'w': float(parts[3]), 'h': float(parts[4]), 'd': float(parts[5]),
            })
    return nodes


def _glyph_to_token(fp: str, char: int) -> Optional[str]:
    key = (fp, char)
    if key in _FC2T:
        return _FC2T[key]
    # fallback: ASCII printable
    if 32 < char < 127:
        c = chr(char)
        if c == '{': return r'\{'
        if c == '}': return r'\}'
        return c
    return None


def _classify_rule(rule: dict, all_nodes: list) -> str:
    rx, ry, rw = rule['x'], rule['y'], rule['w']
    glyphs = [n for n in all_nodes if n['kind'] == 'G']

    # TeX overline/underline rules often have "fill" width stored as large negative
    if rw <= 0 or rw > 500:
        # Estimate from nearby glyphs (within 8pt vertically of this rule)
        nearby = [n for n in glyphs if abs(n['y'] - ry) < 8.0]
        if nearby:
            x0 = min(n['x'] for n in nearby)
            x1 = max(n['x'] + n['w'] for n in nearby)
            rx, rw = x0, max(x1 - x0, 1.0)
        else:
            rw = 20.0

    def in_xrange(n):
        gx = n['x'] + n['w'] / 2
        return rx - 2.0 < gx < rx + rw + 2.0

    above = [n for n in glyphs if n['y'] > ry + 0.5 and in_xrange(n)]
    below = [n for n in glyphs if n['y'] < ry - 0.5 and in_xrange(n)]

    if above and below:
        return r'\frac'
    if below and not above:   # bar is above text → overline
        return r'\overline'
    if above and not below:   # bar is below text → underline
        return r'\underline'
    return r'\frac'


def _nodes_to_bboxes(nodes: list) -> list:
    bboxes = []
    s = _SCALE
    prev_was_ne = False  # True after we output \ne, so next '=' from cmr is skipped

    for node in nodes:
        if node['kind'] == 'G':
            fp, char = node['fp'], node['char']

            # Skip the '=' that is the second glyph of \ne composite
            if prev_was_ne and (fp, char) == _NE_SECOND:
                prev_was_ne = False
                continue
            prev_was_ne = False

            token = _glyph_to_token(fp, char)
            if token is None:
                continue

            if token == r'\ne':
                prev_was_ne = True

            x, y, w, h, d = node['x'], node['y'], node['w'], node['h'], node['d']
            # lualatex: y positive = above baseline
            # synth-bboxes: yMin<0 = above baseline, yMax=baseline for baseline chars
            bboxes.append({
                'token': token,
                'xMin': round(x * s, 2),
                'yMin': round(-(y + h) * s, 2),
                'xMax': round((x + w) * s, 2),
                'yMax': round(-(y - d) * s, 2),
            })

        elif node['kind'] == 'R':
            rule_type = node.get('rule_type', 'R')
            if rule_type == 'RF':
                # Fraction bar: hlist seen before rule in this vlist scope.
                token = r'\frac'
            elif rule_type == 'RO':
                # Overline/vinculum: rule before any hlist in scope → always overline.
                # (_merge_sqrt_and_overline will absorb it into a preceding \sqrt.)
                token = r'\overline'
            else:
                # Inline rule or legacy 'R': use heuristic above/below classification.
                token = _classify_rule(node, nodes)
            x, y, w, h, d = node['x'], node['y'], node['w'], node['h'], node['d']
            if w <= 0:
                w = 1.0
            bboxes.append({
                'token': token,
                'xMin': round(x * s, 2),
                'yMin': round(-(y + h) * s, 2),
                'xMax': round((x + w) * s, 2),
                'yMax': round(-(y - d) * s, 2),
            })

    # Post-process: drop thin vertical rules misclassified as \overline or \frac.
    # Real horizontal bars have width >> height. Strut/hairline rules inside \binom,
    # \genfrac etc. are portrait-shaped (height >> width) and should be discarded.
    _filtered = []
    for b in bboxes:
        if b['token'] in (r'\overline', r'\frac'):
            bw = b['xMax'] - b['xMin']
            bh = b['yMax'] - b['yMin']
            if bh > 0 and bw / bh < 0.15:   # much taller than wide → vertical strut
                continue
        _filtered.append(b)
    bboxes = _filtered

    # Post-process: merge consecutive | tokens (extensible \bigg| uses multiple pieces)
    merged_bars: list = []
    i = 0
    while i < len(bboxes):
        if bboxes[i]['token'] == '|':
            b = bboxes[i].copy()
            j = i + 1
            while j < len(bboxes) and bboxes[j]['token'] == '|':
                o = bboxes[j]
                b['xMin'] = min(b['xMin'], o['xMin'])
                b['yMin'] = min(b['yMin'], o['yMin'])
                b['xMax'] = max(b['xMax'], o['xMax'])
                b['yMax'] = max(b['yMax'], o['yMax'])
                j += 1
            merged_bars.append(b)
            i = j
        else:
            merged_bars.append(bboxes[i])
            i += 1
    bboxes = merged_bars

    # Post-process: clip accent (tilde/hat/vec) bbox so it sits above the base letter
    # rather than overlapping it. TeX overlays accents at the same baseline using a
    # negative kern, so the raw bbox spans from baseline up — but the ink is at the top.
    _ACCENT_TOKENS = frozenset({r'\tilde', r'\hat', r'\vec', r'\widehat', r'\widetilde'})
    _MIN_ACCENT_HEIGHT = 100.0  # minimum accent height in output units
    for i in range(len(bboxes)):
        b = bboxes[i]
        if b['token'] not in _ACCENT_TOKENS:
            continue
        # Find the base letter: nearest following bbox that overlaps in x-range
        for j in range(i + 1, min(i + 5, len(bboxes))):
            nb = bboxes[j]
            if nb['xMin'] < b['xMax'] and nb['xMax'] > b['xMin']:
                # Clip accent's yMax to the base letter's yMin (top of base glyph body)
                new_ymax = nb['yMin']
                height = new_ymax - b['yMin']
                if height < _MIN_ACCENT_HEIGHT:
                    new_ymax = b['yMin'] + _MIN_ACCENT_HEIGHT
                # Never push yMax beyond the original bbox
                if new_ymax > b['yMax']:
                    new_ymax = b['yMax']
                bboxes[i] = {**b, 'yMax': new_ymax}
                break

    # Post-process: merge multi-piece radical glyphs and absorb overline into sqrt.
    bboxes = _merge_sqrt_and_overline(bboxes)

    # Post-process: fix misplaced \overline bars inside nested structures.
    # When \overline{X} appears inside a fraction denominator + sqrt, the bar's y
    # coordinate can be shifted incorrectly (appears in numerator area instead of
    # sitting just above the glyph). Fix: if the bar's bottom (yMax) is more than
    # 50 units away from the glyph's top (yMin), reposition it.
    for i in range(len(bboxes) - 1):
        b = bboxes[i]
        if b['token'] != r'\overline':
            continue
        nxt = bboxes[i + 1]
        if not (b['xMin'] < nxt['xMax'] and b['xMax'] > nxt['xMin']):
            continue
        bar_h = b['yMax'] - b['yMin']
        glyph_top = nxt['yMin']
        if abs(b['yMax'] - glyph_top) > 50:
            bboxes[i] = {**b, 'yMin': glyph_top - bar_h, 'yMax': glyph_top}

    # Post-process: fix nth-root index (e.g. \sqrt[4]{x}).
    # Must run AFTER _merge_sqrt_and_overline so \sqrt has its final merged bbox.
    # The root index glyph appears just before \sqrt, inside its x range (checkmark area).
    # When \sqrt is stretched to fill its full bbox, the vinculum bar covers the index.
    # Fix: clip \sqrt's xMin to index.xMax so the radical starts after the index.
    # Content overlap with the hook is handled by piecewise rendering in synth_from_bboxes.py.
    for i in range(1, len(bboxes)):
        b = bboxes[i]
        if b['token'] != r'\sqrt':
            continue
        prev = bboxes[i - 1]
        sqrt_w = b['xMax'] - b['xMin']
        sqrt_h = b['yMax'] - b['yMin']
        if (prev['token'] not in (r'\sqrt', r'\frac', r'\overline') and
                prev['xMin'] >= b['xMin'] and
                prev['xMax'] <= b['xMin'] + sqrt_w * 0.55 and
                prev['yMin'] <= b['yMin'] + sqrt_h * 0.7):
            bboxes[i] = {**b, 'xMin': prev['xMax']}

    # Post-process: fix tokens inside a \sqrt that is in a fraction's denominator.
    #
    # In display-mode fractions like \frac{num}{\sqrt{\overline{g}}}, TeX metric
    # coordinates place denominator characters so high that their bbox top is above
    # (or at) the fraction bar, making the synthesised strokes overlap the bar.
    #
    # Fix strategy (two-pass):
    #   Pass 1 — regular chars: centre the character in the sqrt interior so it
    #     appears fully below the vinculum.  Track the new position in _moved_chars.
    #   Pass 2 — bar tokens (\overline, \underline): if the bar is above the
    #     vinculum AND we moved a character in the same x-range, reposition the bar
    #     just above that character's new bbox (so the \overline{X} structure is
    #     preserved inside the sqrt).  If no matching moved char exists, drop it.
    #
    # IMPORTANT: only tokens that appear AFTER the \sqrt in the token list are
    # touched, so numerator characters with the same x-range are never affected.
    # Applied only when a \frac before the \sqrt indicates denominator context.

    _final2: list = []
    _active_sqrts2: list = []
    _last_frac = None
    # Records chars moved during pass 1: list of (new_bbox, sq) for pass 2 use.
    _moved_chars: list = []
    # Indices in _final2 for deferred bar tokens (need repositioning after pass 1).
    _deferred_bars: list = []  # (idx_in_final2, bar_bbox, sq)

    for b in bboxes:
        if b['token'] == r'\frac':
            _last_frac = b
            _final2.append(b)
            continue
        if b['token'] == r'\sqrt':
            _active_sqrts2.append(b)
            _final2.append(b)
            continue
        if not _active_sqrts2:
            _final2.append(b)
            continue

        # Find an already-seen sqrt whose x-range contains this token
        sq = None
        for _sq in _active_sqrts2:
            if (b['xMin'] >= _sq['xMin'] - 50 and
                    b['xMax'] <= _sq['xMax'] + 50):
                sq = _sq
                break
        if sq is None:
            _final2.append(b)
            continue

        in_frac_denom = (
            _last_frac is not None and
            _last_frac['yMin'] < sq['yMin']
        )

        # ── bar tokens ────────────────────────────────────────────────
        if b['token'] in (r'\overline', r'\underline'):
            if b['yMax'] <= sq['yMin']:
                # Bar is above the vinculum.
                if in_frac_denom:
                    # Defer: may need to reposition once we know where the char lands.
                    _deferred_bars.append((len(_final2), b, sq))
                    _final2.append(b)   # placeholder; updated in pass 2
                # else: genuinely spurious → drop
                continue
            _final2.append(b)
            continue

        # ── regular characters ────────────────────────────────────────
        if in_frac_denom:
            h_src = b['yMax'] - b['yMin']
            sq_ctr = (sq['yMin'] + sq['yMax']) / 2
            new_yMin = sq_ctr - h_src / 2
            new_yMax = sq_ctr + h_src / 2
            b = {**b, 'yMin': new_yMin, 'yMax': new_yMax}
            _moved_chars.append((b, sq))
        _final2.append(b)

    # Pass 2: reposition deferred bars now that we know the chars' new positions.
    for idx, bar_b, bar_sq in _deferred_bars:
        # Find a moved char x-overlapping with this bar.
        matched = None
        for (char_b, char_sq) in _moved_chars:
            if (bar_sq is char_sq and
                    bar_b['xMin'] < char_b['xMax'] and
                    bar_b['xMax'] > char_b['xMin']):
                matched = char_b
                break
        if matched is None:
            # No associated char found → remove the placeholder.
            _final2[idx] = None
        else:
            bar_h = bar_b['yMax'] - bar_b['yMin']
            new_yMax = matched['yMin']          # bar sits just above the char
            new_yMin = new_yMax - bar_h
            _final2[idx] = {**bar_b, 'yMin': new_yMin, 'yMax': new_yMax}

    bboxes = [b for b in _final2 if b is not None]

    return bboxes


def _merge_sqrt_and_overline(bboxes: list) -> list:
    """Two-pass post-processing for \\sqrt tokens:

    Pass 1: merge consecutive \\sqrt tokens (extensible radicals use multiple
    stacked cmex glyphs; handwriting renders the whole sign as one stroke).

    Pass 2: absorb the immediately following \\overline into the \\sqrt bbox.
    In synth_from_bboxes.py the \\sqrt symbol (in _STRETCH mode) is drawn to
    fill its entire bbox including the top bar.  The overline rule from lualatex
    sits to the right of the checkmark glyph at the same yMin.  Merging them
    gives \\sqrt a bbox that spans the full formula width, so the stretched
    symbol naturally draws the checkmark on the left and the long bar across the
    content.  The separate \\overline token is then dropped.
    """
    # Pass 1: merge consecutive \sqrt tokens
    merged: list = []
    i = 0
    while i < len(bboxes):
        if bboxes[i]['token'] == r'\sqrt':
            b = bboxes[i].copy()
            j = i + 1
            while j < len(bboxes) and bboxes[j]['token'] == r'\sqrt':
                o = bboxes[j]
                b['xMin'] = min(b['xMin'], o['xMin'])
                b['yMin'] = min(b['yMin'], o['yMin'])
                b['xMax'] = max(b['xMax'], o['xMax'])
                b['yMax'] = max(b['yMax'], o['yMax'])
                j += 1
            merged.append(b)
            i = j
        else:
            merged.append(bboxes[i])
            i += 1

    # Pass 2: absorb immediately-following \overline into \sqrt
    result: list = []
    i = 0
    while i < len(merged):
        b = merged[i]
        if b['token'] == r'\sqrt' and i + 1 < len(merged):
            nxt = merged[i + 1]
            if nxt['token'] == r'\overline':
                # The overline bar sits at the same yMin as the sqrt glyph.
                # Extend sqrt's xMax (and yMin/yMax by union) to include the bar.
                b = b.copy()
                b['xMin'] = min(b['xMin'], nxt['xMin'])
                b['yMin'] = min(b['yMin'], nxt['yMin'])
                b['xMax'] = max(b['xMax'], nxt['xMax'])
                # yMax stays as sqrt's yMax (full height of checkmark)
                result.append(b)
                i += 2   # skip the overline token
                continue
        result.append(b)
        i += 1
    return result


def formula_to_bboxes(formula: str,
                      workdir: Optional[Path] = None) -> Optional[list]:
    """Process a single formula (uses a 1-element batch internally)."""
    result = formulas_to_bboxes([formula], workdir=workdir)
    return result.get(0)


def formulas_to_bboxes(formulas: list,
                       workdir: Optional[Path] = None) -> dict:
    """Process a batch of formulas in one lualatex invocation.

    Returns {idx: bboxes_list} for successfully processed formulas.
    Skipped/failed formulas are absent from the dict.
    """
    cleanup = workdir is None
    if workdir is None:
        workdir = Path(tempfile.mkdtemp())

    try:
        by_id = _run_lualatex_batch(formulas, workdir)
        result = {}
        for idx, raw_lines in by_id.items():
            if not raw_lines:
                continue
            nodes = _parse_nodes(raw_lines)
            if not nodes:
                continue
            bboxes = _nodes_to_bboxes(nodes)
            if bboxes:
                result[idx] = bboxes
        return result
    except subprocess.TimeoutExpired:
        return {}
    except Exception as e:
        print(f'[bbox] BATCH ERROR: {e}', file=sys.stderr)
        return {}
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)


def _process_batch(batch_args):
    """Worker: run one lualatex batch, return list of (global_idx, record_or_None)."""
    start_idx, items_slice = batch_args
    formulas = [item.get('formula', '') for item in items_slice]
    # Run all formulas in one lualatex call
    bbox_map = formulas_to_bboxes(formulas)
    results = []
    for local_idx, item in enumerate(items_slice):
        global_idx = start_idx + local_idx
        formula = item.get('formula', '')
        error_type = item.get('error_type', 'unknown')
        bboxes = bbox_map.get(local_idx)
        if not formula or not bboxes:
            results.append((global_idx, None))
        else:
            record = {
                'label': formula,
                'normalizedLabel': formula,
                'error_type': error_type,
                'bboxes': bboxes,
            }
            results.append((global_idx, record))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=min(os.cpu_count() or 4, 8),
                        help='Number of parallel lualatex workers (default: min(cpu_count, 8))')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Formulas per lualatex invocation (default: 50)')
    args = parser.parse_args()

    items = []
    with open(args.input) as f:
        for line in f:
            items.append(json.loads(line))

    total = len(items)
    batch_size = args.batch_size
    batches = [
        (i, items[i:i + batch_size])
        for i in range(0, total, batch_size)
    ]
    print(f'[latex_to_bboxes] {total} formulas, {len(batches)} batches x{batch_size}, '
          f'{args.workers} workers', flush=True)

    ok = skip = done_batches = 0
    processed = 0   # total formulas whose futures completed (for progress display)
    pending: dict = {}   # global_idx → record_or_None
    next_write = 0

    with open(args.output, 'w') as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_batch, ba): ba
                for ba in batches
            }
            for future in as_completed(futures):
                batch_results = future.result()
                done_batches += 1
                processed += len(batch_results)
                for global_idx, record in batch_results:
                    pending[global_idx] = record

                # Write completed records in order
                while next_write in pending:
                    rec = pending.pop(next_write)
                    if rec is not None:
                        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        ok += 1
                    else:
                        skip += 1
                    next_write += 1

                if done_batches % 2 == 0 or done_batches == len(batches):
                    print(f'  [{processed}/{total}] written={ok} skip={skip}',
                          flush=True)

    print(f'\n[done] ok={ok}  skip={skip}  -> {args.output}')


def debug_formula(formula: str):
    """Run lualatex on a formula with debug output (DH/DV/G/R lines) and print results."""
    workdir = Path(tempfile.mkdtemp())
    try:
        parts = [_TEX_HEADER]
        block = _TEX_FORMULA_BLOCK_DEBUG.replace('FML_ID', '0').replace('FML_FORMULA', formula)
        parts.append(block)
        parts.append(r"\usebox{\fmlbox}" + "\n")
        parts.append(_TEX_FOOTER)
        tex = ''.join(parts)

        tex_file = workdir / 'fml.tex'
        tex_file.write_text(tex, encoding='utf-8')
        result = subprocess.run(
            ['lualatex', '-interaction=nonstopmode', str(tex_file)],
            cwd=workdir, capture_output=True, text=True, timeout=120,
        )
        lines = (result.stdout + result.stderr).splitlines()
        node_lines = []
        for line in lines:
            if line.startswith('NODE:0:'):
                node_lines.append(line[7:])  # strip 'NODE:0:'

        print(f'=== DEBUG: {formula!r} ===')
        print(f'Raw node lines ({len(node_lines)}):')
        for l in node_lines:
            print(' ', l)

        # Also show processed bboxes
        print('\nProcessed bboxes:')
        nodes = _parse_nodes([l for l in node_lines if not l.startswith('D')])
        bboxes = _nodes_to_bboxes(nodes)
        for b in bboxes:
            print(f"  {b['token']:15s}  xMin={b['xMin']:8.1f}  yMin={b['yMin']:8.1f}"
                  f"  xMax={b['xMax']:8.1f}  yMax={b['yMax']:8.1f}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) >= 3 and _sys.argv[1] == '--debug-formula':
        formula = _sys.argv[2]
        debug_formula(formula)
    else:
        main()
