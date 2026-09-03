# air_calculator_py — 项目详情文档

毕业设计「空中手势交互型智能语音计算器」的 Python 端工程。负责数据合成、模型训练、checkpoint 评测、端侧部署导出与端侧 benchmark 数据处理。Flutter App 见同级 `air_calculator/`。

---

## 1. 功能实现

### 1.1 在线手写数学公式识别模型
- decoder-only prefix-LM 架构（48.9 M 参数，可训练 ~34.5 M）：
  - **图像分支** `VisualPrefixEncoder`：DeiT-Small（冻结前 8 个 block），patch embed → 64 个 token，线性投影到 d=512；
  - **笔画分支** `StrokeEncoder`：轻量 Transformer，13 维笔画时序特征（位置 / 速度 / 曲率 / 抬笔标志 / 全局比例）；
  - **解码器** `CausalTransformerDecoder`：8 层因果 self-attention，nhead=8，d=512，词表 230 token；
  - 两路 prefix 拼接后送入解码器，由统一因果 self-attention 完成跨模态对齐 + LaTeX 生成（无 cross-attention）。
- 训练策略：合成数据预热 + 人工数据续训对齐两阶段；错误归类驱动靶向数据补充。
- MathWriting 全量验证集：EM 77.09% / char-CER 3.84%。

### 1.2 数据合成 pipeline
LLM 生成 LaTeX → MathWriting 官方 token bbox → 真实手写笔画字库 InkML，三步可拆可合：
1. `synth/generate_latex*.py` —— LLM（Claude / OpenAI / Qwen）批量生成 LaTeX，可用 `extract_labels.py` 提供 seed；
2. `synth/latex_to_bboxes.py` —— LaTeX 渲染为 token 级 bbox 坐标；
3. `synth/synth_from_bboxes.py` —— bbox + `assets/symbol_library.pkl`（从人工 InkML 抽取的真实手写笔画字库）→ 合成 InkML 轨迹。
- 一体化入口：`synth/latex_to_inkml.py`。

### 1.3 端侧部署导出
三个核心导出脚本，对应两种 runtime × 两种精度配置：

| 脚本 | runtime | 精度 | 用途 | 产物 |
|---|---|---|---|---|
| `export/export_et_hybrid_fp16.py` | ExecuTorch | fp16 | iOS 部署 | 三件套 `.pte` ~138 MB |
| `export/export_tf_android_fp32.py` | LiteRT (TFLite) | fp32 | Android 部署 | 两件套 `.tflite` ~187 MB |
| `export/export_et_baseline_nokv_fp32.py` | ExecuTorch | fp32 | 论文 baseline（无 KV cache） | 两件套 `.pte` |

导出模型严格按 `prefix_enc / decoder_prefill / decoder_step` 三段切分，与 Dart 端 `services/mwh_*_engine.dart` 一一对应。

### 1.4 评测
- `eval/eval_errors.py` —— 基于 edit-distance 的错误分析（substitution / deletion / insertion 统计，分类标签噪音、连笔、形近混淆等错误来源）。
- `eval/mobile/sample_benchmark.py` —— 从 validation 集分层采样 500 条作为端侧 benchmark 输入。
- `eval/mobile/eval_benchmark.py` —— 离线评测 Flutter benchmark 导出的 `results.jsonl`，计算 strict / normalized ExpRate 与 char-level CER。

### 1.5 错误可视化
- `vis/vis_errors.py` —— Streamlit App，按错误类别浏览 checkpoint 预测样本，辅助定位标签噪音与系统性错误。

---

## 2. 代码组织架构

```
air_calculator_py/
├── train/                                 # 模型定义 + 训练入口 + 数据集 + 编码器
│   ├── train.py                           # 训练主程序（main + _save_latest）
│   ├── models.py                          # StrokeEncoder / VisualPrefixEncoder / CausalTransformerDecoder
│   ├── dataset.py                         # MathWritingDataset / Vocabulary / collate_fn
│   │                                       # （自动识别 parquet 或 InkML 目录）
│   ├── evaluation.py                      # 验证集贪心 / beam EM + CER
│   ├── stroke_features.py                 # 笔画 13 维时序特征
│   │                                       # ★ 与 Dart sequence_feature_extractor.dart 严格对齐
│   ├── stroke_renderer.py                 # 笔画 → 64×256 灰度图渲染
│   │                                       # ★ 与 Dart stroke_rasterizer.dart 严格对齐
│   ├── image_encoder.py                   # DeiT-Small 封装 + patch embedding 取 64 token
│   └── build_parquet.py                   # InkML → parquet 数据打包
│
├── export/                                # 端侧部署导出
│   ├── export_et_hybrid_fp16.py           # iOS 部署版（ExecuTorch + CoreML + fp16）
│   ├── export_tf_android_fp32.py          # Android 部署版（LiteRT/TFLite + fp32）
│   └── export_et_baseline_nokv_fp32.py    # baseline（无 KV cache，跨平台 fp32）
│
├── eval/                                  # checkpoint 评测 + 端侧 benchmark 数据处理
│   ├── eval_errors.py                     # 错误分类分析
│   └── mobile/
│       ├── sample_benchmark.py            # 分层采样 500 条作端侧输入
│       └── eval_benchmark.py              # 评测 Flutter 导出 results.jsonl
│
├── synth/                                 # 合成数据 pipeline
│   ├── generate_latex.py                  # LLM 生成 LaTeX（Claude）
│   ├── generate_latex_openai.py           # LLM 生成 LaTeX（OpenAI）
│   ├── generate_latex_qwen.py             # LLM 生成 LaTeX（Qwen）
│   ├── extract_labels.py                  # 从 InkML 抽取 normalizedLabel 作为 LLM seed
│   ├── latex_to_bboxes.py                 # LaTeX → 官方 token bbox
│   ├── synth_from_bboxes.py               # bbox + 字库 → InkML 真实笔画
│   └── latex_to_inkml.py                  # 一体化入口（串联三步）
│
├── scripts/                               # 杂项工具
│   └── clean_human_data.py                # InkML 人工数据清洗：模板匹配 + 异常 trace 替换
│
├── vis/                                   # 可视化
│   └── vis_errors.py                      # Streamlit 错误浏览 App
│
└── README.md / this PROJECT.md
```

### 2.1 训练流水线

```
   InkML (MathWriting 2024)
       │
       │  build_parquet.py        # 一次性打包
       ▼
   parquet 数据集
       │
       │  MathWritingDataset + collate_fn
       ▼
   ┌──────────────────────────┐
   │  StrokeEncoder           │ ← 13 维笔画特征 (stroke_features.py)
   │  VisualPrefixEncoder     │ ← 64×256 灰度图 (stroke_renderer.py + image_encoder.py)
   └──────────────────────────┘
       │ 双流 prefix 拼接
       ▼
   CausalTransformerDecoder  ──→  LaTeX token 序列
       │
       │  evaluation.evaluate()
       ▼
   EM / CER 指标
```

入口（默认 d=512，n_layers=8，nhead=8，~35M 参数）：
```bash
cd train
MATHWRITING_DIR=../../dataset/mathwriting-2024 \
python3 train.py --out ./decoder_only --epochs 30 --batch 64
```

### 2.2 部署流水线

```
   PyTorch checkpoint (.pt)
       │
       │  export/*.py                          （平台分支 + 精度选择）
       ▼
   prefix_enc + decoder_prefill + decoder_step
       │
       │  fp16 / fp32 编译
       ▼
   ┌────────────────────────────────────┐
   │  .pte (iOS, ExecuTorch CoreML)     │
   │  .tflite (Android, LiteRT)         │
   └────────────────────────────────────┘
       │
       │  拷贝至 air_calculator/assets/models/
       ▼
   Flutter App 端侧加载
```

iOS 部署：
```bash
python3 export/export_et_hybrid_fp16.py --platform ios \
  --ckpt ../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
  --data-dir ../dataset/mathwriting-2024 \
  --out ../weights/exports/executorch_fp16_ios
```

Android 部署：
```bash
python3 export/export_tf_android_fp32.py \
  --ckpt ../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
  --data-dir ../dataset/mathwriting-2024 \
  --out ../weights/exports/tflite_android
```

### 2.3 数据合成流水线

```
   现有 InkML normalizedLabel
       │
       │  extract_labels.py
       ▼
   LaTeX seed
       │
       │  generate_latex*.py (Claude / OpenAI / Qwen)
       ▼
   合成 LaTeX 公式
       │
       │  latex_to_bboxes.py
       ▼
   token bbox 序列
       │
       │  synth_from_bboxes.py + assets/symbol_library.pkl
       ▼
   合成 InkML（真实手写笔画）
       │
       │  build_parquet.py
       ▼
   合成 parquet 数据集（与人工数据混合训练）
```

### 2.4 与 Flutter 端的契约

| Python | Dart | 契约 |
|---|---|---|
| `train/stroke_features.py` | `services/sequence_feature_extractor.dart` | 13 维笔画时序特征 |
| `train/stroke_renderer.py` | `services/stroke_rasterizer.dart` | 64×256 灰度图渲染 |
| `train/dataset.Vocabulary` | `assets/models/vocab.json` | 230 token 字符级词表 |
| `export/export_et_hybrid_fp16.py` | iOS `.pte` 三件套 + `services/mwh_executorch_engine.dart` | ExecuTorch fp16 |
| `export/export_tf_android_fp32.py` | Android `.tflite` 两件套 + `services/mwh_litert_engine.dart` | LiteRT fp32 |
| `eval/mobile/sample_benchmark.py` | `assets/benchmark_samples.jsonl` | 500 条评测样本 |
| `eval/mobile/eval_benchmark.py` | `pages/benchmark_page.dart` 内置离线评测 | ExpRate / CER 计算逻辑一致 |

修改上述任意一方时必须同步另一方，否则会出现沉默的推理错误（如词表 off-by-one、笔画归一化常数不一致、图像通道顺序错位等）。

### 2.5 关键依赖

| 包 | 作用 |
|---|---|
| `torch` / `torchvision` / `timm` | 训练框架与 DeiT 预训练权重 |
| `executorch` | iOS 部署导出（CoreML backend） |
| `tensorflow` / `ai-edge-torch` | Android 部署导出（LiteRT/TFLite） |
| `pyarrow` / `pandas` | parquet 数据集 IO |
| `anthropic` / `openai` / `dashscope` | LLM 合成数据生成 |
| `streamlit` | 错误可视化 App |
| `python-Levenshtein` | edit-distance 错误分析 |
