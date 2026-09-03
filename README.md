# air_calculator_py

空中手写计算器的 Python 端：仅轨迹（stroke-only）模型的训练、端侧导出、合成数据
生成与评测。

本项目分为五个仓库，需要**并排 checkout**——Rust 侧是 path 依赖。

| 仓库 | 职责 |
|---|---|
| [air_calculator](https://github.com/wilinz/air_calculator) | Flutter 客户端：UI、相机接入、手势交互与三端集成 |
| [air_calculator-rs](https://github.com/wilinz/air_calculator-rs) | Rust 核心：识别解码循环、LaTeX 求值与 C ABI |
| [hand-track](https://github.com/wilinz/hand-track) | 手部检测：palm + landmark 两段式流水线与两端权重 |
| [edge-infer](https://github.com/wilinz/edge-infer) | 推理抽象：`Engine` trait + LiteRT / Core ML 后端 |
| **air_calculator_py** ← 本仓 | 模型训练、合成数据生成与端侧导出 |

## 目录结构

```
air_calculator_py/
├── train/        模型定义 + 训练入口 + 数据集 + 特征/图像/笔画编码器
├── export/       端侧部署导出（ExecuTorch fp16/fp32, TFLite fp32）
├── eval/         checkpoint 评测 + 端侧 benchmark 数据处理
├── synth/        合成数据生成（LaTeX → bbox → InkML）+ 字库 / 标签提取
├── scripts/      杂项工具（数据清洗等）
└── vis/          Streamlit 错误可视化
```

## train/ —— 训练与模型

| 文件 | 内容 |
|---|---|
| `train.py` | 训练入口（`main()` + `_save_latest()`，从头训练 decoder-only 模型）|
| `models.py` | `StrokeEncoder` / `VisualPrefixEncoder` / `CausalTransformerDecoder` |
| `dataset.py` | `MathWritingDataset`、`Vocabulary`、`collate_fn`（支持 parquet/InkML 自动检测）|
| `evaluation.py` | `evaluate()` 验证集贪心/beam EM + CER |
| `stroke_features.py` | 笔画 13 维时序特征（位置/速度/曲率/抬笔标志/全局比例）|
| `stroke_renderer.py` | 笔画轨迹 → 64×256 灰度图渲染 |
| `image_encoder.py` | `ImageEncoder`（DeiT-Small ViT，patch embed → 64 tokens）|
| `build_parquet.py` | InkML → parquet 数据集打包脚本 |

跑训练（默认 d=512，n_layers=8，nhead=8，~35M 参数）：

```bash
cd train
MATHWRITING_DIR=../../dataset/mathwriting-2024 \
python3 train.py --out ./decoder_only --epochs 30 --batch 64
```

## export/ —— 端侧部署导出

**三个最终脚本**，分别对应两种 runtime × 两种精度配置：

| 文件 | 用途 | 部署产物 |
|---|---|---|
| `export_et_hybrid_fp16.py` | ExecuTorch + 平台分支 + fp16（iOS 部署版）| 三件套 `.pte` ~138 MB |
| `export_tf_android_fp32.py` | TFLite + Android + fp32（Android 部署版）| 两件套 `.tflite` ~187 MB |
| `export_et_baseline_nokv_fp32.py` | ExecuTorch + 跨平台 + fp32（论文 §6 baseline，无 KV-cache）| 两件套 `.pte` |

**iOS 部署**：
```bash
cd export
python3 export_et_hybrid_fp16.py --platform ios \
  --ckpt ../../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
  --data-dir ../../dataset/mathwriting-2024 \
  --out ../../weights/exports/executorch_fp16_ios
```

**Android 部署**：
```bash
cd export
python3 export_tf_android_fp32.py \
  --ckpt ../../weights/v3_final/v3_best_epoch104_acc0.7720.pt \
  --data-dir ../../dataset/mathwriting-2024 \
  --out ../../weights/exports/tflite_android
```

历史 backend / 精度组合脚本（Core ML fp16 单端、KV-cache fp16 实验、static int8 等）在 `deserted2/dead_export/` 与 `deserted2/phase5_v3_legacy/`，详见 `docs_final/mobile/` 各实验记录。

## eval/ —— checkpoint 评测

| 文件 | 用途 |
|---|---|
| `eval_errors.py` | edit-distance 错误分析（subs/dels/ins 统计 + 错误类别归类）|
| `mobile/eval_benchmark.py` | 离线评测端侧 Flutter benchmark 导出的 `results.jsonl` |
| `mobile/sample_benchmark.py` | 从 validation 集分层采样 500 条作端侧 benchmark 输入 |

## synth/ —— 合成数据 pipeline

`latex_to_inkml.py` 是一体化入口，三步可拆开：

1. `generate_latex*.py` —— LLM 批量生成 LaTeX 公式（Claude / OpenAI / Qwen 三个 backend）
2. `latex_to_bboxes.py` —— LaTeX → MathWriting 官方 token bbox 坐标
3. `synth_from_bboxes.py` —— bbox + `assets/symbol_library.pkl` 字库 → 真实手写笔画 InkML

`extract_labels.py` —— 从已有 InkML 提取 normalizedLabel 作为 LLM 生成的 seed。

## scripts/ —— 杂项工具

| 文件 | 用途 |
|---|---|
| `clean_human_data.py` | inkml 人工标注数据清洗（模板匹配 + 异常 trace 替换）|

## vis/ —— 可视化

`vis_errors.py` —— Streamlit App，按错误类别浏览 ckpt 预测样本。

```bash
cd vis
MATHWRITING_DIR=../../dataset/mathwriting-2024 \
streamlit run vis_errors.py --server.port 8501
```

## 数据集 / 权重位置

| 路径 | 内容 |
|---|---|
| `../dataset/mathwriting-2024/{train,valid,test,synthetic}.parquet` | 主数据集（parquet 格式）|
| `../dataset/mathwriting-2024/vocab.json` / `bpe_vocab.json` | char vocab（230） / BPE vocab（1000）|
| `../weights/<run>/*.pt` | 训练 checkpoint |
| `../weights/exports/` | 各后端导出产物归档 |

数据与权重不在仓库里，路径按上表相对本仓摆放。

## 产物去向

端侧导出的产物由 [air_calculator](https://github.com/wilinz/air_calculator) 的 `tool/copy_platform_models.sh` 就位，
`export_coreml_ios.py` 的 `--max-decode` 要与 Rust 侧的 `MAX_DECODE` 一致。

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`.

The code is free to use, modify, redistribute and commercialize, including
publishing derivative applications on the App Store, Google Play or anywhere
else. Per section 6 of the Apache License 2.0, no trademark or product name
rights are granted: **Air Calculator**, **AirCalculator**, `air_calculator` as
a product name, and the application icons and logos in this repository are
reserved, and may not be used to publish or promote a derivative work without
prior written permission. Fork it, but ship it under your own name. Factual
references such as "based on Air Calculator" are fine, as long as they do not
suggest endorsement.
