# Baidu Unlimited-OCR-2 Setup Guide

This directory (`~/prog/uocr2`) contains the repository, virtual environment, and inference utilities for **Baidu Unlimited-OCR-2** (Sparse Reference Sliding Window Attention / SR-SWA long-document OCR model).

---

## 1. Environment & Setup

A Python 3.10 virtual environment has been configured at `.venv` using `uv`:
- **Python**: 3.10.22
- **Core Packages**: `torch`, `torchvision`, `transformers`, `tokenizers`, `huggingface-hub`, `accelerate`, `pymupdf`, `fastokens`, `einops`, `addict`, `easydict`, `pybase64`, `triton`, `starlette`, `fastapi`, `pydantic`, `ipython`, `openai`, `sglang` (custom Baidu build).

To activate the environment:
```bash
cd ~/prog/uocr2
source .venv/bin/activate
```

---

## 2. Model Weights

The model weights (~6.7 GB across 2 safetensor shards) are hosted on Hugging Face at [`baidu/Unlimited-OCR-2`](https://huggingface.co/baidu/Unlimited-OCR-2).

To pre-download / cache the model weights:
```bash
./download_model.py
```
*(The weights will be cached in `~/.cache/huggingface/hub/`)*.

---

## 3. Usage Modes

### Mode A: SGLang Client (`./ocr_client.py`)
Connects to an SGLang server running on localhost or a remote GPU server. Converts images or PDFs into formatted Markdown in a single pass.

```bash
# OCR a PDF document:
./ocr_client.py --pdf my_document.pdf --output my_document.md

# OCR specific page images:
./ocr_client.py --images page_1.png page_2.png --output result.md

# Connect to a remote SGLang server:
./ocr_client.py --server-url http://<gpu-server-ip>:10000 --pdf doc.pdf -o doc.md

# Output raw layout blocks instead of post-processed Markdown:
./ocr_client.py --pdf doc.pdf --raw
```

### Mode B: Direct Transformers Inference (`./ocr_transformers.py`)
Runs inference locally without a separate SGLang server using Hugging Face Transformers (`AutoModel` / `AutoTokenizer`).

```bash
# Run on GPU (default if available) or CPU:
./ocr_transformers.py --pdf sample.pdf --output result.md

# On CPU / lower-power systems, use --crop-mode base for faster single-view inference:
./ocr_transformers.py --images test_docs/simple_test.png --crop-mode base --output result.md

# Save both raw tokens and post-processed markdown:
./ocr_transformers.py --images test_docs/simple_test.png --crop-mode base --save-raw raw.txt -o result.md
```

### Mode C: Benchmark Evaluation (`infer.py` + `utils/process_unlimitedOCR2.py`)
Runs batch evaluations on **MPDocBench** or **MarathonDocBench** against an SGLang server:

```bash
# Step 1: Query server and save raw layout outputs
python infer.py --url 127.0.0.1:10000 --benchmark MPDocBench \
    --dataset-root /path/to/MPDocBench --all --raw-dir raw --concurrency 4

# Step 2: Post-process raw outputs into clean evaluation Markdown
python utils/process_unlimitedOCR2.py --infer-dir raw --out-dir pred
```

---

## 5. Test Documents & Verification

Sample documents and verified test results are stored in `test_docs/`:
- `test_docs/simple_test.pdf` and `test_docs/simple_test.png`: Verification document with headings, paragraphs, and a markdown table.
- `test_docs/simple_raw.txt`: Verified raw layout blocks from Unlimited-OCR-2.
- `test_docs/simple_result.md`: Verified converted Markdown output.

---

## 6. Serving (GPU Environments)

Unlimited-OCR-2 is a ~3.34B vision-language MoE model requiring an NVIDIA CUDA GPU (e.g. A800, RTX 3090/4090, A100) with BF16 support for SGLang serving.

To start the SGLang server:
```bash
./start_sglang_server.sh [PORT] [HOST]
# Default is http://0.0.0.0:10000
```

