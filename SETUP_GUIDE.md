# Baidu Unlimited-OCR-2 Setup & Execution Guide

This repository contains the model code, utilities, benchmark tools, and project configuration for **Baidu Unlimited-OCR-2** (Sparse Reference Sliding Window Attention / SR-SWA long-document OCR model).

---

## 1. Environment & Automatic Setup with `uv`

The project uses `pyproject.toml` to declare all required dependencies and target Python version (`>=3.10, <3.11`).

No manual virtual environment creation or activation is necessary. When you invoke any script using **`uv run`**, `uv` automatically resolves all dependencies, installs packages, and runs the script in an isolated environment:

```bash
# Verify uv environment and display help for any tool
uv run python ocr_client.py --help
uv run python ocr_transformers.py --help
uv run python benchmark.py --help
```

---

## 2. Model Weights

The model weights (~6.7 GB across 2 safetensor shards) are hosted on Hugging Face at [`baidu/Unlimited-OCR-2`](https://huggingface.co/baidu/Unlimited-OCR-2).

To pre-download and verify the model snapshot using `uv`:
```bash
uv run python download_model.py
```
*(Weights are cached in the default Hugging Face cache directory).*

---

## 3. Running OCR

### Mode A: SGLang Client (`ocr_client.py`)
Connects to an SGLang server running locally or on a remote GPU server. Converts images or PDFs into formatted Markdown in a single pass:

```bash
# OCR a PDF document to Markdown:
uv run python ocr_client.py --pdf document.pdf --output document.md

# OCR specific page images:
uv run python ocr_client.py --images page_1.png page_2.png --output result.md

# Connect to a remote SGLang server:
uv run python ocr_client.py --server-url http://<server-ip>:10000 --pdf document.pdf -o document.md

# Output raw layout tokens instead of Markdown:
uv run python ocr_client.py --pdf document.pdf --raw
```

### Mode B: Direct Transformers Inference (`ocr_transformers.py`)
Runs inference directly without needing an external server:

```bash
# Standard inference (automatically uses CUDA if available, otherwise CPU):
uv run python ocr_transformers.py --pdf document.pdf --output result.md

# On CPU systems, use base mode (single global view) for faster processing:
uv run python ocr_transformers.py --images test_docs/simple_test.png --crop-mode base --output result.md

# Save both raw layout tokens and post-processed markdown:
uv run python ocr_transformers.py --images test_docs/simple_test.png --crop-mode base --save-raw raw.txt -o result.md
```

### Mode C: Performance Benchmark (`benchmark.py`)
Measures PDF rendering time, model load latency, vision prefill time, decoding throughput (tokens/sec), peak RAM, and markdown conversion:

```bash
# Run benchmark on a sample page:
uv run python benchmark.py --pdf Unlimited-OCR-2.pdf --max-pages 1 --output-json bench.json --output-md bench.md

# Run benchmark on full document:
uv run python benchmark.py --pdf Unlimited-OCR-2.pdf --output-json bench.json --output-md bench.md
```

### Mode D: Evaluation Pipeline (`infer.py` + `utils/process_unlimitedOCR2.py`)
Runs batch evaluations on **MPDocBench** or **MarathonDocBench** datasets against an SGLang server:

```bash
# Step 1: Send documents to server and collect raw layout outputs
uv run python infer.py --url 127.0.0.1:10000 --benchmark MPDocBench \
    --dataset-root /path/to/MPDocBench --all --raw-dir raw --concurrency 4

# Step 2: Post-process raw outputs into evaluation Markdown
uv run python utils/process_unlimitedOCR2.py --infer-dir raw --out-dir pred
```

---

## 4. Test Documents & Verification

Sample documents and verified test results are available in `test_docs/`:
- `test_docs/simple_test.pdf` and `test_docs/simple_test.png`: Verification document with headings, paragraphs, and a formatted table.
- `test_docs/simple_raw.txt`: Verified raw layout blocks from Unlimited-OCR-2.
- `test_docs/simple_result.md`: Verified converted Markdown output.
- `test_docs/bench_page1.json` & `test_docs/bench_page1.md`: Verified benchmark run on Page 1 of the research paper.

---

## 5. Serving (GPU Environments)

Unlimited-OCR-2 is a ~3.34B vision-language MoE model requiring an NVIDIA CUDA GPU with BF16 support for high-throughput SGLang serving.

To start the SGLang server:
```bash
./start_sglang_server.sh [PORT] [HOST]
# Default: http://0.0.0.0:10000
```
