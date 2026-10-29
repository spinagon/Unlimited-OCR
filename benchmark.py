#!/usr/bin/env python3
"""Comprehensive benchmark tool for Unlimited-OCR-2.

Measures:
  - Model load time & memory (RSS)
  - PDF rendering & image transform time
  - Vision encoder & LLM prefill time
  - Autoregressive token generation throughput (tokens/sec)
  - Total latency & memory footprint
Saves structured benchmark results to JSON.
"""

import argparse
import json
import os
import sys
import time
import psutil
import torch
from transformers import AutoModel, AutoTokenizer
import pymupdf as fitz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.process_unlimitedOCR2 import convert


def get_rss_mb():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def pdf_to_images(pdf_path, max_pages=None, dpi=200):
    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    pages_to_process = min(total_doc_pages, max_pages) if max_pages else total_doc_pages
    image_paths = []
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_docs", "bench_pages")
    os.makedirs(tmp_dir, exist_ok=True)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i in range(pages_to_process):
        img_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        doc[i].get_pixmap(matrix=mat).save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths, total_doc_pages


def main():
    parser = argparse.ArgumentParser(description="Unlimited-OCR-2 Benchmark")
    parser.add_argument("--pdf", default="Unlimited-OCR-2.pdf", help="Input PDF path")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process (None = all)")
    parser.add_argument("--crop-mode", choices=["auto", "gundam", "base"], default="base", help="Cropping mode")
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Max tokens budget")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF rendering")
    parser.add_argument("--output-json", default="test_docs/benchmark_result.json", help="Path to save benchmark JSON")
    parser.add_argument("--output-md", default="test_docs/benchmark_output.md", help="Path to save output Markdown")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print("==========================================================")
    print("           UNLIMITED-OCR-2 PERFORMANCE BENCHMARK          ")
    print("==========================================================")
    print(f"Target Document : {pdf_path}")
    print(f"Crop Mode       : {args.crop_mode}")
    print(f"Initial RSS     : {get_rss_mb()} MB")
    print(f"CPU Cores       : {os.cpu_count()} (PyTorch threads: {torch.get_num_threads()})")
    print("==========================================================\n")

    bench_data = {
        "device": "cpu",
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "crop_mode": args.crop_mode,
        "pdf_name": os.path.basename(pdf_path),
    }

    # 1. PDF Conversion
    t0 = time.time()
    image_paths, total_doc_pages = pdf_to_images(pdf_path, max_pages=args.max_pages, dpi=args.dpi)
    t_pdf = time.time() - t0
    bench_data["total_doc_pages"] = total_doc_pages
    bench_data["processed_pages"] = len(image_paths)
    bench_data["pdf_render_time_sec"] = round(t_pdf, 2)
    print(f"[1/4] Rendered {len(image_paths)}/{total_doc_pages} page(s) in {t_pdf:.2f}s (RAM: {get_rss_mb()} MB)")

    # 2. Model Loading
    t0 = time.time()
    model_name = "baidu/Unlimited-OCR-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().to("cpu")
    t_load = time.time() - t0
    bench_data["model_load_time_sec"] = round(t_load, 2)
    bench_data["post_load_rss_mb"] = get_rss_mb()
    print(f"[2/4] Model loaded in {t_load:.2f}s (RAM: {get_rss_mb()} MB)")

    # 3. Inference
    crop_mode = None if args.crop_mode == "auto" else (args.crop_mode == "gundam")
    print(f"[3/4] Starting inference on {len(image_paths)} page(s)...")

    t0_infer = time.time()
    if len(image_paths) == 1:
        res = model.infer(
            tokenizer,
            image_file=image_paths[0],
            crop_mode=crop_mode,
            max_new_tokens=args.max_new_tokens,
            tps_interval=20,
            return_dict=True,
            save_results=False,
        )
        raw_text = res["text"]
        token_count = len(res["token_ids"])
    else:
        raw_text, token_count = model.infer_multi(
            tokenizer,
            image_files=image_paths,
            crop_mode=crop_mode,
            max_new_tokens=args.max_new_tokens,
            tps_interval=20,
            save_results=False,
        )

    t_total_infer = time.time() - t0_infer
    tokens_per_sec = round(token_count / max(t_total_infer, 1e-6), 2)
    bench_data["inference_time_sec"] = round(t_total_infer, 2)
    bench_data["generated_tokens"] = token_count
    bench_data["overall_tokens_per_sec"] = tokens_per_sec
    bench_data["peak_rss_mb"] = get_rss_mb()

    print(f"\n[3/4] Inference finished in {t_total_infer:.2f}s")
    print(f"      Generated Tokens : {token_count}")
    print(f"      Throughput       : {tokens_per_sec} tokens/sec")
    print(f"      Peak RAM         : {get_rss_mb()} MB")

    # 4. Post-processing to Markdown
    t0 = time.time()
    md_result = convert(raw_text)
    t_convert = time.time() - t0
    bench_data["markdown_convert_time_sec"] = round(t_convert, 4)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_result)

    print(f"[4/4] Converted to Markdown in {t_convert:.4f}s")
    print(f"[OK] Benchmark JSON saved to : {args.output_json}")
    print(f"[OK] Output Markdown saved to: {args.output_md}")


if __name__ == "__main__":
    main()
