#!/usr/bin/env python3
"""Run Unlimited-OCR-2 directly using Hugging Face Transformers.

Usage:
  # OCR single image:
  ./ocr_transformers.py --images page_1.png --output result.md

  # OCR PDF document:
  ./ocr_transformers.py --pdf document.pdf --output result.md

  # Specify device (cuda / cpu):
  ./ocr_transformers.py --pdf document.pdf --device cpu
"""

import argparse
import os
import sys
import tempfile
import torch
from transformers import AutoModel, AutoTokenizer
import pymupdf as fitz

# Add current directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.process_unlimitedOCR2 import convert


def pdf_to_images(pdf_path, dpi=200):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths = []
    for i, page in enumerate(doc):
        image_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(image_path)
        image_paths.append(image_path)
    doc.close()
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Unlimited-OCR 2 Direct Transformers Inference")
    parser.add_argument("--model-name", default="baidu/Unlimited-OCR-2", help="Hugging Face model ID or path")
    parser.add_argument("--pdf", help="Path to input PDF document")
    parser.add_argument("--images", nargs="+", help="Path(s) to input image(s) in page order")
    parser.add_argument("--output", "-o", help="File to write result to (default: stdout)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run on")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="Model torch dtype")
    parser.add_argument("--crop-mode", choices=["auto", "gundam", "base"], default="auto", help="Visual mode: gundam (tiles) or base (single global view)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max tokens to generate (default: 2048)")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream generated tokens in real time")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable token streaming")
    parser.add_argument("--raw", action="store_true", help="Output raw layout tokens instead of Markdown")
    parser.add_argument("--save-raw", help="File path to also save raw layout tokens")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering (default: 200)")
    args = parser.parse_args()

    if not args.pdf and not args.images:
        parser.error("Must supply either --pdf or --images")

    if args.pdf:
        print(f"[INFO] Converting PDF '{args.pdf}' to page images...", file=sys.stderr)
        pages = pdf_to_images(args.pdf, dpi=args.dpi)
        print(f"[INFO] Converted {len(pages)} pages.", file=sys.stderr)
    else:
        pages = args.images

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"[INFO] Loading model '{args.model_name}' on {device} ({dtype})...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    ).eval().to(device)

    crop_mode = None if args.crop_mode == "auto" else (args.crop_mode == "gundam")
    print(f"[INFO] Running inference on {len(pages)} page(s) (crop_mode={args.crop_mode})...", file=sys.stderr)
    if len(pages) == 1:
        raw_output = model.infer(
            tokenizer,
            image_file=pages[0],
            crop_mode=crop_mode,
            max_new_tokens=args.max_new_tokens,
            eval_mode=(not args.stream),
            save_results=False,
        )
    else:
        raw_output, num_tokens = model.infer_multi(
            tokenizer,
            image_files=pages,
            crop_mode=crop_mode,
            max_new_tokens=args.max_new_tokens,
            eval_mode=(not args.stream),
            save_results=False,
        )
        print(f"[INFO] Generated {num_tokens} tokens.", file=sys.stderr)

    if args.save_raw:
        with open(args.save_raw, "w", encoding="utf-8") as f:
            f.write(raw_output)
        print(f"[INFO] Raw layout tokens saved to {args.save_raw}", file=sys.stderr)

    if args.raw:
        result = raw_output
    else:
        result = convert(raw_output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[SUCCESS] Result written to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
