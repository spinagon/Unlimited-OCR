#!/usr/bin/env python3
"""Unlimited-OCR 2 Client: Send images or PDF documents to an SGLang server and convert to Markdown.

Usage examples:
  # OCR a PDF and save to output.md
  ./ocr_client.py --pdf document.pdf --output document.md

  # OCR page images
  ./ocr_client.py --images page_1.png page_2.png --output result.md

  # Use a remote SGLang server
  ./ocr_client.py --server-url http://192.168.1.100:10000 --pdf paper.pdf
"""

import argparse
import base64
import json
import os
import sys
import tempfile

import pymupdf as fitz
import requests
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

# Add current directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.process_unlimitedOCR2 import convert


def pdf_to_images(pdf_path, dpi=300):
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


def encode_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(image_paths):
    prompt = "<PAGE><image>" * len(image_paths) + "Multi page parsing."
    return [encode_image(path) for path in image_paths] + [{"type": "text", "text": prompt}]


def generate(server_url, image_paths, crop_mode=True, tokens_per_page=8000, stream=True):
    ngram_processor = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    window_size = 1024 if len(image_paths) > 1 else 128
    payload = {
        "model": "Unlimited-OCR-2",
        "messages": [{"role": "user", "content": build_content(image_paths)}],
        "max_tokens": tokens_per_page * len(image_paths),
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"crop_mode": crop_mode},
        "custom_logit_processor": ngram_processor,
        "custom_params": {
            "ngram_size": 35,
            "window_size": window_size,
            "page_token_id": 128827,
            "eos_token_id": 1,
            "max_page_tokens": 20000,
        },
        "stream": stream,
    }

    session = requests.Session()
    session.trust_env = False

    response = session.post(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=6000,
        stream=stream,
    )
    response.raise_for_status()

    chunks = []
    if stream:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                chunks.append(delta)
    else:
        res_json = response.json()
        chunks.append(res_json["choices"][0]["message"]["content"])

    return "".join(chunks)


def main():
    parser = argparse.ArgumentParser(description="Unlimited-OCR 2 Client")
    parser.add_argument("--server-url", default="http://127.0.0.1:10000", help="SGLang server base URL")
    parser.add_argument("--pdf", help="Path to input PDF document")
    parser.add_argument("--images", nargs="+", help="Path(s) to input image(s) in page order")
    parser.add_argument("--output", "-o", help="File to write result to (default: stdout)")
    parser.add_argument("--raw", action="store_true", help="Output raw layout tokens instead of Markdown")
    parser.add_argument("--crop-mode", action="store_true", default=True, help="Use Gundam mode (crop tiles) for detailed recognition")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering PDF pages to image (default 300)")

    args = parser.parse_args()

    if not args.pdf and not args.images:
        parser.error("Must supply either --pdf or --images")

    if args.pdf:
        print(f"[INFO] Converting PDF '{args.pdf}' to page images...", file=sys.stderr)
        image_paths = pdf_to_images(args.pdf, dpi=args.dpi)
        print(f"[INFO] Converted {len(image_paths)} pages.", file=sys.stderr)
    else:
        image_paths = args.images

    print(f"[INFO] Sending {len(image_paths)} page(s) to server at {args.server_url}...", file=sys.stderr)
    try:
        raw_output = generate(args.server_url, image_paths, crop_mode=args.crop_mode)
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to SGLang server at {args.server_url}.", file=sys.stderr)
        print("Please start the SGLang server using ./start_sglang_server.sh or provide a valid --server-url.", file=sys.stderr)
        sys.exit(1)

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
