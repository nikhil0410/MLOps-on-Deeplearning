#!/usr/bin/env python3
"""Post-deployment model performance tracking (M5).

Generates a small batch of *labeled* synthetic requests, sends them to the
deployed ``/predict`` endpoint, compares predictions against the known true
labels, and writes a performance report (accuracy + per-class counts).

Usage:
    python scripts/simulate_requests.py --url http://localhost:8000 --n 40
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Reuse the exact generators used to create the training data so the labels
# are trustworthy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.generate_synthetic import _make_cat_image, _make_dog_image  # noqa: E402


def _post_image(url: str, image_bytes: bytes, timeout: int = 15) -> dict:
    boundary = f"----sim{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="req.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _image_bytes(kind: str, rng: random.Random, size: int = 224) -> bytes:
    img = _make_cat_image(size, rng) if kind == "cats" else _make_dog_image(size, rng)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=40, help="total requests to send")
    parser.add_argument("--out", default="monitoring/performance_report.json")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    rng = random.Random(args.seed)
    classes = ["cats", "dogs"]

    correct = 0
    total = 0
    latencies = []
    confusion = Counter()  # (true, pred) -> count

    for i in range(args.n):
        true_label = classes[i % 2]
        payload = _image_bytes(true_label, rng)
        try:
            result = _post_image(f"{base}/predict", payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] request {i} failed: {exc}")
            continue

        pred = result["label"]
        latencies.append(result.get("inference_time_ms", 0.0))
        confusion[(true_label, pred)] += 1
        total += 1
        correct += int(pred == true_label)

    if total == 0:
        print("[FAIL] no successful requests")
        return 1

    accuracy = correct / total
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": base,
        "num_requests": total,
        "accuracy": round(accuracy, 4),
        "avg_inference_ms": round(avg_latency, 2),
        "confusion": {f"true={t},pred={p}": c for (t, p), c in confusion.items()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nSaved report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
