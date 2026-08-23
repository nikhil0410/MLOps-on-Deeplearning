#!/usr/bin/env python3
"""Post-deploy smoke test (M4).

Calls the health endpoint and one prediction endpoint against a running
service. Exits non-zero on any failure so a CI/CD pipeline can gate on it.

Usage:
    python scripts/smoke_test.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import uuid


def _make_test_image() -> bytes:
    """Create a small in-memory JPEG. Uses Pillow if available, else a tiny
    embedded fallback JPEG so the smoke test has zero hard dependencies."""
    try:
        from PIL import Image  # type: ignore

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 90, 40)).save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        # 1x1 red JPEG (base64-free, minimal valid JPEG).
        import base64

        b64 = (
            "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////////////////////"
            "////////////////////////////////////////////////wgALCAABAAEBAREA"
            "/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        )
        return base64.b64decode(b64)


def _http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def _http_post_image(url: str, image: bytes, timeout: int = 15) -> tuple[int, str]:
    boundary = f"----smoke{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    # 1) Health check ------------------------------------------------------
    try:
        status, body = _http_get(f"{base}/health")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] health request errored: {exc}")
        return 1
    if status != 200:
        print(f"[FAIL] health returned HTTP {status}: {body}")
        return 1
    print(f"[PASS] health -> {body}")

    # 2) Prediction check --------------------------------------------------
    try:
        status, body = _http_post_image(f"{base}/predict", _make_test_image())
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] predict request errored: {exc}")
        return 1
    if status != 200:
        print(f"[FAIL] predict returned HTTP {status}: {body}")
        return 1
    if '"label"' not in body:
        print(f"[FAIL] predict response missing 'label': {body}")
        return 1
    print(f"[PASS] predict -> {body}")

    print("\nSmoke test PASSED ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
