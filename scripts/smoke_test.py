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
        # 8x8 solid-colour JPEG fallback (valid, decodable without Pillow).
        import base64

        b64 = (
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEA"
            "AAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIh"
            "MUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
            "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZ"
            "mqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx"
            "8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
            "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAV"
            "YnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hp"
            "anN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPE"
            "xcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCp"
            "RRRXzB9of//Z"
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

    print("\nSmoke test PASSED ?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
