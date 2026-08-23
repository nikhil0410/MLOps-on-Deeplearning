"""Endpoint tests for the FastAPI service (M2/M3).

Uses Starlette's TestClient. The model is injected directly into the app state
so these tests don't depend on a trained artifact being present.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import main
from src.models.model import build_model


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        # Ensure a model is available regardless of whether model.pt exists.
        main.state.model = build_model(num_classes=2).eval()
        main.state.class_names = ["cats", "dogs"]
        main.state.image_size = 224
        yield c


def _image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (120, 60, 30)).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/predict" in resp.json()["endpoints"]


def test_predict_returns_label_and_probs(client):
    resp = client.post(
        "/predict",
        files={"file": ("cat.jpg", _image_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ["cats", "dogs"]
    assert set(body["probabilities"].keys()) == {"cats", "dogs"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_non_image(client):
    resp = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
