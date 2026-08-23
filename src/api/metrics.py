"""Prometheus metrics for the inference service (M5).

Exposes counters/histograms that the ``/metrics`` endpoint renders in the
Prometheus text exposition format. Import these objects wherever you need to
record a measurement.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

# Total HTTP requests, partitioned by method / endpoint / status code.
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

# End-to-end request latency (seconds), per endpoint.
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

# Number of predictions made, partitioned by the predicted class.
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of predictions, by predicted class",
    ["predicted_class"],
)

# Pure model inference latency (seconds), excludes HTTP overhead.
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Model inference latency in seconds",
)
