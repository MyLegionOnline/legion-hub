from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app import metrics
from app.main import app


def test_metrics_count_requests_by_route_template_not_raw_url():
    metrics.reset()
    with TestClient(app) as client:
        client.get("/health")
        client.get("/v1/tools", params={"installed_item_id": str(uuid4()), "key": "lgn_secret"})
        client.get("/no/such/path?token=abc")
        client.get("/metrics")
        body = client.get("/metrics").text

    assert 'legion_gateway_requests_total{path="/health",status="200"} 1' in body
    assert 'legion_gateway_requests_total{path="/v1/tools",status="401"} 1' in body
    assert 'legion_gateway_requests_total{path="unmatched",status="404"} 1' in body
    assert 'legion_gateway_request_duration_seconds_count{path="/health"} 1' in body
    assert "legion_gateway_start_time_seconds " in body
    # Scrapes are not counted, and no query string or raw URL leaks into labels.
    assert 'path="/metrics"' not in body
    assert "lgn_secret" not in body and "token=abc" not in body and "/no/such/path" not in body
