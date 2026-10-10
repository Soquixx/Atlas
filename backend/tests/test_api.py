from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["student_count"] == 15000


def test_overview_has_expected_fields():
    payload = client.get("/api/overview").json()
    assert payload["student_count"] == 15000
    assert 0 <= payload["average_success_score"] <= 100
    assert len(payload["branches"]) >= 5
    assert len(payload["weekly_trend"]) >= 4
    assert "synthetic" in payload["provenance"].lower()


def test_students_and_detail():
    page = client.get("/api/students", params={"search": "ST00001", "page_size": 5}).json()
    assert page["total"] == 1
    sid = page["items"][0]["student_id"]
    detail = client.get(f"/api/students/{sid}")
    assert detail.status_code == 200
    assert detail.json()["student_id"] == sid
    assert 0 <= detail.json()["success_score"] <= 100
    assert len(detail.json()["course_records"]) > 0
    if detail.json()["feedback_records"]:
        assert detail.json()["feedback_records"][0]["satisfaction"] is not None


def test_segments_and_quality():
    assert len(client.get("/api/segments").json()["items"]) > 0
    quality = client.get("/api/data-quality").json()
    assert quality["student_count"] == 15000
    assert all(item["orphan_student_rows"] == 0 for item in quality["sources"])


def test_report():
    response = client.post("/api/reports/generate", json={"report_type": "risk_snapshot", "branch": "CSE"})
    assert response.status_code == 200
    body = response.json()
    assert body["scope"]["students"] > 0
    assert len(body["findings"]) >= 3
