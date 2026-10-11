from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["student_count"] == 20000


def test_overview_has_expected_fields():
    payload = client.get("/api/overview").json()
    assert payload["student_count"] == 20000
    assert 0 <= payload["average_success_score"] <= 100
    assert len(payload["branches"]) >= 5
    assert len(payload["weekly_trend"]) >= 4
    assert "synthetic" in payload["provenance"].lower()


def test_students_and_detail():
    page = client.get("/api/students", params={"search": "ST000001", "page_size": 5}).json()
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
    assert quality["student_count"] == 20000
    assert all(item["orphan_student_rows"] == 0 for item in quality["sources"])


def test_report():
    response = client.post("/api/reports/generate", json={"report_type": "risk_snapshot", "branch": "CSE"})
    assert response.status_code == 200
    body = response.json()
    assert body["scope"]["students"] > 0
    assert len(body["findings"]) >= 3


def test_new_dataset_and_model_are_loaded():
    overview = client.get("/api/overview").json()
    assert overview["student_count"] == 20000
    assert overview["model_metrics"]["available"] is True
    assert overview["model_metrics"]["model"] == "Logistic Regression"
    sources = client.get("/api/sources").json()["items"]
    names = {item["filename"] for item in sources}
    assert "placement_outcomes.csv" in names
    quality = client.get("/api/data-quality").json()
    assert len(quality["sources"]) == 8


def test_student_predictions_and_explanations():
    page = client.get("/api/students", params={"search": "ST000001", "page_size": 1}).json()
    assert page["total"] == 1
    student = page["items"][0]
    assert student["average_course_marks"] is not None
    assert student["placement_risk_probability"] is not None
    assert isinstance(student["explanation_drivers"], list)


def test_uploaded_lms_source_activates_and_recalculates(tmp_path, monkeypatch):
    """A staged LMS CSV becomes active only after sync and changes the live analytics."""
    import shutil
    import pandas as pd
    import app.main as main_module
    from app.services.analytics import DatasetStore

    active_dir = tmp_path / "active_data"
    staging_dir = tmp_path / "staging"
    active_dir.mkdir()
    staging_dir.mkdir()
    for path in main_module.DATA_DIR.glob("*.csv"):
        shutil.copy2(path, active_dir / path.name)

    monkeypatch.setattr(main_module, "DATA_DIR", active_dir)
    monkeypatch.setattr(main_module, "STAGING_DIR", staging_dir)
    monkeypatch.setattr(main_module, "SYNC_STATE", {k: dict(v) for k, v in main_module.SYNC_STATE.items()})
    monkeypatch.setattr(main_module, "store", DatasetStore(active_dir))

    local_client = TestClient(main_module.app)
    before = local_client.get("/api/overview").json()
    source_path = active_dir / "lms_weekly_activity.csv"
    incoming = pd.read_csv(source_path)
    incoming["lms_activity_index"] = (pd.to_numeric(incoming["lms_activity_index"], errors="coerce") - 15).clip(0, 100)

    uploaded = local_client.post(
        "/api/sources/lms_weekly_activity.csv/upload",
        files={"file": ("lms_weekly_activity.csv", incoming.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert uploaded.status_code == 200
    assert local_client.get("/api/sources").json()["items"][2]["staged_upload"] is True
    # Staging alone must not change active metrics.
    assert local_client.get("/api/overview").json()["average_engagement_score"] == before["average_engagement_score"]

    activated = local_client.post("/api/sources/lms_weekly_activity.csv/sync")
    assert activated.status_code == 200
    assert activated.json()["activated_upload"] is True
    after = local_client.get("/api/overview").json()
    assert after["student_count"] == before["student_count"]
    assert after["average_engagement_score"] < before["average_engagement_score"]
    assert activated.json()["analytics_after"]["average_lms_activity"] < activated.json()["analytics_before"]["average_lms_activity"]
    sources = {item["filename"]: item for item in local_client.get("/api/sources").json()["items"]}
    assert sources["lms_weekly_activity.csv"]["staged_upload"] is False
    assert sources["lms_weekly_activity.csv"]["row_count"] == len(incoming)


def test_uploaded_placement_preparation_recalculates_readiness_and_segments(tmp_path, monkeypatch):
    """Changed placement CSV values alter readiness/risk and derived segment counts."""
    import shutil
    import pandas as pd
    import app.main as main_module
    from app.services.analytics import DatasetStore

    active_dir = tmp_path / "active_data"
    staging_dir = tmp_path / "staging"
    active_dir.mkdir()
    staging_dir.mkdir()
    for path in main_module.DATA_DIR.glob("*.csv"):
        shutil.copy2(path, active_dir / path.name)

    monkeypatch.setattr(main_module, "DATA_DIR", active_dir)
    monkeypatch.setattr(main_module, "STAGING_DIR", staging_dir)
    monkeypatch.setattr(main_module, "SYNC_STATE", {k: dict(v) for k, v in main_module.SYNC_STATE.items()})
    monkeypatch.setattr(main_module, "store", DatasetStore(active_dir))
    local_client = TestClient(main_module.app)
    before = local_client.get("/api/overview").json()
    before_segments = local_client.get("/api/segments").json()["items"]

    incoming = pd.read_csv(active_dir / "placement_preparation_activity.csv", low_memory=False)
    score_cols = ["aptitude_assessment_score_100", "coding_assessment_score_100", "mock_coding_score_100",
                  "mock_interview_score_100", "resume_review_score_100"]
    for col in score_cols:
        incoming[col] = (pd.to_numeric(incoming[col], errors="coerce") - 18).clip(0, 100)

    uploaded = local_client.post(
        "/api/sources/placement_preparation_activity.csv/upload",
        files={"file": ("placement_preparation_activity.csv", incoming.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["join_diagnostic"]["match_pct"] == 100.0
    activated = local_client.post("/api/sources/placement_preparation_activity.csv/sync")
    assert activated.status_code == 200
    assert activated.json()["source_content_changed"] is True
    after = local_client.get("/api/overview").json()
    after_segments = local_client.get("/api/segments").json()["items"]
    assert after["average_placement_readiness_score"] < before["average_placement_readiness_score"]
    assert after["high_placement_risk_count"] != before["high_placement_risk_count"]
    assert after_segments != before_segments


def test_student_master_in_placement_source_is_rejected():
    """A wrong source type must fail required-column validation, not silently sync."""
    import pandas as pd
    import app.main as main_module
    master = pd.read_csv(main_module.DATA_DIR / "student_master.csv", nrows=10)
    response = client.post(
        "/api/sources/placement_preparation_activity.csv/upload",
        files={"file": ("student_master.csv", master.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "missing_columns" in detail


def test_blindspots_are_calculated_and_have_definitions():
    response = client.get('/api/blindspots')
    assert response.status_code == 200
    payload = response.json()
    assert payload['source'] == 'calculated from active dataset'
    assert payload['count'] == 4
    assert all('definition' in item and 'student_count' in item for item in payload['items'])


def test_interventions_are_persisted_in_sqlite():
    import app.main as main_module
    page = client.get('/api/students', params={'page_size': 1}).json()
    student_id = page['items'][0]['student_id']
    created = client.post('/api/interventions', json={
        'student_id': student_id,
        'action': 'Run a human-reviewed follow-up assessment',
        'owner': 'Placement team',
        'status': 'Planned',
        'baseline_metric': 'placement readiness',
        'baseline_value': 52.0,
        'follow_up_metric': 'placement readiness',
        'follow_up_date': '2026-11-01',
    })
    assert created.status_code == 200
    intervention_id = created.json()['id']
    found = client.get('/api/interventions', params={'student_id': student_id}).json()
    assert any(row['id'] == intervention_id and row['baseline_value'] == 52.0 for row in found['items'])


def test_schema_adapter_handles_common_kpmg_export_gaps():
    import pandas as pd
    from app.services.analytics import adapt_source_schema
    master = adapt_source_schema('student_master.csv', pd.DataFrame({'Student_ID':['X1'], 'Branch':['CSE']}))
    assert {'Program_Year', 'Snapshot_Date'}.issubset(master.columns)
    lms = adapt_source_schema('lms_weekly_activity.csv', pd.DataFrame({
        'Student_ID':['X1','X1'], 'week_start':['2026-01-05','2026-01-12'], 'lms_activity_index':[50,40]
    }))
    assert lms['week_number'].tolist() == [1,2]
    prep = adapt_source_schema('placement_preparation_activity.csv', pd.DataFrame({'Student_ID':['X1']}))
    assert 'aptitude_assessment_score_100' in prep and pd.isna(prep.loc[0, 'aptitude_assessment_score_100'])
