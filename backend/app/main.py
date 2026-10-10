from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import shutil
from datetime import datetime, timezone

from app.services.analytics import DatasetStore

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_DIR.parents[1] / "data"
DATA_DIR = Path(os.getenv("STUDENT_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()

store = DatasetStore(DATA_DIR)
STAGING_DIR = DATA_DIR.parent / "staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_META = {
    "student_master.csv": {"label": "Student Master / SIS", "group": "Core", "mode": "File import", "required": ["Student_ID", "Branch", "CGPA", "Attendance_Percentage", "Placement_Status"]},
    "academic_course_records.csv": {"label": "Academic Records", "group": "Academic", "mode": "File import", "required": ["Student_ID", "course_id", "total_marks_100", "backlog_flag"]},
    "lms_weekly_activity.csv": {"label": "Learning Management System", "group": "Learning", "mode": "File import", "required": ["Student_ID", "week_start", "assignments_due", "assignments_submitted", "assignments_on_time", "lms_activity_index"]},
    "engagement_summary.csv": {"label": "Engagement & Activities", "group": "Engagement", "mode": "File import", "required": ["Student_ID", "campus_events_attended", "projects_count"]},
    "placement_preparation_activity.csv": {"label": "Placement Preparation", "group": "Placement", "mode": "File import", "required": ["Student_ID", "mock_coding_score_100", "resume_review_score_100"]},
    "student_feedback.csv": {"label": "Student Feedback", "group": "Feedback", "mode": "File import", "required": ["Student_ID", "satisfaction_score_1_5", "support_requested"]},
    "course_catalog.csv": {"label": "Course Catalogue", "group": "Academic", "mode": "File import", "required": ["course_id", "branch", "course_name"]},
}
SYNC_STATE = {name: {"last_sync": None, "status": "Ready", "rows": 0, "message": "Using bundled synthetic demo data"} for name in SOURCE_META}
for _name in SOURCE_META:
    _path = DATA_DIR / _name
    if _path.exists(): SYNC_STATE[_name]["rows"] = sum(1 for _ in _path.open(encoding="utf-8", errors="ignore")) - 1
app = FastAPI(
    title="KPMG Student Success Intelligence API",
    description="Hackathon prototype API for synthetic campus analytics, risk review, segmentation and reporting.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    report_type: Literal["risk_snapshot", "institutional_brief"] = "risk_snapshot"
    branch: str = "All"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "student-success-api", "data_dir_configured": DATA_DIR.exists(), "student_count": int(len(store.students))}


@app.get("/api/sources")
def list_sources() -> dict:
    """Return source registry and latest sync state; real external connectors are not configured in demo mode."""
    items = []
    for filename, meta in SOURCE_META.items():
        state = SYNC_STATE[filename]
        path = DATA_DIR / filename
        items.append({"id": filename, "filename": filename, "label": meta["label"], "group": meta["group"], "connection_mode": meta["mode"], "status": state["status"], "row_count": state["rows"], "last_sync": state["last_sync"], "message": state["message"], "configured": path.exists(), "required_columns": meta["required"], "demo_only": True})
    return {"items": items, "integration_modes": ["REST API adapter", "Read-only database view", "Scheduled secure file import", "Webhook/event adapter"], "note": "Only file-backed synthetic demo sources are active. External systems require institution authorization and connector configuration."}

@app.post("/api/sources/{filename}/sync")
def sync_source(filename: str) -> dict:
    """Validate and refresh one bundled source. This is a real local file sync, not an external live connector."""
    global store
    if filename not in SOURCE_META:
        raise HTTPException(status_code=404, detail="Unknown source")
    path = DATA_DIR / filename
    if not path.exists(): raise HTTPException(status_code=404, detail="Source file is missing")
    try:
        frame = pd.read_csv(path)
        missing = sorted(set(SOURCE_META[filename]["required"]) - set(frame.columns))
        if missing: raise HTTPException(status_code=422, detail={"missing_columns": missing})
        if "Student_ID" in frame.columns and frame["Student_ID"].isna().any():
            raise HTTPException(status_code=422, detail="Student_ID contains empty values")
        # Rebuild the derived view from all source files so every sync is internally consistent.
        refreshed = DatasetStore(DATA_DIR)
        store = refreshed
        state = SYNC_STATE[filename]
        state.update({"last_sync": datetime.now(timezone.utc).isoformat(), "status": "Synced", "rows": int(len(frame)), "message": "Schema validated; analytics store refreshed from bundled demo sources"})
        return {"ok": True, "source": filename, "row_count": int(len(frame)), "last_sync": state["last_sync"], "student_count": int(len(store.students)), "message": state["message"]}
    except HTTPException: raise
    except Exception as exc:
        SYNC_STATE[filename].update({"status": "Error", "message": str(exc)[:240]})
        raise HTTPException(status_code=422, detail=f"Source refresh failed: {str(exc)[:240]}")

@app.post("/api/sources/sync-all")
def sync_all_sources() -> dict:
    """Validate every configured local demo source and rebuild analytics once."""
    global store
    checks = []
    for filename, meta in SOURCE_META.items():
        path = DATA_DIR / filename
        if not path.exists(): raise HTTPException(status_code=404, detail=f"Missing source: {filename}")
        try: frame = pd.read_csv(path)
        except Exception as exc: raise HTTPException(status_code=422, detail=f"Cannot read {filename}: {str(exc)[:160]}")
        missing = sorted(set(meta["required"]) - set(frame.columns))
        if missing: raise HTTPException(status_code=422, detail={"source": filename, "missing_columns": missing})
        checks.append((filename, int(len(frame))))
    try: store = DatasetStore(DATA_DIR)
    except Exception as exc: raise HTTPException(status_code=422, detail=f"Unified refresh failed: {str(exc)[:200]}")
    now = datetime.now(timezone.utc).isoformat()
    for filename, rows in checks:
        SYNC_STATE[filename].update({"last_sync": now, "status": "Synced", "rows": rows, "message": "Schema validated; analytics store refreshed"})
    return {"ok": True, "synced_sources": len(checks), "total_source_rows": sum(rows for _, rows in checks), "student_count": int(len(store.students)), "last_sync": now, "message": "All local demo sources validated and analytics refreshed once."}

@app.post("/api/sources/{filename}/upload")
async def upload_source(filename: str, file: UploadFile = File(...)) -> dict:
    """Stage and validate a CSV; it is not activated until a sync succeeds."""
    if filename not in SOURCE_META: raise HTTPException(status_code=404, detail="Unknown source")
    if not file.filename or not file.filename.lower().endswith(".csv"): raise HTTPException(status_code=415, detail="Upload a .csv file")
    staged = STAGING_DIR / filename
    try:
        with staged.open("wb") as target: shutil.copyfileobj(file.file, target)
        frame = pd.read_csv(staged, nrows=1000)
        missing = sorted(set(SOURCE_META[filename]["required"]) - set(frame.columns))
        if missing:
            staged.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail={"missing_columns": missing})
        return {"ok": True, "source": filename, "staged": True, "preview_rows": int(len(frame)), "columns": frame.columns.tolist(), "message": "File passed header validation and is staged. Bundled demo data remains active until an authorized activation workflow is configured."}
    except HTTPException: raise
    except Exception as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not validate CSV: {str(exc)[:200]}")

@app.get("/api/overview")
def overview() -> dict:
    return store.overview()


@app.get("/api/data-quality")
def data_quality() -> dict:
    return store.data_quality()


@app.get("/api/branches")
def branches() -> dict:
    values = sorted(store.students["Branch"].dropna().astype(str).unique().tolist())
    return {"items": ["All", *values]}


@app.get("/api/segments")
def segments() -> dict:
    return {"items": store.segments_summary()}


@app.get("/api/students")
def students(
    search: str = "",
    branch: str = "All",
    segment: str = "All",
    risk: str = "All",
    sort_by: str = "student_success_score",
    sort_dir: Literal["asc", "desc"] = "asc",
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> dict:
    return store.student_rows(search, branch, segment, risk, sort_by, sort_dir, page, page_size)


@app.get("/api/students/{student_id}")
def student_detail(student_id: str) -> dict:
    result = store.student_detail(student_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' was not found")
    return result


@app.get("/api/cohorts")
def cohorts() -> dict:
    return {"items": store.overview()["branches"]}


@app.get("/api/model-metrics")
def model_metrics() -> dict:
    return store.model_metrics


@app.post("/api/reports/generate")
def generate_report(request: ReportRequest) -> dict:
    return store.generate_report(request.report_type, request.branch)


@app.get("/api/score-method")
def score_method() -> dict:
    return {
        "academic": {"weight_in_success": 0.50, "formula": "45% normalized CGPA + 35% average course marks + 20% backlog score", "backlog_score": "max(0, 100 - 20 * backlog_count)"},
        "engagement": {"weight_in_success": 0.20, "formula": "40% attendance + 35% assignment on-time rate + 25% average LMS activity; reweight available components when a value is missing"},
        "placement_readiness": {"weight_in_success": 0.30, "formula": "25% aptitude + 25% mock coding + 20% mock interview + 10% resume review + 10% group discussion + 10% soft skills; interview excluded if no mock interview was completed"},
        "student_success": "50% academic + 20% engagement + 30% placement readiness",
        "academic_risk_rules": {"high": "academic score < 55, or at least 2 backlogs and CGPA < 7", "moderate": "academic score < 70, attendance < 65%, on-time rate < 50%, or LMS activity change <= -20; High takes precedence"},
        "placement_risk_model": "RandomForestClassifier trained on a stratified train/test split; Placement_Status is target only. Package_LPA, Age, Gender and State are excluded from predictors.",
        "important_caveat": "Synthetic demonstration only. Thresholds and scores require institutional review and validation before real use.",
    }
