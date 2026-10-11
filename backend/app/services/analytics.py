from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import date

import joblib
import numpy as np
import pandas as pd

from app.services.atlas_model_pipeline import build_feature_table


PRIMARY_KEYS = {
    "student_master.csv": "Student_ID",
    "academic_course_records.csv": "record_id",
    "lms_weekly_activity.csv": "activity_id",
    "engagement_summary.csv": "engagement_record_id",
    "placement_preparation_activity.csv": "prep_record_id",
    "student_feedback.csv": "feedback_id",
    "course_catalog.csv": "course_id",
    "placement_outcomes.csv": "Student_ID",
}


SOURCES = {
    "student_master.csv": ["Student_ID", "Branch", "Program_Year", "Snapshot_Date"],
    "academic_course_records.csv": ["record_id", "Student_ID", "semester", "course_id", "total_marks_100", "attendance_pct_course", "assignment_score_100", "backlog_flag"],
    "lms_weekly_activity.csv": ["activity_id", "Student_ID", "week_number", "week_start", "assignments_due", "assignments_submitted", "assignments_on_time", "lms_activity_index"],
    "engagement_summary.csv": ["engagement_record_id", "Student_ID", "campus_events_attended", "projects_count", "certifications_count"],
    "placement_preparation_activity.csv": ["prep_record_id", "Student_ID", "aptitude_assessment_score_100", "coding_assessment_score_100", "mock_coding_score_100", "mock_interview_score_100", "resume_review_score_100"],
    "student_feedback.csv": ["feedback_id", "Student_ID", "survey_date", "satisfaction_score_1_5", "support_requested"],
    "course_catalog.csv": ["course_id", "branch", "course_name"],
    "placement_outcomes.csv": ["Student_ID", "Placement_Status"],
}



# Low-cardinality text columns of the two big files. Storing them as pandas "category"
# instead of Python strings cuts RAM by ~5x (needed to fit Render's 512MB free tier).
CATEGORY_COLUMNS = {
    "academic_course_records.csv": [
        "course_id", "course_name", "course_start_date", "course_end_date",
        "result_status", "recorded_at", "source_system", "record_review_flag",
    ],
    "lms_weekly_activity.csv": ["week_start", "data_received_at", "source_system"],
}


def read_source_csv(path: Path, filename: str) -> pd.DataFrame:
    """Read a source CSV; large files get category dtypes for repetitive text columns."""
    cols = CATEGORY_COLUMNS.get(filename)
    if not cols:
        return pd.read_csv(path, low_memory=False)
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "category" for c in cols if c in header}
    return pd.read_csv(path, low_memory=False, dtype=dtypes)


def adapt_source_schema(filename: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common institution export variants to ATLAS canonical column names.

    Derived placeholder fields are clearly documented; this does not manufacture assessment scores.
    """
    aliases = {
        "Student_ID": ["student_id", "Student ID", "studentId", "Roll_No", "roll_no"],
        "Program_Year": ["program_year", "Year", "year_of_study", "Program Year"],
        "Snapshot_Date": ["snapshot_date", "as_of_date", "report_date", "Snapshot Date"],
        "week_number": ["week", "Week_Number", "week_no", "week_of_term"],
        "aptitude_assessment_score_100": ["aptitude_score", "aptitude_marks", "aptitude_score_100"],
        "coding_assessment_score_100": ["coding_score", "coding_marks", "coding_score_100"],
    }
    for canonical, candidates in aliases.items():
        if canonical not in frame.columns:
            source = next((c for c in candidates if c in frame.columns), None)
            if source:
                frame[canonical] = frame[source]
    if filename == "student_master.csv":
        if "Program_Year" not in frame:
            frame["Program_Year"] = pd.NA
        if "Snapshot_Date" not in frame:
            frame["Snapshot_Date"] = date.today().isoformat()
    if filename == "lms_weekly_activity.csv" and "week_number" not in frame and "week_start" in frame:
        dates = pd.to_datetime(frame["week_start"], errors="coerce")
        unique_dates = sorted(dates.dropna().unique())
        mapping = {pd.Timestamp(value): idx + 1 for idx, value in enumerate(unique_dates)}
        frame["week_number"] = dates.map(lambda value: mapping.get(value, pd.NA) if pd.notna(value) else pd.NA)
    if filename == "placement_preparation_activity.csv":
        # Preserve missing assessments as missing; the scoring pipeline reweights observed inputs.
        for column in ["aptitude_assessment_score_100", "coding_assessment_score_100"]:
            if column not in frame:
                frame[column] = pd.NA
    return frame

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 2)
    return value.item() if isinstance(value, np.generic) else value


def _weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted average that reweights available components instead of treating missing as zero."""
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for col, weight in weights.items():
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").clip(0, 100)
        present = values.notna()
        numerator.loc[present] += values.loc[present] * weight
        denominator.loc[present] += weight
    return (numerator / denominator.replace(0, np.nan)).clip(0, 100)


@dataclass
class DatasetStore:
    data_dir: Path

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.raw: dict[str, pd.DataFrame] = {}
        for filename, required in SOURCES.items():
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {path}")
            frame = adapt_source_schema(filename, read_source_csv(path, filename))
            missing_cols = sorted(set(required) - set(frame.columns))
            if missing_cols:
                raise ValueError(f"{filename} is missing required columns: {', '.join(missing_cols)}")
            if "Student_ID" in frame.columns:
                frame["Student_ID"] = frame["Student_ID"].astype("string").str.strip()
            self.raw[filename] = frame
        self.students = self._build_student_features()
        gc.collect()
        self.model, self.model_metrics = self._load_placement_model()
        self._apply_scores_and_segments()

    def _build_student_features(self) -> pd.DataFrame:
        master = self.raw["student_master.csv"].copy()
        master["Student_ID"] = master["Student_ID"].astype(str).str.strip()
        master = master.drop_duplicates("Student_ID", keep="first")

        academic = self.raw["academic_course_records.csv"][["record_id", "Student_ID", "total_marks_100", "backlog_flag", "attendance_pct_course", "assignment_score_100", "semester"]].copy()
        for col in ["total_marks_100", "backlog_flag", "attendance_pct_course", "assignment_score_100", "semester"]:
            academic[col] = pd.to_numeric(academic[col], errors="coerce")
        academic_agg = academic.groupby("Student_ID", as_index=False).agg(
            course_record_count=("record_id", "count"),
            average_course_marks=("total_marks_100", "mean"),
            marks_std=("total_marks_100", "std"),
            backlog_count=("backlog_flag", "sum"),
            pass_rate=("backlog_flag", lambda s: 100 * (1 - pd.to_numeric(s, errors="coerce").mean())),
            course_attendance_average=("attendance_pct_course", "mean"),
            course_assignment_average=("assignment_score_100", "mean"),
            semesters_observed=("semester", "nunique"),
        )
        # Expose semester means for cohort/profile context without treating them as separate histories.
        sem = academic.pivot_table(index="Student_ID", columns="semester", values="total_marks_100", aggfunc="mean")
        sem.columns = [f"semester_{int(c)}_average_marks" for c in sem.columns]
        academic_agg = academic_agg.merge(sem.reset_index(), on="Student_ID", how="left", validate="one_to_one")

        lms = self.raw["lms_weekly_activity.csv"][["Student_ID", "week_number", "week_start", "assignments_due", "assignments_submitted", "assignments_on_time", "lms_activity_index", "login_days", "active_minutes"]].copy()
        lms["week_start"] = pd.to_datetime(lms["week_start"].astype("object"), errors="coerce")
        for col in ["week_number", "assignments_due", "assignments_submitted", "assignments_on_time", "lms_activity_index", "login_days", "active_minutes"]:
            if col in lms:
                lms[col] = pd.to_numeric(lms[col], errors="coerce")
        lms_agg = lms.groupby("Student_ID", as_index=False).agg(
            lms_weeks_observed=("week_number", "nunique"),
            lms_average_activity=("lms_activity_index", "mean"),
            lms_activity_std=("lms_activity_index", "std"),
            lms_average_login_days=("login_days", "mean"),
            lms_average_active_minutes=("active_minutes", "mean"),
            lms_assignments_due=("assignments_due", "sum"),
            lms_assignments_submitted=("assignments_submitted", "sum"),
            lms_assignments_on_time=("assignments_on_time", "sum"),
        )
        lms_agg["lms_weeks_missing"] = (16 - lms_agg["lms_weeks_observed"]).clip(lower=0)
        lms_agg["assignment_submission_rate"] = np.where(
            lms_agg["lms_assignments_due"] > 0,
            100 * lms_agg["lms_assignments_submitted"] / lms_agg["lms_assignments_due"], np.nan,
        )
        lms_agg["assignment_on_time_rate"] = np.where(
            lms_agg["lms_assignments_submitted"] > 0,
            100 * lms_agg["lms_assignments_on_time"] / lms_agg["lms_assignments_submitted"], np.nan,
        )
        # Faster analysis of large datasets (300K rows) without per-student loops; missing weeks remain missing, not zero.
        trend_data = lms.dropna(subset=["lms_activity_index", "week_number"]).sort_values(["Student_ID", "week_number"]).copy()
        if not trend_data.empty:
            trend_data["_week_x"] = trend_data["week_number"].astype(float)
            trend_data["_activity_y"] = trend_data["lms_activity_index"].astype(float)
            trend_data["_xy"] = trend_data["_week_x"] * trend_data["_activity_y"]
            trend_stats = trend_data.groupby("Student_ID", sort=False).agg(
                _n=("_activity_y", "count"), _sx=("_week_x", "sum"), _sy=("_activity_y", "sum"),
                _sxx=("_week_x", lambda values: float(np.square(values).sum())), _sxy=("_xy", "sum"),
                _unique_weeks=("week_number", "nunique"),
            )
            _denom = trend_stats["_sxx"] - trend_stats["_sx"].pow(2) / trend_stats["_n"].replace(0, np.nan)
            _numer = trend_stats["_sxy"] - trend_stats["_sx"] * trend_stats["_sy"] / trend_stats["_n"].replace(0, np.nan)
            trend_stats["lms_activity_trend_per_week"] = _numer / _denom.replace(0, np.nan)
            trend_data["_within_student_index"] = trend_data.groupby("Student_ID", sort=False).cumcount()
            trend_data["_early_half"] = trend_data["_within_student_index"] < (trend_data.groupby("Student_ID", sort=False)["_activity_y"].transform("count") / 2)
            halves = trend_data.groupby(["Student_ID", "_early_half"])["_activity_y"].mean().unstack("_early_half")
            if False not in halves.columns: halves[False] = np.nan
            if True not in halves.columns: halves[True] = np.nan
            trend_stats["lms_early_activity"] = halves[True]
            trend_stats["lms_recent_activity"] = halves[False]
            trend_stats["lms_activity_change"] = trend_stats["lms_recent_activity"] - trend_stats["lms_early_activity"]
            invalid = (trend_stats["_n"] < 3) | (trend_stats["_unique_weeks"] < 2)
            trend_cols = ["lms_early_activity", "lms_recent_activity", "lms_activity_change", "lms_activity_trend_per_week"]
            trend_stats.loc[invalid, trend_cols] = np.nan
            lms_agg = lms_agg.merge(trend_stats[trend_cols].reset_index(), on="Student_ID", how="left", validate="one_to_one")

        engagement = self.raw["engagement_summary.csv"].copy()
        engagement = engagement.drop_duplicates("Student_ID", keep="last")
        for col in ["clubs_joined", "campus_events_attended", "hackathons_participated", "career_events_attended", "volunteer_hours", "certifications_count", "projects_count", "internships_count", "open_source_contributions", "github_contributions"]:
            if col in engagement:
                engagement[col] = pd.to_numeric(engagement[col], errors="coerce")

        feedback = self.raw["student_feedback.csv"].copy()
        feedback["support_requested_flag"] = feedback["support_requested"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"]).astype(int)
        for col in ["satisfaction_score_1_5", "faculty_clarity_1_5", "workload_manageability_1_5", "lms_usability_1_5"]:
            feedback[col] = pd.to_numeric(feedback[col], errors="coerce")
        feedback_agg = feedback.groupby("Student_ID", as_index=False).agg(
            feedback_response_count=("feedback_id", "count"),
            satisfaction_average=("satisfaction_score_1_5", "mean"),
            faculty_clarity_average=("faculty_clarity_1_5", "mean"),
            workload_manageability_average=("workload_manageability_1_5", "mean"),
            lms_usability_average=("lms_usability_1_5", "mean"),
            support_request_count=("support_requested_flag", "sum"),
        )
        feedback_agg["support_requested"] = (feedback_agg["support_request_count"] > 0).astype(int)
        prep = self.raw["placement_preparation_activity.csv"].drop_duplicates("Student_ID", keep="last").copy()
        for col in ["aptitude_assessment_score_100", "coding_assessment_score_100", "communication_assessment_score_100", "mock_coding_score_100", "mock_interview_score_100", "resume_review_score_100", "group_discussion_score_100", "mock_interviews_completed"]:
            if col in prep:
                prep[col] = pd.to_numeric(prep[col], errors="coerce")

        pieces = [academic_agg, lms_agg, engagement, feedback_agg, prep]
        merged = master
        for right in pieces:
            merged = merged.merge(right, on="Student_ID", how="left", validate="one_to_one")
        # Student-level completion is based on indicators, not whether optional feedback exists.
        completeness_cols = [
            "average_course_marks", "course_attendance_average", "course_assignment_average",
            "lms_average_activity", "assignment_on_time_rate", "aptitude_assessment_score_100",
            "mock_coding_score_100", "resume_review_score_100", "satisfaction_average",
        ]
        observed = [c for c in completeness_cols if c in merged]
        merged["data_completeness_pct"] = (merged[observed].notna().sum(axis=1) / len(observed) * 100).round(1)
        merged["feedback_response_count"] = merged.get("feedback_response_count", pd.Series(index=merged.index, dtype=float)).fillna(0)
        merged["support_requested"] = merged.get("support_requested", pd.Series(index=merged.index, dtype=float)).fillna(0).astype(int)
        return merged

    def _load_placement_model(self):
        backend_dir = Path(__file__).resolve().parents[2]
        model_path = backend_dir / "models" / "atlas_placement_logistic_pipeline.joblib"
        metrics_path = backend_dir / "models" / "placement_model_metrics.json"
        metrics: dict[str, Any] = {"available": False, "reason": "The bundled placement outlook model is not available."}
        if metrics_path.exists():
            try:
                raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                lr = raw_metrics.get("logistic_regression", {})
                dummy = raw_metrics.get("dummy_baseline", {})
                metrics = {
                    "available": model_path.exists(),
                    "model": "Logistic Regression",
                    "target": "Placement_Status (synthetic outcome)",
                    "rows_total": raw_metrics.get("rows_total"),
                    "features_used": raw_metrics.get("features_used"),
                    "train_rows": raw_metrics.get("train_rows"),
                    "test_rows": raw_metrics.get("test_rows"),
                    "placement_rate_overall": raw_metrics.get("placement_rate_overall"),
                    "roc_auc_not_placed": lr.get("roc_auc"),
                    "pr_auc_not_placed": lr.get("pr_auc"),
                    "accuracy_at_0_5": lr.get("accuracy_at_0_5"),
                    "precision_at_0_5": lr.get("precision_at_0_5"),
                    "recall_at_0_5": lr.get("recall_at_0_5"),
                    "f1_at_0_5": lr.get("f1_at_0_5"),
                    "dummy_baseline_roc_auc": dummy.get("roc_auc"),
                    "provenance_warning": raw_metrics.get("dataset_provenance", "Synthetic data only; not real-world performance."),
                }
            except Exception as exc:
                metrics = {"available": False, "reason": f"Could not read model metrics: {type(exc).__name__}"}
        if not model_path.exists():
            return None, metrics
        try:
            model = joblib.load(model_path)
           
            X, _, _ = build_feature_table(self.data_dir)
            expected = list(getattr(model, "feature_names_in_", []))
            
            if expected:
                X = X.reindex(columns=expected)
            self.model_features = X.reindex(self.students["Student_ID"].astype(str).tolist())
            return model, metrics
        except Exception as exc:
            return None, {"available": False, "reason": f"Placement model could not load: {type(exc).__name__}: {exc}"}

    def _apply_scores_and_segments(self) -> None:
        df = self.students
        # Academic score
        acad_src = pd.DataFrame({
            "marks": pd.to_numeric(df["average_course_marks"], errors="coerce"),
            "attendance": pd.to_numeric(df["course_attendance_average"], errors="coerce"),
            "pass_rate": pd.to_numeric(df["pass_rate"], errors="coerce"),
        }, index=df.index)
        df["academic_score"] = _weighted_score(acad_src, {"marks": .70, "attendance": .20, "pass_rate": .10}).round(1)

        attendance = pd.to_numeric(df["course_attendance_average"], errors="coerce")
        activity = pd.to_numeric(df["lms_average_activity"], errors="coerce")
        on_time = pd.to_numeric(df["assignment_on_time_rate"], errors="coerce")
        eng_src = pd.DataFrame({"attendance": attendance, "on_time": on_time, "activity": activity}, index=df.index)
        df["engagement_score"] = _weighted_score(eng_src, {"attendance": .35, "on_time": .35, "activity": .30}).round(1)

        interview_count = pd.to_numeric(df["mock_interviews_completed"], errors="coerce").fillna(0)
        interview_score = pd.to_numeric(df["mock_interview_score_100"], errors="coerce").where(interview_count > 0)
        place_src = pd.DataFrame({
            "aptitude": pd.to_numeric(df["aptitude_assessment_score_100"], errors="coerce"),
            "coding": pd.to_numeric(df["mock_coding_score_100"], errors="coerce"),
            "interview": interview_score,
            "resume": pd.to_numeric(df["resume_review_score_100"], errors="coerce"),
            "group_discussion": pd.to_numeric(df["group_discussion_score_100"], errors="coerce"),
            "communication": pd.to_numeric(df["communication_assessment_score_100"], errors="coerce"),
        }, index=df.index)
        df["placement_readiness_score"] = _weighted_score(place_src, {
            "aptitude": .25, "coding": .25, "interview": .20, "resume": .10,
            "group_discussion": .10, "communication": .10,
        }).round(1)
        df["student_success_score"] = _weighted_score(pd.DataFrame({
            "academic": df["academic_score"], "engagement": df["engagement_score"],
            "placement": df["placement_readiness_score"],
        }, index=df.index), {"academic": .50, "engagement": .20, "placement": .30}).round(1)

        df["academic_risk_level"] = "Unavailable"
        has_acad = df["course_record_count"].fillna(0) > 0
        high_acad = has_acad & ((df["academic_score"] < 55) | ((df["backlog_count"] >= 2) & (df["average_course_marks"] < 55)))
        moderate_acad = has_acad & ((df["academic_score"] < 70) | (attendance < 65) | (on_time < 50) | (pd.to_numeric(df["lms_activity_change"], errors="coerce") <= -15))
        df.loc[has_acad, "academic_risk_level"] = "Low"
        df.loc[moderate_acad, "academic_risk_level"] = "Moderate"
        df.loc[high_acad, "academic_risk_level"] = "High"

        df["placement_risk_probability"] = np.nan
        df["placement_risk_level"] = "Unavailable"
        if self.model is not None:
            X = self.model_features.reindex(df["Student_ID"].astype(str).tolist())
            classes = list(self.model.classes_)
            not_placed_class = 0 if 0 in classes else ("Not Placed" if "Not Placed" in classes else None)
            if not_placed_class is not None:
                probs = self.model.predict_proba(X)[:, classes.index(not_placed_class)]
                df["placement_risk_probability"] = probs * 100
                df["placement_risk_level"] = np.select([probs >= .65, probs >= .40], ["High", "Moderate"], default="Low")

        drivers: list[list[str]] = []
        segments: list[str] = []
        for row in df.itertuples(index=False):
            why: list[str] = []
            if _safe_float(getattr(row, "average_course_marks", np.nan), np.nan) < 55:
                why.append("Average course marks below 55/100")
            if _safe_float(getattr(row, "course_attendance_average", np.nan), np.nan) < 65:
                why.append("Course attendance below 65%")
            if _safe_float(getattr(row, "backlog_count", np.nan), np.nan) >= 1:
                why.append(f"{int(_safe_float(getattr(row, 'backlog_count', 0)))} course backlog(s) recorded")
            if _safe_float(getattr(row, "assignment_on_time_rate", np.nan), np.nan) < 50:
                why.append("Fewer than half of submitted assignments were on time")
            if _safe_float(getattr(row, "lms_activity_change", np.nan), np.nan) <= -15:
                why.append("LMS activity lower in the recent observed weeks")
            if _safe_float(getattr(row, "mock_coding_score_100", np.nan), np.nan) < 50:
                why.append("Mock coding assessment below 50/100")
            if _safe_float(getattr(row, "aptitude_assessment_score_100", np.nan), np.nan) < 40:
                why.append("Aptitude assessment below 40/100")
            if _safe_float(getattr(row, "support_requested", 0)) > 0:
                why.append("Student feedback includes a support request")
            drivers.append(why[:6])

            a = _safe_float(getattr(row, "academic_score", np.nan), np.nan)
            p = _safe_float(getattr(row, "placement_readiness_score", np.nan), np.nan)
            e = _safe_float(getattr(row, "engagement_score", np.nan), np.nan)
            ar = getattr(row, "academic_risk_level", "Unavailable")
            pr = getattr(row, "placement_risk_level", "Unavailable")
            if np.isfinite(a) and a < 60 and np.isfinite(e) and e < 60:
                label = "Academic & Engagement Support"
            elif _safe_float(getattr(row, "support_requested", 0)) > 0:
                label = "Student-Requested Support"
            elif np.isfinite(a) and a >= 70 and np.isfinite(p) and p < 65:
                skill_columns = {
                    "Coding": "mock_coding_score_100",
                    "Aptitude": "aptitude_assessment_score_100",
                    "Interview": "mock_interview_score_100",
                    "Resume": "resume_review_score_100",
                    "Group Discussion": "group_discussion_score_100",
                    "Communication": "communication_assessment_score_100",
                }
                observed_skills = [(name, _safe_float(getattr(row, col, np.nan), np.nan)) for name, col in skill_columns.items()]
                observed_skills = [(name, value) for name, value in observed_skills if np.isfinite(value)]
                weakest = min(observed_skills, key=lambda item: item[1]) if observed_skills else None
                label = f"Placement · {weakest[0]} Gap" if weakest and weakest[1] < 65 else "Placement Readiness Review"
            elif ar == "High" or pr == "High" or (np.isfinite(e) and e < 45):
                label = "Priority Review"
            elif np.isfinite(a) and a >= 70 and np.isfinite(p) and p >= 70:
                label = "Balanced / Placement Ready"
            else:
                label = "Mixed Signals · Monitor"
            segments.append(label)
        df["explanation_drivers"] = drivers
        df["segment"] = segments

    def data_quality(self) -> dict[str, Any]:
        master = self.raw["student_master.csv"]
        student_ids = set(master["Student_ID"].astype(str).str.strip())
        source_rows, join_results = [], []
        for filename, frame in self.raw.items():
            key = PRIMARY_KEYS.get(filename)
            duplicates = int(frame[key].duplicated().sum()) if key and key in frame.columns else 0
            student_key = "Student_ID" if "Student_ID" in frame.columns else None
            orphans = int((~frame[student_key].astype(str).str.strip().isin(student_ids)).sum()) if student_key and filename != "student_master.csv" else 0
            missing_required = [c for c in SOURCES[filename] if c in frame and frame[c].isna().any()]
            status = "Check" if duplicates or orphans or missing_required else "OK"
            source_rows.append({"source": filename, "row_count": int(len(frame)), "column_count": int(len(frame.columns)),
                                "duplicate_key_rows": duplicates, "orphan_student_rows": orphans,
                                "required_columns_with_missing_values": missing_required, "status": status})
            if student_key and filename != "student_master.csv":
                ids = set(frame[student_key].dropna().astype(str).str.strip())
                join_results.append({"source": filename, "unique_student_ids": len(ids),
                                     "matched_student_ids": len(ids & student_ids), "orphan_student_ids": len(ids - student_ids),
                                     "coverage_pct_of_master": round(100 * len(ids & student_ids) / max(1, len(student_ids)), 1)})
        feedback_ids = set(self.raw["student_feedback.csv"]["Student_ID"].dropna().astype(str).str.strip())
        ids = set(master["Student_ID"].astype(str).str.strip())
        return {
            "provenance": "All records in this dataset bundle are synthetic demonstration records.",
            "student_count": int(len(ids)), "duplicate_master_ids": int(master["Student_ID"].duplicated().sum()),
            "students_with_feedback": len(feedback_ids & ids), "students_without_feedback": len(ids - feedback_ids),
            "sources": source_rows, "join_coverage": join_results,
            "notes": [
                "One-to-many academic, LMS and feedback records are aggregated before the unified student view is created.",
                "Placement_Status and package/offer fields are outcomes and never predictors.",
                "Branch, age, gender and home state are excluded from model inputs and individual score drivers.",
                "Missing records remain missing; no feedback response is not interpreted as dissatisfaction.",
                "Model estimates come from a synthetic experiment and are not validated real-college predictions.",
            ],
        }

    def overview(self) -> dict[str, Any]:
        df = self.students
        branch = df.groupby("Branch", dropna=False).agg(
            students=("Student_ID", "nunique"), average_success_score=("student_success_score", "mean"),
            academic_risk_count=("academic_risk_level", lambda x: int((x == "High").sum())),
            placement_risk_count=("placement_risk_level", lambda x: int((x == "High").sum())),
        ).reset_index().sort_values("students", ascending=False)
        lms = self.raw["lms_weekly_activity.csv"][["week_start", "week_number", "lms_activity_index", "assignments_due", "assignments_submitted", "assignments_on_time"]].copy()
        lms["week_start"] = pd.to_datetime(lms["week_start"].astype("object"), errors="coerce")
        lms["week_number"] = pd.to_numeric(lms["week_number"], errors="coerce")
        trend = lms.groupby("week_number").agg(
            week_start=("week_start", "min"), average_activity=("lms_activity_index", "mean"),
            assignments_due=("assignments_due", "sum"), assignments_submitted=("assignments_submitted", "sum"),
            assignments_on_time=("assignments_on_time", "sum"),
        ).reset_index().sort_values("week_number")
        trend["on_time_rate"] = np.where(trend["assignments_submitted"] > 0, 100 * trend["assignments_on_time"] / trend["assignments_submitted"], np.nan)
        feedback = self.raw["student_feedback.csv"]
        return {
            "provenance": "Synthetic demonstration data — not real college records.",
            "student_count": int(df["Student_ID"].nunique()),
            "average_success_score": round(float(df["student_success_score"].mean()), 1),
            "average_academic_score": round(float(df["academic_score"].mean()), 1),
            "average_engagement_score": round(float(df["engagement_score"].mean()), 1),
            "average_placement_readiness_score": round(float(df["placement_readiness_score"].mean()), 1),
            "high_academic_risk_count": int((df["academic_risk_level"] == "High").sum()),
            "moderate_academic_risk_count": int((df["academic_risk_level"] == "Moderate").sum()),
            "high_placement_risk_count": int((df["placement_risk_level"] == "High").sum()),
            "moderate_placement_risk_count": int((df["placement_risk_level"] == "Moderate").sum()),
            "support_requests": int(pd.to_numeric(df["support_requested"], errors="coerce").fillna(0).sum()),
            "average_data_completeness_pct": round(float(df["data_completeness_pct"].mean()), 1),
            "model_metrics": self.model_metrics,
            "branches": [{"branch": str(r.Branch), "students": int(r.students),
                          "average_success_score": round(float(r.average_success_score), 1),
                          "academic_risk_count": int(r.academic_risk_count), "placement_risk_count": int(r.placement_risk_count)}
                         for r in branch.itertuples(index=False)],
            "weekly_trend": [{"week": f"W{int(r.week_number):02d}" if pd.notna(r.week_number) else "Unknown",
                              "average_activity": round(float(r.average_activity), 1) if pd.notna(r.average_activity) else None,
                              "on_time_rate": round(float(r.on_time_rate), 1) if pd.notna(r.on_time_rate) else None}
                             for r in trend.itertuples(index=False)],
            "feedback_records": int(len(feedback)),
        }

    def segments_summary(self) -> list[dict[str, Any]]:
        grouped = self.students.groupby("segment").agg(
            student_count=("Student_ID", "nunique"), average_success_score=("student_success_score", "mean"),
            average_academic_score=("academic_score", "mean"), average_placement_score=("placement_readiness_score", "mean"),
            high_academic_risk=("academic_risk_level", lambda x: int((x == "High").sum())),
            high_placement_risk=("placement_risk_level", lambda x: int((x == "High").sum())),
        ).reset_index().sort_values("student_count", ascending=False)
        return [{"segment": r.segment, "student_count": int(r.student_count),
                 "average_success_score": round(float(r.average_success_score), 1),
                 "average_academic_score": round(float(r.average_academic_score), 1),
                 "average_placement_score": round(float(r.average_placement_score), 1),
                 "high_academic_risk": int(r.high_academic_risk), "high_placement_risk": int(r.high_placement_risk)}
                for r in grouped.itertuples(index=False)]

    def student_rows(self, search: str = "", branch: str = "All", segment: str = "All", risk: str = "All",
                     sort_by: str = "student_success_score", sort_dir: str = "asc", page: int = 1, page_size: int = 25) -> dict[str, Any]:
        df = self.students.copy()
        if search.strip():
            df = df[df["Student_ID"].astype(str).str.lower().str.contains(search.strip().lower(), regex=False)]
        if branch and branch != "All": df = df[df["Branch"].astype(str) == branch]
        if segment and segment != "All": df = df[df["segment"] == segment]
        filters = {"Academic High": ("academic_risk_level", "High"), "Academic Moderate": ("academic_risk_level", "Moderate"),
                   "Placement High": ("placement_risk_level", "High"), "Placement Moderate": ("placement_risk_level", "Moderate")}
        if risk == "Any High": df = df[(df["academic_risk_level"] == "High") | (df["placement_risk_level"] == "High")]
        elif risk in filters:
            col, level = filters[risk]
            df = df[df[col] == level]
        sort_map = {"student_success_score": "student_success_score", "academic_score": "academic_score", "engagement_score": "engagement_score",
                    "placement_readiness_score": "placement_readiness_score", "average_course_marks": "average_course_marks", "Branch": "Branch",
                    "placement_risk_probability": "placement_risk_probability"}
        df = df.sort_values(sort_map.get(sort_by, "student_success_score"), ascending=(sort_dir != "desc"), na_position="last")
        total = len(df)
        page, page_size = max(1, page), max(1, min(100, page_size))
        subset = df.iloc[(page - 1) * page_size:page * page_size]
        items = []
        for _, row in subset.iterrows():
            items.append({
                "student_id": str(row["Student_ID"]), "branch": str(row["Branch"]),
                "program_year": _maybe(row.get("Program_Year")),
                "average_course_marks": _maybe(row.get("average_course_marks")),
                "attendance": _maybe(row.get("course_attendance_average")),
                "success_score": _maybe(row.get("student_success_score")), "academic_score": _maybe(row.get("academic_score")),
                "engagement_score": _maybe(row.get("engagement_score")), "placement_readiness_score": _maybe(row.get("placement_readiness_score")),
                "academic_risk": str(row.get("academic_risk_level", "Unavailable")), "placement_risk": str(row.get("placement_risk_level", "Unavailable")),
                "placement_risk_probability": _maybe(row.get("placement_risk_probability")), "segment": str(row.get("segment", "Mixed Signals · Monitor")),
                "explanation_drivers": list(row.get("explanation_drivers", [])),
                "data_completeness_pct": _maybe(row.get("data_completeness_pct")),
            })
        return {"total": int(total), "page": page, "page_size": page_size, "items": items}

    def student_detail(self, student_id: str) -> dict[str, Any] | None:
        matched = self.students[self.students["Student_ID"].astype(str) == str(student_id)]
        if matched.empty: return None
        row = matched.iloc[0]
        sid = str(row["Student_ID"])
        lms = self.raw["lms_weekly_activity.csv"]
        lms_rows = lms[lms["Student_ID"].astype(str) == sid].copy().sort_values("week_number")
        trend = [{"week": f"W{int(r.week_number):02d}", "activity_index": _maybe(r.lms_activity_index),
                  "on_time_rate": (round(100 * _safe_float(r.assignments_on_time) / _safe_float(r.assignments_submitted), 1)
                                   if _safe_float(r.assignments_submitted) > 0 else None)}
                 for r in lms_rows.itertuples(index=False)]
        acad = self.raw["academic_course_records.csv"]
        acad_rows = acad[acad["Student_ID"].astype(str) == sid].sort_values(["semester", "course_id"])
        course_records = [{"course_id": str(r.course_id), "course_name": str(r.course_name), "semester": int(r.semester),
                           "marks": _maybe(r.total_marks_100), "attendance": _maybe(r.attendance_pct_course),
                           "assignment_score": _maybe(r.assignment_score_100), "backlog": bool(_safe_float(r.backlog_flag))}
                          for r in acad_rows.itertuples(index=False)]
        feedback = self.raw["student_feedback.csv"]
        feedback_rows = feedback[feedback["Student_ID"].astype(str) == sid].copy().sort_values("survey_date")
        feedback_records = [{"date": str(r.survey_date), "satisfaction": _maybe(r.satisfaction_score_1_5),
                             "faculty_clarity": _maybe(r.faculty_clarity_1_5), "workload_manageability": _maybe(r.workload_manageability_1_5),
                             "support_requested": str(r.support_requested), "topic": str(r.feedback_topic)}
                            for r in feedback_rows.itertuples(index=False)]
        drivers = list(row.get("explanation_drivers", []))
        return {
            "student_id": sid, "branch": str(row["Branch"]), "program_year": _maybe(row.get("Program_Year")),
            "success_score": _maybe(row.get("student_success_score")), "academic_score": _maybe(row.get("academic_score")),
            "engagement_score": _maybe(row.get("engagement_score")), "placement_readiness_score": _maybe(row.get("placement_readiness_score")),
            "academic_risk": str(row["academic_risk_level"]), "placement_risk": str(row["placement_risk_level"]),
            "placement_risk_probability": _maybe(row.get("placement_risk_probability")), "segment": str(row["segment"]),
            "data_completeness_pct": _maybe(row.get("data_completeness_pct")), "explanation_drivers": drivers,
            "metrics": {"average_course_marks": _maybe(row.get("average_course_marks")), "course_attendance_average": _maybe(row.get("course_attendance_average")),
                        "course_assignment_average": _maybe(row.get("course_assignment_average")), "backlog_count": _maybe(row.get("backlog_count")),
                        "assignment_submission_rate": _maybe(row.get("assignment_submission_rate")), "assignment_on_time_rate": _maybe(row.get("assignment_on_time_rate")),
                        "lms_average_activity": _maybe(row.get("lms_average_activity")), "lms_activity_change": _maybe(row.get("lms_activity_change")),
                        "aptitude_score": _maybe(row.get("aptitude_assessment_score_100")), "coding_assessment_score": _maybe(row.get("coding_assessment_score_100")),
                        "mock_coding_score": _maybe(row.get("mock_coding_score_100")), "mock_interviews_completed": _maybe(row.get("mock_interviews_completed")),
                        "mock_interview_score": _maybe(row.get("mock_interview_score_100")) if _safe_float(row.get("mock_interviews_completed")) > 0 else None,
                        "resume_review_score": _maybe(row.get("resume_review_score_100")), "satisfaction_average": _maybe(row.get("satisfaction_average")),
                        "support_requested": bool(_safe_float(row.get("support_requested", 0)))},
            "weekly_lms_activity": trend, "course_records": course_records, "feedback_records": feedback_records,
            "score_method": {
                "academic": "70% average course marks + 20% course attendance + 10% pass rate; reweighted over available components.",
                "engagement": "35% course attendance + 35% on-time rate among submitted assignments + 30% average LMS activity; missing components are reweighted.",
                "placement_readiness": "25% aptitude + 25% mock coding + 20% mock interview + 10% resume + 10% group discussion + 10% communication; missing assessments are reweighted.",
                "success": "50% academic score + 20% engagement score + 30% placement-readiness score; missing components are reweighted.",
                "caveat": "Prototype rules on synthetic data; thresholds are not validated institutional policy.",
            },
        }

    def generate_report(self, report_type: str = "risk_snapshot", branch: str = "All") -> dict[str, Any]:
        df = self.students.copy()
        if branch and branch != "All": df = df[df["Branch"].astype(str) == branch]
        if df.empty:
            return {"title": "Student Success Report", "summary": "No records match the selected filters.", "findings": [], "scope": {"branch": branch, "students": 0}}
        high_acad = int((df["academic_risk_level"] == "High").sum())
        high_place = int((df["placement_risk_level"] == "High").sum())
        weak_ready = int(((df["academic_score"] >= 70) & (df["placement_readiness_score"] < 55)).sum())
        low_trend = int((pd.to_numeric(df["lms_activity_change"], errors="coerce") <= -15).sum())
        findings = [
            {"title": "Academic review queue", "metric": high_acad, "unit": "students", "detail": "Students flagged High by transparent review rules. Verify source records and context before intervening."},
            {"title": "Placement outlook review", "metric": high_place, "unit": "students", "detail": "Students with a high synthetic-model estimate of not being placed; this is a review signal, not a guarantee."},
            {"title": "Academic-to-placement gap", "metric": weak_ready, "unit": "students", "detail": "Students whose academic score is at least 70 while placement readiness is below 55."},
            {"title": "Lower recent LMS activity", "metric": low_trend, "unit": "students", "detail": "Students whose recent observed LMS activity is at least 15 points below the earlier observed period."},
        ]
        title = "Student Success Risk Snapshot" if report_type == "risk_snapshot" else "Student Success Institutional Brief"
        return {"title": title,
                "summary": f"Analyzed {len(df):,} synthetic student profiles for {branch if branch != 'All' else 'all branches'}. Findings identify groups for review; they do not establish causes or validated real-world outcomes.",
                "scope": {"branch": branch, "students": int(len(df)), "synthetic_data": True},
                "mean_scores": {"success": round(float(df["student_success_score"].mean()), 1), "academic": round(float(df["academic_score"].mean()), 1),
                                "engagement": round(float(df["engagement_score"].mean()), 1), "placement_readiness": round(float(df["placement_readiness_score"].mean()), 1)},
                "findings": findings,
                "recommended_checks": ["Review source records and student context before deciding on support.",
                                       "Compare the same course and semester across cohorts before inferring a bottleneck.",
                                       "Track follow-up outcomes only when intervention delivery and later outcomes are recorded."],
                "limitations": ["The dataset is synthetic.", "Score thresholds are prototype heuristics, not validated institutional policy.",
                                "Descriptive associations do not establish causes."],
        }
