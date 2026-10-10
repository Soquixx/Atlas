from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PRIMARY_KEYS = {
    "student_master.csv": "Student_ID",
    "academic_course_records.csv": "record_id",
    "lms_weekly_activity.csv": "activity_id",
    "engagement_summary.csv": "engagement_record_id",
    "placement_preparation_activity.csv": "prep_record_id",
    "student_feedback.csv": "feedback_id",
    "course_catalog.csv": "course_id",
}

SOURCES = {
    "student_master.csv": ["Student_ID", "Branch", "CGPA", "Attendance_Percentage", "Placement_Status"],
    "academic_course_records.csv": ["Student_ID", "course_id", "total_marks_100", "backlog_flag"],
    "lms_weekly_activity.csv": ["Student_ID", "week_start", "assignments_due", "assignments_submitted", "assignments_on_time", "lms_activity_index"],
    "engagement_summary.csv": ["Student_ID", "campus_events_attended", "projects_count"],
    "placement_preparation_activity.csv": ["Student_ID", "mock_coding_score_100", "resume_review_score_100"],
    "student_feedback.csv": ["Student_ID", "satisfaction_score_1_5", "support_requested"],
    "course_catalog.csv": ["course_id", "branch", "course_name"],
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    result = numerator / denominator.replace(0, np.nan)
    return result.clip(0, 100)


@dataclass
class DatasetStore:
    data_dir: Path

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.raw: dict[str, pd.DataFrame] = {}
        for filename in SOURCES:
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {path}")
            frame = pd.read_csv(path)
            missing_cols = sorted(set(SOURCES[filename]) - set(frame.columns))
            if missing_cols:
                raise ValueError(f"{filename} is missing required columns: {', '.join(missing_cols)}")
            if "Student_ID" in frame.columns:
                frame["Student_ID"] = frame["Student_ID"].astype(str).str.strip()
            self.raw[filename] = frame
        self.students = self._build_student_features()
        self.model, self.model_metrics = self._train_placement_model()
        self._apply_scores_and_segments()

    def _build_student_features(self) -> pd.DataFrame:
        master = self.raw["student_master.csv"].copy()
        master["Student_ID"] = master["Student_ID"].astype(str).str.strip()
        master = master.drop_duplicates("Student_ID", keep="first")

        academic = self.raw["academic_course_records.csv"].copy()
        for col in ["total_marks_100", "backlog_flag", "attendance_pct_course", "assignment_score_100"]:
            academic[col] = pd.to_numeric(academic[col], errors="coerce")
        academic_agg = academic.groupby("Student_ID", as_index=False).agg(
            course_record_count=("course_id", "nunique"),
            average_course_marks=("total_marks_100", "mean"),
            backlog_count=("backlog_flag", "sum"),
            course_attendance_average=("attendance_pct_course", "mean"),
            course_assignment_average=("assignment_score_100", "mean"),
        )

        lms = self.raw["lms_weekly_activity.csv"].copy()
        lms["week_start"] = pd.to_datetime(lms["week_start"], errors="coerce")
        for col in ["assignments_due", "assignments_submitted", "assignments_on_time", "lms_activity_index", "login_days", "active_minutes"]:
            lms[col] = pd.to_numeric(lms[col], errors="coerce")
        due_by_student = lms.groupby("Student_ID").agg(
            lms_weeks=("week_start", "nunique"),
            lms_average_activity=("lms_activity_index", "mean"),
            lms_average_login_days=("login_days", "mean"),
            lms_average_active_minutes=("active_minutes", "mean"),
            lms_assignments_due=("assignments_due", "sum"),
            lms_assignments_submitted=("assignments_submitted", "sum"),
            lms_assignments_on_time=("assignments_on_time", "sum"),
        ).reset_index()
        due_by_student["assignment_submission_rate"] = np.where(
            due_by_student["lms_assignments_due"] > 0,
            100 * due_by_student["lms_assignments_submitted"] / due_by_student["lms_assignments_due"], np.nan,
        )
        due_by_student["assignment_on_time_rate"] = np.where(
            due_by_student["lms_assignments_due"] > 0,
            100 * due_by_student["lms_assignments_on_time"] / due_by_student["lms_assignments_due"], np.nan,
        )
        # Compare each student's first and last four observed weeks, not unrelated raw rows.
        lms_sorted = lms.sort_values(["Student_ID", "week_start"])
        lms_trends = []
        for sid, group in lms_sorted.groupby("Student_ID", sort=False):
            vals = pd.to_numeric(group["lms_activity_index"], errors="coerce").dropna().to_numpy()
            if len(vals) >= 2:
                split = max(1, len(vals) // 2)
                early = float(np.mean(vals[:split]))
                late = float(np.mean(vals[split:])) if split < len(vals) else early
                change = late - early
            else:
                early = late = change = np.nan
            lms_trends.append({"Student_ID": sid, "lms_early_activity": early, "lms_recent_activity": late, "lms_activity_change": change})
        due_by_student = due_by_student.merge(pd.DataFrame(lms_trends), on="Student_ID", how="left", validate="one_to_one")

        engagement = self.raw["engagement_summary.csv"].copy()
        engagement = engagement.drop_duplicates("Student_ID", keep="last")
        feedback = self.raw["student_feedback.csv"].copy()
        feedback["support_requested_flag"] = feedback["support_requested"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"]).astype(int)
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

        merged = master
        for right, name in [(academic_agg, "academic"), (due_by_student, "lms"), (engagement, "engagement"), (feedback_agg, "feedback"), (prep, "placement_preparation")]:
            merged = merged.merge(right, on="Student_ID", how="left", validate="one_to_one", suffixes=("", f"_{name}"))
        merged["data_completeness_pct"] = self._row_completeness(merged)
        return merged

    @staticmethod
    def _row_completeness(df: pd.DataFrame) -> pd.Series:
        important = [
            "CGPA", "Attendance_Percentage", "average_course_marks", "backlog_count",
            "lms_average_activity", "assignment_on_time_rate", "Aptitude_Score",
            "mock_coding_score_100", "resume_review_score_100", "satisfaction_average",
        ]
        present_cols = [c for c in important if c in df.columns]
        if not present_cols:
            return pd.Series(0.0, index=df.index)
        return (df[present_cols].notna().sum(axis=1) / len(present_cols) * 100).round(1)

    def _train_placement_model(self) -> tuple[Pipeline | None, dict[str, Any]]:
        df = self.raw["student_master.csv"].copy()
        # Outcomes are labels only, never predictors. Demographic columns are excluded as risk drivers.
        target_col = "Placement_Status"
        numeric_features = [
            "CGPA", "Attendance_Percentage", "DSA_Problems_Solved", "Aptitude_Score",
            "Communication_Score", "Projects_Count", "Internships_Count", "Certifications_Count",
            "Open_Source_Contributions", "GitHub_Contributions", "LinkedIn_Activity_Score",
            "Hackathons_Participated", "Competitive_Programming_Rating", "Study_Hours_Per_Week",
            "Soft_Skills_Score", "Extracurricular_Activities",
        ]
        categorical_features = [c for c in ["Branch", "College_Tier", "Leadership_Experience"] if c in df.columns]
        numeric_features = [c for c in numeric_features if c in df.columns]
        if target_col not in df or df[target_col].nunique() < 2:
            return None, {"available": False, "reason": "A usable labelled Placement_Status column is not available."}
        X = df[numeric_features + categorical_features].copy()
        y = df[target_col].astype(str).str.strip()
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        prep = ColumnTransformer([("numeric", num_pipe, numeric_features), ("categorical", cat_pipe, categorical_features)])
        model = Pipeline([("preprocess", prep), ("classifier", RandomForestClassifier(
            n_estimators=160, max_depth=12, min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ))])
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            classes = list(model.named_steps["classifier"].classes_)
            metrics: dict[str, Any] = {
                "available": True,
                "target": "Placement_Status",
                "positive_class": "Not Placed",
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
                "precision_not_placed": round(float(precision_score(y_test, predicted, pos_label="Not Placed", zero_division=0)), 4),
                "recall_not_placed": round(float(recall_score(y_test, predicted, pos_label="Not Placed", zero_division=0)), 4),
                "f1_not_placed": round(float(f1_score(y_test, predicted, pos_label="Not Placed", zero_division=0)), 4),
                "provenance_warning": "Metrics are only a technical demonstration on synthetic labels; they are not evidence of real-world placement prediction accuracy.",
            }
            if "Not Placed" in classes and len(classes) == 2:
                probabilities = model.predict_proba(X_test)[:, classes.index("Not Placed")]
                truth = (y_test == "Not Placed").astype(int)
                metrics["roc_auc_not_placed"] = round(float(roc_auc_score(truth, probabilities)), 4)
            return model, metrics
        except Exception as exc:  # Keep the dashboard available even if the optional classifier cannot train.
            return None, {"available": False, "reason": f"Placement model could not be trained: {type(exc).__name__}: {exc}"}

    def _apply_scores_and_segments(self) -> None:
        df = self.students
        cgpa_score = (pd.to_numeric(df["CGPA"], errors="coerce") * 10).clip(0, 100)
        marks_score = pd.to_numeric(df["average_course_marks"], errors="coerce")
        backlog_count = pd.to_numeric(df["backlog_count"], errors="coerce").fillna(0)
        backlog_score = (100 - 20 * backlog_count).clip(0, 100)
        academics_source = pd.DataFrame({"cgpa_component": cgpa_score, "marks_component": marks_score, "backlog_component": backlog_score}, index=df.index)
        df["academic_score"] = _weighted_score(academics_source, {"cgpa_component": 0.45, "marks_component": 0.35, "backlog_component": 0.20}).round(1)

        attendance = pd.to_numeric(df["Attendance_Percentage"], errors="coerce")
        activity = pd.to_numeric(df["lms_average_activity"], errors="coerce")
        on_time = pd.to_numeric(df["assignment_on_time_rate"], errors="coerce")
        engagement_source = pd.DataFrame({"attendance_component": attendance, "ontime_component": on_time, "activity_component": activity}, index=df.index)
        df["engagement_score"] = _weighted_score(engagement_source, {"attendance_component": 0.40, "ontime_component": 0.35, "activity_component": 0.25}).round(1)

        # If no mock interview was completed, the stored interview score is not treated as an observed assessment.
        prep_count = pd.to_numeric(df["mock_interviews_completed"], errors="coerce")
        interview_score = pd.to_numeric(df["mock_interview_score_100"], errors="coerce").where(prep_count.fillna(0) > 0)
        placement_source = pd.DataFrame({
            "aptitude_component": pd.to_numeric(df["Aptitude_Score"], errors="coerce"),
            "coding_component": pd.to_numeric(df["mock_coding_score_100"], errors="coerce"),
            "interview_component": interview_score,
            "resume_component": pd.to_numeric(df["resume_review_score_100"], errors="coerce"),
            "group_discussion_component": pd.to_numeric(df["group_discussion_score_100"], errors="coerce"),
            "soft_skills_component": pd.to_numeric(df["Soft_Skills_Score"], errors="coerce"),
        }, index=df.index)
        df["placement_readiness_score"] = _weighted_score(placement_source, {
            "aptitude_component": 0.25, "coding_component": 0.25, "interview_component": 0.20,
            "resume_component": 0.10, "group_discussion_component": 0.10, "soft_skills_component": 0.10,
        }).round(1)
        df["student_success_score"] = (0.50 * df["academic_score"] + 0.20 * df["engagement_score"] + 0.30 * df["placement_readiness_score"]).round(1)

        # Transparent academic risk rules; risk flags trigger review, not a diagnosis.
        df["academic_risk_level"] = "Low"
        high_acad = (df["academic_score"] < 55) | ((backlog_count >= 2) & (pd.to_numeric(df["CGPA"], errors="coerce") < 7))
        moderate_acad = (df["academic_score"] < 70) | (attendance < 65) | (on_time < 50) | (pd.to_numeric(df["lms_activity_change"], errors="coerce") <= -20)
        df.loc[moderate_acad, "academic_risk_level"] = "Moderate"
        df.loc[high_acad, "academic_risk_level"] = "High"

        df["placement_risk_probability"] = np.nan
        df["placement_risk_level"] = "Unavailable"
        if self.model is not None:
            src = self.raw["student_master.csv"].copy()
            numeric_features = [
                "CGPA", "Attendance_Percentage", "DSA_Problems_Solved", "Aptitude_Score", "Communication_Score",
                "Projects_Count", "Internships_Count", "Certifications_Count", "Open_Source_Contributions",
                "GitHub_Contributions", "LinkedIn_Activity_Score", "Hackathons_Participated",
                "Competitive_Programming_Rating", "Study_Hours_Per_Week", "Soft_Skills_Score", "Extracurricular_Activities",
            ]
            categorical_features = [c for c in ["Branch", "College_Tier", "Leadership_Experience"] if c in src.columns]
            x = src[[c for c in numeric_features + categorical_features if c in src.columns]].copy()
            classes = list(self.model.named_steps["classifier"].classes_)
            if "Not Placed" in classes:
                probs = self.model.predict_proba(x)[:, classes.index("Not Placed")]
                df["placement_risk_probability"] = probs * 100
                df["placement_risk_level"] = np.select([probs >= 0.65, probs >= 0.40], ["High", "Moderate"], default="Low")

        # Explainable rules for actionable, student-level review.
        segments = []
        reasons = []
        for row in df.itertuples(index=False):
            r = []
            if _safe_float(getattr(row, "CGPA", np.nan), np.nan) < 6.5:
                r.append("CGPA below 6.5")
            if _safe_float(getattr(row, "Attendance_Percentage", np.nan), np.nan) < 65:
                r.append("Attendance below 65%")
            if _safe_float(getattr(row, "average_course_marks", np.nan), np.nan) < 55:
                r.append("Average course marks below 55/100")
            if _safe_float(getattr(row, "backlog_count", 0)) >= 1:
                r.append(f"{int(_safe_float(getattr(row, 'backlog_count', 0)))} course backlog(s)")
            if _safe_float(getattr(row, "lms_activity_change", np.nan), np.nan) <= -20:
                r.append("LMS activity dropped by at least 20 points")
            if _safe_float(getattr(row, "mock_coding_score_100", np.nan), np.nan) < 50:
                r.append("Mock coding score below 50/100")
            if _safe_float(getattr(row, "Aptitude_Score", np.nan), np.nan) < 35:
                r.append("Aptitude score below 35/100")
            if int(_safe_float(getattr(row, "support_requested", 0))) == 1:
                r.append("Student feedback records a support request")
            reasons.append(r[:6])

            a = _safe_float(getattr(row, "academic_score", np.nan), np.nan)
            p = _safe_float(getattr(row, "placement_readiness_score", np.nan), np.nan)
            e = _safe_float(getattr(row, "engagement_score", np.nan), np.nan)
            ar = getattr(row, "academic_risk_level", "Low")
            pr = getattr(row, "placement_risk_level", "Unavailable")
            if np.isfinite(a) and a >= 70 and np.isfinite(p) and p < 55:
                label = "Strong Academics · Low Placement Readiness"
            elif np.isfinite(a) and a < 60 and np.isfinite(e) and e < 60:
                label = "Academic & Engagement Support"
            elif ar == "High" or pr == "High" or (np.isfinite(e) and e < 45):
                label = "Priority Review"
            elif np.isfinite(a) and a >= 70 and np.isfinite(p) and p >= 70:
                label = "Balanced / Placement Ready"
            elif int(_safe_float(getattr(row, "support_requested", 0))) == 1:
                label = "Student-Requested Support"
            else:
                label = "Mixed Signals · Monitor"
            segments.append(label)
        df["explanation_drivers"] = reasons
        df["segment"] = segments

    def data_quality(self) -> dict[str, Any]:
        student_ids = set(self.raw["student_master.csv"]["Student_ID"].astype(str).str.strip())
        source_rows = []
        join_results = []
        for filename, frame in self.raw.items():
            primary_key = PRIMARY_KEYS.get(filename)
            duplicates = int(frame[primary_key].duplicated().sum()) if primary_key and primary_key in frame.columns else 0
            student_key = "Student_ID" if "Student_ID" in frame.columns else None
            orphans = int((~frame[student_key].astype(str).str.strip().isin(student_ids)).sum()) if student_key and filename != "student_master.csv" else 0
            missing_required = [c for c in SOURCES[filename] if c in frame and frame[c].isna().any()]
            source_rows.append({
                "source": filename, "row_count": int(len(frame)), "column_count": int(len(frame.columns)),
                "duplicate_key_rows": duplicates, "orphan_student_rows": orphans,
                "required_columns_with_missing_values": missing_required,
                "status": "Check" if duplicates or orphans or missing_required else "OK",
            })
            if student_key and filename != "student_master.csv":
                present_ids = set(frame[student_key].astype(str).str.strip())
                join_results.append({
                    "source": filename,
                    "unique_student_ids": len(present_ids),
                    "matched_student_ids": len(present_ids & student_ids),
                    "orphan_student_ids": len(present_ids - student_ids),
                    "coverage_pct_of_master": round(100 * len(present_ids & student_ids) / max(1, len(student_ids)), 1),
                })
        master = self.raw["student_master.csv"]
        feedback_ids = set(self.raw["student_feedback.csv"]["Student_ID"].astype(str).str.strip())
        all_ids = set(master["Student_ID"].astype(str).str.strip())
        return {
            "provenance": "All records in this hackathon data bundle are synthetic demonstration data.",
            "student_count": int(len(all_ids)),
            "duplicate_master_ids": int(master["Student_ID"].duplicated().sum()),
            "students_with_feedback": len(feedback_ids & all_ids),
            "students_without_feedback": len(all_ids - feedback_ids),
            "sources": source_rows,
            "join_coverage": join_results,
            "notes": [
                "No raw one-to-many table is joined directly into the final student view; records are aggregated first.",
                "Placement_Status and Package_LPA are outcomes and are excluded from model predictors.",
                "Age, Gender and State are excluded from individual success scoring and risk predictors.",
                "Missing feedback is treated as no response, not as negative feedback.",
            ],
        }

    def overview(self) -> dict[str, Any]:
        df = self.students
        branch = df.groupby("Branch", dropna=False).agg(
            students=("Student_ID", "nunique"), average_success_score=("student_success_score", "mean"),
            academic_risk_count=("academic_risk_level", lambda x: int((x == "High").sum())),
            placement_risk_count=("placement_risk_level", lambda x: int((x == "High").sum())),
        ).reset_index().sort_values("students", ascending=False)
        lms = self.raw["lms_weekly_activity.csv"].copy()
        lms["week_start"] = pd.to_datetime(lms["week_start"], errors="coerce")
        trend = lms.groupby("week_start").agg(
            average_activity=("lms_activity_index", "mean"),
            assignments_due=("assignments_due", "sum"),
            assignments_submitted=("assignments_submitted", "sum"),
            assignments_on_time=("assignments_on_time", "sum"),
        ).reset_index().sort_values("week_start")
        trend["on_time_rate"] = np.where(trend["assignments_due"] > 0, 100 * trend["assignments_on_time"] / trend["assignments_due"], np.nan)
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
            "support_requests": int(pd.to_numeric(df.get("support_requested", 0), errors="coerce").fillna(0).sum()),
            "average_data_completeness_pct": round(float(df["data_completeness_pct"].mean()), 1),
            "model_metrics": self.model_metrics,
            "branches": [{
                "branch": str(r.Branch), "students": int(r.students),
                "average_success_score": round(float(r.average_success_score), 1),
                "academic_risk_count": int(r.academic_risk_count), "placement_risk_count": int(r.placement_risk_count),
            } for r in branch.itertuples(index=False)],
            "weekly_trend": [{
                "week": r.week_start.strftime("%d %b") if pd.notna(r.week_start) else "Unknown",
                "average_activity": round(float(r.average_activity), 1),
                "on_time_rate": round(float(r.on_time_rate), 1) if pd.notna(r.on_time_rate) else None,
            } for r in trend.itertuples(index=False)],
        }

    def segments_summary(self) -> list[dict[str, Any]]:
        grouped = self.students.groupby("segment").agg(
            student_count=("Student_ID", "nunique"), average_success_score=("student_success_score", "mean"),
            average_academic_score=("academic_score", "mean"), average_placement_score=("placement_readiness_score", "mean"),
            high_academic_risk=("academic_risk_level", lambda x: int((x == "High").sum())),
            high_placement_risk=("placement_risk_level", lambda x: int((x == "High").sum())),
        ).reset_index().sort_values("student_count", ascending=False)
        return [{
            "segment": r.segment, "student_count": int(r.student_count),
            "average_success_score": round(float(r.average_success_score), 1),
            "average_academic_score": round(float(r.average_academic_score), 1),
            "average_placement_score": round(float(r.average_placement_score), 1),
            "high_academic_risk": int(r.high_academic_risk), "high_placement_risk": int(r.high_placement_risk),
        } for r in grouped.itertuples(index=False)]

    def student_rows(self, search: str = "", branch: str = "All", segment: str = "All", risk: str = "All", sort_by: str = "student_success_score", sort_dir: str = "asc", page: int = 1, page_size: int = 25) -> dict[str, Any]:
        df = self.students.copy()
        if search.strip():
            needle = search.strip().lower()
            df = df[df["Student_ID"].astype(str).str.lower().str.contains(needle, regex=False)]
        if branch and branch != "All":
            df = df[df["Branch"] == branch]
        if segment and segment != "All":
            df = df[df["segment"] == segment]
        if risk == "Academic High":
            df = df[df["academic_risk_level"] == "High"]
        elif risk == "Academic Moderate":
            df = df[df["academic_risk_level"] == "Moderate"]
        elif risk == "Placement High":
            df = df[df["placement_risk_level"] == "High"]
        elif risk == "Placement Moderate":
            df = df[df["placement_risk_level"] == "Moderate"]
        elif risk == "Any High":
            df = df[(df["academic_risk_level"] == "High") | (df["placement_risk_level"] == "High")]
        if sort_by not in {"student_success_score", "academic_score", "engagement_score", "placement_readiness_score", "CGPA", "Branch", "placement_risk_probability"}:
            sort_by = "student_success_score"
        df = df.sort_values(sort_by, ascending=(sort_dir != "desc"), na_position="last")
        total = len(df)
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        start = (page - 1) * page_size
        subset = df.iloc[start:start + page_size]
        items = []
        for _, row in subset.iterrows():
            items.append({
                "student_id": str(row["Student_ID"]), "branch": str(row["Branch"]), "college_tier": str(row.get("College_Tier", "—")),
                "cgpa": _safe_float(row.get("CGPA")), "attendance": _safe_float(row.get("Attendance_Percentage")),
                "success_score": _safe_float(row.get("student_success_score")), "academic_score": _safe_float(row.get("academic_score")),
                "engagement_score": _safe_float(row.get("engagement_score")), "placement_readiness_score": _safe_float(row.get("placement_readiness_score")),
                "academic_risk": str(row.get("academic_risk_level", "Unavailable")), "placement_risk": str(row.get("placement_risk_level", "Unavailable")),
                "placement_risk_probability": round(_safe_float(row.get("placement_risk_probability"), np.nan), 1) if pd.notna(row.get("placement_risk_probability")) else None,
                "segment": str(row.get("segment", "Mixed Signals · Monitor")), "data_completeness_pct": _safe_float(row.get("data_completeness_pct")),
            })
        return {"total": int(total), "page": page, "page_size": page_size, "items": items}

    def student_detail(self, student_id: str) -> dict[str, Any] | None:
        matched = self.students[self.students["Student_ID"] == str(student_id)]
        if matched.empty:
            return None
        row = matched.iloc[0]
        def maybe(col: str) -> Any:
            v = row[col] if col in row.index else None
            if v is None or pd.isna(v):
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                return round(float(v), 2)
            return v.item() if isinstance(v, np.generic) else v
        sid = str(row["Student_ID"])
        lms = self.raw["lms_weekly_activity.csv"]
        lms_rows = lms[lms["Student_ID"] == sid].copy().sort_values("week_start")
        trend = [{"week": str(r.week_start), "activity_index": _safe_float(r.lms_activity_index), "on_time_rate": round(100 * _safe_float(r.assignments_on_time) / _safe_float(r.assignments_due), 1) if _safe_float(r.assignments_due) > 0 else None} for r in lms_rows.itertuples(index=False)]
        acad = self.raw["academic_course_records.csv"]
        acad_rows = acad[acad["Student_ID"] == sid]
        course_records = [{
            "course_id": str(r.course_id), "course_name": str(r.course_name), "semester": int(r.semester),
            "marks": _safe_float(r.total_marks_100), "attendance": _safe_float(r.attendance_pct_course),
            "assignment_score": _safe_float(r.assignment_score_100), "backlog": bool(int(_safe_float(r.backlog_flag))),
        } for r in acad_rows.itertuples(index=False)]
        feedback = self.raw["student_feedback.csv"]
        feedback_rows = feedback[feedback["Student_ID"] == sid]
        def value_or_none(value: Any) -> Any:
            if value is None or pd.isna(value):
                return None
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating, float)):
                return round(float(value), 2)
            return value
        feedback_records = [{
            "date": str(r.survey_date), "satisfaction": value_or_none(r.satisfaction_score_1_5),
            "faculty_clarity": value_or_none(r.faculty_clarity_1_5), "workload_manageability": value_or_none(r.workload_manageability_1_5),
            "support_requested": str(r.support_requested), "topic": str(r.feedback_topic),
        } for r in feedback_rows.itertuples(index=False)]
        drivers = list(row.get("explanation_drivers", []))
        return {
            "student_id": sid, "branch": str(row["Branch"]), "college_tier": str(row.get("College_Tier", "—")),
            "cgpa": maybe("CGPA"), "attendance": maybe("Attendance_Percentage"),
            "success_score": maybe("student_success_score"), "academic_score": maybe("academic_score"),
            "engagement_score": maybe("engagement_score"), "placement_readiness_score": maybe("placement_readiness_score"),
            "academic_risk": str(row["academic_risk_level"]), "placement_risk": str(row["placement_risk_level"]),
            "placement_risk_probability": maybe("placement_risk_probability"), "segment": str(row["segment"]),
            "data_completeness_pct": maybe("data_completeness_pct"), "explanation_drivers": drivers,
            "metrics": {
                "average_course_marks": maybe("average_course_marks"), "backlog_count": maybe("backlog_count"),
                "assignment_submission_rate": maybe("assignment_submission_rate"), "assignment_on_time_rate": maybe("assignment_on_time_rate"),
                "lms_average_activity": maybe("lms_average_activity"), "lms_activity_change": maybe("lms_activity_change"),
                "aptitude_score": maybe("Aptitude_Score"), "mock_coding_score": maybe("mock_coding_score_100"),
                "mock_interviews_completed": maybe("mock_interviews_completed"), "mock_interview_score": maybe("mock_interview_score_100") if _safe_float(row.get("mock_interviews_completed")) > 0 else None,
                "resume_review_score": maybe("resume_review_score_100"), "satisfaction_average": maybe("satisfaction_average"),
                "support_requested": bool(_safe_float(row.get("support_requested", 0))),
            },
            "weekly_lms_activity": trend, "course_records": course_records, "feedback_records": feedback_records,
            "score_method": {
                "academic": "45% CGPA normalized to 100 + 35% average course marks + 20% backlog score (100 − 20 × backlog count, floor 0).",
                "engagement": "40% attendance + 35% assignment on-time rate + 25% average LMS activity; missing components are reweighted among available inputs.",
                "placement_readiness": "25% aptitude + 25% mock coding + 20% mock interview + 10% resume review + 10% group discussion + 10% soft skills; unavailable interviews are excluded when no mock interview was completed, and remaining components are reweighted.",
                "success": "50% academic score + 20% engagement score + 30% placement-readiness score.",
                "caveat": "These are prototype rules on synthetic data, not validated institutional thresholds.",
            },
        }

    def generate_report(self, report_type: str = "risk_snapshot", branch: str = "All") -> dict[str, Any]:
        df = self.students.copy()
        if branch and branch != "All":
            df = df[df["Branch"] == branch]
        if df.empty:
            return {"title": "Student Success Report", "summary": "No records match the selected filters.", "findings": [], "scope": {"branch": branch, "students": 0}}
        high_acad = int((df["academic_risk_level"] == "High").sum())
        high_place = int((df["placement_risk_level"] == "High").sum())
        weak_ready = int(((df["academic_score"] >= 70) & (df["placement_readiness_score"] < 55)).sum())
        low_trend = int((pd.to_numeric(df["lms_activity_change"], errors="coerce") <= -20).sum())
        findings = [
            {"title": "Academic review queue", "metric": high_acad, "unit": "students", "detail": "Students flagged High by the transparent academic review rules. Review underlying records before acting."},
            {"title": "Placement readiness review", "metric": high_place, "unit": "students", "detail": "Students with a high synthetic-model estimate of Not Placed. This is a prioritization signal, not a guarantee."},
            {"title": "Academic-to-placement gap", "metric": weak_ready, "unit": "students", "detail": "Students with an academic score of at least 70 and placement-readiness below 55."},
            {"title": "Declining LMS activity", "metric": low_trend, "unit": "students", "detail": "Students whose last-half average LMS activity is at least 20 points below the first half."},
        ]
        title = "Student Success Risk Snapshot" if report_type == "risk_snapshot" else "Student Success Institutional Brief"
        return {
            "title": title,
            "summary": f"Analyzed {len(df):,} synthetic student profiles for {branch if branch != 'All' else 'all branches'}. The findings identify groups for review; they do not establish causes or predict individual outcomes with validated real-world accuracy.",
            "scope": {"branch": branch, "students": int(len(df)), "synthetic_data": True},
            "mean_scores": {"success": round(float(df["student_success_score"].mean()), 1), "academic": round(float(df["academic_score"].mean()), 1), "engagement": round(float(df["engagement_score"].mean()), 1), "placement_readiness": round(float(df["placement_readiness_score"].mean()), 1)},
            "findings": findings,
            "recommended_checks": [
                "Review the source records and the student's context before deciding on an intervention.",
                "For a course-level issue, compare the same course and semester across cohorts before inferring a bottleneck.",
                "Track follow-up outcomes only when actual intervention delivery and follow-up records are available.",
            ],
            "limitations": ["The dataset is synthetic.", "The prototype score thresholds are heuristics, not validated institutional policy.", "Descriptive associations should not be interpreted as causes."],
        }
