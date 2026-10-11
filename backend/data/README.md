# ATLAS bundled demo data

The active dashboard reads the CSV source tables in this folder. This bundle is **synthetic**, with 20,000 fictional student profiles. It is intended for software development and demonstrations only; it is not genuine institutional or KPMG data.

## Active source tables

- `student_master.csv` — pseudonymous student/program context.
- `academic_course_records.csv` — 300,000 course-level records over semesters 5, 6 and 7.
- `course_catalog.csv` — course lookup/reference records.
- `lms_weekly_activity.csv` — observed weekly activity with some missing weeks and fields.
- `engagement_summary.csv` — campus and extracurricular activity; some records/fields are missing.
- `placement_preparation_activity.csv` — aptitude, coding, communication, interview and resume indicators.
- `student_feedback.csv` — optional, sometimes repeated survey responses. No response does not imply dissatisfaction.
- `placement_outcomes.csv` — synthetic placement outcomes, used as the target in the bundled placement outlook experiment and never as an input feature.

## How the application uses it

The backend aggregates one-to-many data sources before joining them into a student-level view. It calculates an explainable Student Success Score, review flags, student segments, 16-week LMS patterns and cohort summaries. A bundled Logistic Regression placement-outlook pipeline uses only pre-outcome numeric features. Branch and demographics are excluded from model inputs; placement outcome and package/offer fields are never model predictors.

The bundled model result (about 0.775 ROC-AUC on a held-out synthetic test split) verifies only that the generated features carry learnable simulated signal. It is not a real-college performance estimate and must not guide real student decisions.

## Rebuild the placement experiment

From the project root: `python scripts/train_placement_model.py --data-dir data --output-dir outputs` (or run the included Colab notebook). Use Python dependencies listed in `backend/requirements.txt` and `resources/ATLAS_Realistic_Campus_Dataset_README.md`.
