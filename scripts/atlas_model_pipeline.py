"""Feature engineering and a leakage-conscious placement-outlook baseline for ATLAS."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

OUTCOME_ONLY = {
    'Placement_Status', 'Offer_Count', 'Package_LPA', 'Days_To_First_Offer',
    'First_Offer_Date', 'outcome_window_start', 'outcome_window_end',
    'placement_cycle', 'outcome_source'
}
EXCLUDED_CONTEXT = {
    'Age', 'Gender', 'Home_State', 'Branch', 'Program_Year', 'Enrollment_Year',
    'Expected_Graduation_Year', 'Snapshot_Date', 'Source_System', 'College_Tier'
}


def build_feature_table(data_dir):
    """Return (X, y, outcome_frame). Uses only pre-outcome data as predictors."""
    data_dir = Path(data_dir)
    master = pd.read_csv(data_dir / 'student_master.csv')
    acad = pd.read_csv(data_dir / 'academic_course_records.csv')
    lms = pd.read_csv(data_dir / 'lms_weekly_activity.csv')
    engagement = pd.read_csv(data_dir / 'engagement_summary.csv')
    prep = pd.read_csv(data_dir / 'placement_preparation_activity.csv')
    outcomes = pd.read_csv(data_dir / 'placement_outcomes.csv')

    
    student_ids = master[['Student_ID']].drop_duplicates().set_index('Student_ID')

    
    acad['total_marks_100'] = pd.to_numeric(acad['total_marks_100'], errors='coerce')
    sem_marks = acad.pivot_table(index='Student_ID', columns='semester', values='total_marks_100', aggfunc='mean')
    sem_marks = sem_marks.rename(columns={5:'academic_sem5_mean_marks', 6:'academic_sem6_mean_marks', 7:'academic_sem7_mean_marks'})
    a = acad.groupby('Student_ID').agg(
        academic_mean_marks=('total_marks_100','mean'),
        academic_marks_sd=('total_marks_100','std'),
        academic_course_attendance_mean=('attendance_pct_course','mean'),
        academic_assignment_score_mean=('assignment_score_100','mean'),
        academic_backlog_course_count=('backlog_flag','sum'),
        academic_pass_rate=('backlog_flag', lambda x: 1 - float(np.mean(x))),
        academic_courses_observed=('record_id','count'),
    )
    a = a.join(sem_marks, how='left')
    if 'academic_sem7_mean_marks' in a and 'academic_sem5_mean_marks' in a:
        a['academic_marks_change_sem5_to_sem7'] = a['academic_sem7_mean_marks'] - a['academic_sem5_mean_marks']
    else:
        a['academic_marks_change_sem5_to_sem7'] = np.nan

    
    lms['lms_activity_index'] = pd.to_numeric(lms['lms_activity_index'], errors='coerce')
    lms['active_minutes'] = pd.to_numeric(lms['active_minutes'], errors='coerce')
    lms['resource_clicks'] = pd.to_numeric(lms['resource_clicks'], errors='coerce')
    l = lms.groupby('Student_ID').agg(
        lms_weeks_observed=('week_number','nunique'),
        lms_activity_mean=('lms_activity_index','mean'),
        lms_activity_sd=('lms_activity_index','std'),
        lms_active_minutes_mean=('active_minutes','mean'),
        lms_resource_clicks_mean=('resource_clicks','mean'),
        lms_login_days_mean=('login_days','mean'),
        lms_login_sessions_mean=('login_sessions','mean'),
        lms_assignments_due=('assignments_due','sum'),
        lms_assignments_submitted=('assignments_submitted','sum'),
        lms_assignments_on_time=('assignments_on_time','sum'),
    )
    l['lms_weeks_missing'] = 16 - l['lms_weeks_observed']
    l['lms_submission_rate'] = (l['lms_assignments_submitted'] / l['lms_assignments_due'].replace(0, np.nan)).clip(0,1)
    l['lms_on_time_rate'] = (l['lms_assignments_on_time'] / l['lms_assignments_submitted'].replace(0, np.nan)).clip(0,1)
    
    trend_src = lms.dropna(subset=['lms_activity_index']).copy()
    trend_src['week_x'] = trend_src['week_number'].astype(float)
    trend_src['xy'] = trend_src['week_x'] * trend_src['lms_activity_index']
    trend = trend_src.groupby('Student_ID').agg(
        n=('lms_activity_index','count'), sx=('week_x','sum'), sy=('lms_activity_index','sum'),
        sxx=('week_x', lambda x: float(np.square(x).sum())), sxy=('xy','sum')
    )
    denom = trend['sxx'] - trend['sx'].pow(2) / trend['n'].replace(0, np.nan)
    numer = trend['sxy'] - trend['sx'] * trend['sy'] / trend['n'].replace(0, np.nan)
    trend['lms_activity_trend_per_week'] = numer / denom.replace(0, np.nan)
    l = l.join(trend[['lms_activity_trend_per_week']], how='left')
    l = l.drop(columns=['lms_assignments_due','lms_assignments_submitted','lms_assignments_on_time'])

    
    e = engagement.drop(columns=[c for c in ['engagement_record_id','source_system','activity_period_start','activity_period_end'] if c in engagement.columns]).copy()
    e['leadership_experience'] = e['leadership_experience'].map({'Yes':1,'No':0})
    e = e.set_index('Student_ID')
    e = e.apply(pd.to_numeric, errors='coerce')
    e = e.add_prefix('engagement_')

    
    p = prep.drop(columns=[c for c in ['prep_record_id','source_system','last_assessment_date'] if c in prep.columns]).set_index('Student_ID')
    p = p.apply(pd.to_numeric, errors='coerce').add_prefix('prep_')

    features = student_ids.join(a, how='left').join(l, how='left').join(e, how='left').join(p, how='left')
    features = features.replace([np.inf,-np.inf], np.nan)
    
    excluded_prefixes = ('outcome_',)
    features = features[[c for c in features.columns if c != 'Student_ID' and c not in EXCLUDED_CONTEXT and c not in OUTCOME_ONLY and not c.startswith(excluded_prefixes)]]
    features = features.select_dtypes(include=[np.number]).astype(float)

    outcome = outcomes.set_index('Student_ID').reindex(features.index)
    y = outcome['Placement_Status'].map({'Placed':1, 'Not Placed':0})
    valid = y.notna()
    X = features.loc[valid].copy()
    y = y.loc[valid].astype(int)
    return X, y, outcome.loc[valid].copy()


def train_baseline(data_dir, output_dir=None, test_size=0.20, random_state=42):
    """Train Logistic Regression and a DummyClassifier comparator, then return metrics/model."""
    X, y, outcome = build_feature_table(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1500, solver='lbfgs')),
    ])
    dummy = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('model', DummyClassifier(strategy='prior')),
    ])
    model.fit(X_train, y_train)
    dummy.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= .5).astype(int)
    dummy_probs = dummy.predict_proba(X_test)[:,1]
    dummy_preds = dummy.predict(X_test)
    metrics = {
        'dataset_provenance':'synthetic; model scores do not establish real-world performance',
        'rows_total':int(len(X)), 'features_used':int(X.shape[1]), 'train_rows':int(len(X_train)),
        'test_rows':int(len(X_test)), 'placement_rate_overall':float(y.mean()),
        'logistic_regression':{
            'roc_auc':float(roc_auc_score(y_test, probs)),
            'pr_auc':float(average_precision_score(y_test, probs)),
            'accuracy_at_0_5':float(accuracy_score(y_test, preds)),
            'precision_at_0_5':float(precision_score(y_test, preds, zero_division=0)),
            'recall_at_0_5':float(recall_score(y_test, preds, zero_division=0)),
            'f1_at_0_5':float(f1_score(y_test, preds, zero_division=0)),
            'brier_score':float(brier_score_loss(y_test, probs)),
            'classification_report':classification_report(y_test, preds, output_dict=True, zero_division=0),
            'confusion_matrix':confusion_matrix(y_test, preds).tolist(),
        },
        'dummy_baseline':{
            'roc_auc':float(roc_auc_score(y_test, dummy_probs)),
            'accuracy':float(accuracy_score(y_test, dummy_preds)),
            'pr_auc':float(average_precision_score(y_test, dummy_probs)),
        }
    }
    
    imp = model.named_steps['imputer']
    try:
        feature_names = list(imp.get_feature_names_out(X_train.columns))
    except Exception:
        feature_names = list(X_train.columns) + [f'missing_{c}' for c in X_train.columns[imp.indicator_.features_]]
    coef = model.named_steps['model'].coef_[0]
    coef_df = pd.DataFrame({'feature':feature_names, 'standardized_coefficient':coef})
    coef_df['abs_coefficient'] = coef_df['standardized_coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_dir/'atlas_placement_logistic_pipeline.joblib')
        (output_dir/'placement_model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        coef_df.to_csv(output_dir/'placement_model_coefficients.csv', index=False)
        test_results = pd.DataFrame({'Student_ID': X_test.index, 'Actual_Placement': y_test.to_numpy(),
                                     'Estimated_Placement_Probability': probs, 'Predicted_Placed_at_0_5':preds})
        test_results.to_csv(output_dir/'placement_test_predictions.csv', index=False)
    return model, metrics, coef_df, (X_train, X_test, y_train, y_test, probs, preds)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train ATLAS placement outlook baseline on synthetic dataset.')
    parser.add_argument('--data-dir', default=str(Path(__file__).resolve().parents[1]/'data'))
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parents[1]/'outputs'))
    args = parser.parse_args()
    model, metrics, coef_df, _ = train_baseline(args.data_dir, args.output_dir)
    print(json.dumps(metrics, indent=2))
    print('\nLargest standardized model associations (not causal):')
    print(coef_df.head(12).to_string(index=False))
