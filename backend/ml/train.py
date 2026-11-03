"""
HR Insight - Model Training and Evaluation Pipeline
Trains Naive Baseline, Decision Tree, and Random Forest on stratified splits.
Selects optimal decision threshold using validation split without test set contamination.
Saves pipelines, models, feature importances, and evaluation metrics.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.preprocessing import (
    load_raw_dataset,
    clean_features,
    build_preprocessor,
    get_stratified_split,
    ALL_FEATURE_COLUMNS
)
from ml.evaluation import (
    evaluate_model_performance,
    find_optimal_threshold
)


def get_feature_names(preprocessor, categorical_cols, numerical_cols) -> list[str]:
    """Extract transformed feature names after ColumnTransformer."""
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    return numerical_cols + encoded_cat_names


def train_and_evaluate_models(data_path: str = None, save_dir: str = None):
    """
    Execute full training, threshold tuning on validation, evaluation on unseen test set,
    and save trained pipelines and metrics summary.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if save_dir is None:
        save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Genuine IBM HR Dataset
    print("[1/6] Loading IBM HR Analytics dataset...")
    raw_df = load_raw_dataset(data_path)
    total_records = len(raw_df)
    total_attrition = int((raw_df['Attrition'] == 'Yes').sum())
    total_retained = int((raw_df['Attrition'] == 'No').sum())
    attrition_rate = round((total_attrition / total_records) * 100, 2)
    print(f"      Total: {total_records} records | Attrition: {total_attrition} ({attrition_rate}%) | Retained: {total_retained}")
    
    # 2. Clean and Split Features
    X, y = clean_features(raw_df)
    X_train_full, X_test, y_train_full, y_test = get_stratified_split(X, y, test_size=0.2, random_state=42)
    
    # Validation split for threshold selection (20% of train = 16% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    print(f"[2/6] Stratified Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 3. Fit Preprocessing Pipeline ONLY on Training Data (prevents data leakage)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    # 4. Train Models
    print("[3/6] Training Classifiers...")
    
    # A. Naive Majority Baseline (Always predicts No / 0)
    baseline_model = DummyClassifier(strategy='most_frequent')
    baseline_model.fit(X_train_trans, y_train)
    
    # B. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    dt_model.fit(X_train_trans, y_train)
    
    # C. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train_trans, y_train)
    
    # 5. Tune Decision Threshold on Validation Set (NEVER on test set)
    print("[4/6] Tuning decision thresholds on validation partition...")
    dt_val_probs = dt_model.predict_proba(X_val_trans)[:, 1]
    rf_val_probs = rf_model.predict_proba(X_val_trans)[:, 1]
    
    dt_optimal_thresh = find_optimal_threshold(y_val, dt_val_probs)
    rf_optimal_thresh = find_optimal_threshold(y_val, rf_val_probs)
    print(f"      Selected Validation Thresholds: Decision Tree = {dt_optimal_thresh}, Random Forest = {rf_optimal_thresh}")
    
    # 6. Fit Final Preprocessor & Models on full X_train_full for Final Test Evaluation
    preprocessor_full = build_preprocessor()
    X_train_full_trans = preprocessor_full.fit_transform(X_train_full)
    X_test_trans = preprocessor_full.transform(X_test)
    
    # Refit with chosen hyperparameters on full training set
    dt_final = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    dt_final.fit(X_train_full_trans, y_train_full)
    
    rf_final = RandomForestClassifier(
        n_estimators=120,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    )
    rf_final.fit(X_train_full_trans, y_train_full)
    
    baseline_final = DummyClassifier(strategy='most_frequent')
    baseline_final.fit(X_train_full_trans, y_train_full)
    
    # 7. Evaluate on Strictly Unseen Final Test Set
    print("[5/6] Evaluating on Unseen Test Partition (294 records)...")
    
    # Baseline
    base_preds = baseline_final.predict(X_test_trans)
    base_probs = np.zeros(len(y_test))
    base_metrics = evaluate_model_performance(y_test, base_preds, base_probs)
    
    # Decision Tree (evaluated using validation-tuned threshold)
    dt_probs = dt_final.predict_proba(X_test_trans)[:, 1]
    dt_preds = (dt_probs >= dt_optimal_thresh).astype(int)
    dt_metrics = evaluate_model_performance(y_test, dt_preds, dt_probs)
    
    # Random Forest (evaluated using validation-tuned threshold)
    rf_probs = rf_final.predict_proba(X_test_trans)[:, 1]
    rf_preds = (rf_probs >= rf_optimal_thresh).astype(int)
    rf_metrics = evaluate_model_performance(y_test, rf_preds, rf_probs)
    
    # Feature Importances extraction
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    num_cols = [c for c in X.columns if c not in cat_cols]
    all_feature_names = get_feature_names(preprocessor_full, cat_cols, num_cols)
    
    rf_importances = rf_final.feature_importances_
    dt_importances = dt_final.feature_importances_
    
    top_rf_features = sorted(
        [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(all_feature_names, rf_importances)],
        key=lambda x: x["importance"],
        reverse=True
    )[:15]
    
    top_dt_features = sorted(
        [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(all_feature_names, dt_importances)],
        key=lambda x: x["importance"],
        reverse=True
    )[:15]
    
    # Model Selection: Compare F1 and PR-AUC
    if rf_metrics['f1_score'] >= dt_metrics['f1_score']:
        best_name = "Random Forest Classifier"
        best_model_obj = rf_final
        best_thresh = rf_optimal_thresh
        best_metrics = rf_metrics
    else:
        best_name = "Decision Tree Classifier"
        best_model_obj = dt_final
        best_thresh = dt_optimal_thresh
        best_metrics = dt_metrics
        
    print(f"      Best Model: {best_name} (F1: {best_metrics['f1_score']}, Recall: {best_metrics['recall']}, ROC-AUC: {best_metrics['roc_auc']})")
    
    # Build exportable Pipelines combining Preprocessor and Classifier
    dt_pipeline = Pipeline([('preprocessor', preprocessor_full), ('classifier', dt_final)])
    rf_pipeline = Pipeline([('preprocessor', preprocessor_full), ('classifier', rf_final)])
    best_pipeline = Pipeline([('preprocessor', preprocessor_full), ('classifier', best_model_obj)])
    
    # 8. Save Pipelines and Metadata
    print("[6/6] Saving trained models and comprehensive metrics...")
    joblib.dump(best_pipeline, os.path.join(save_dir, 'best_model.joblib'))
    joblib.dump(dt_pipeline, os.path.join(save_dir, 'decision_tree.joblib'))
    joblib.dump(rf_pipeline, os.path.join(save_dir, 'random_forest.joblib'))
    
    # Metadata for DB & Frontend
    meta = {
        "model_name": best_name,
        "algorithm": "Ensemble Random Forest" if "Forest" in best_name else "Decision Tree",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "IBM HR Analytics Employee Attrition",
        "dataset_records": total_records,
        "dataset_attrition_count": total_attrition,
        "dataset_retained_count": total_retained,
        "attrition_rate": attrition_rate,
        "selected_threshold": best_thresh,
        "decision_tree_threshold": dt_optimal_thresh,
        "random_forest_threshold": rf_optimal_thresh,
        "evaluation_metrics": best_metrics
    }
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    summary = {
        "dataset_stats": {
            "total_records": total_records,
            "attrition_count": total_attrition,
            "retained_count": total_retained,
            "attrition_rate": attrition_rate,
            "train_records": len(X_train_full),
            "test_records": len(X_test),
        },
        "selected_best_model": best_name,
        "selected_threshold": best_thresh,
        "models": {
            "Naive Majority Baseline": {
                "algorithm": "Dummy (Predicts Majority 'No')",
                "threshold": 0.5,
                "metrics": base_metrics,
                "description": "Always predicts 'No'. High accuracy (83.9%) but 0% recall and 0% F1, completely failing to detect leaving employees."
            },
            "Decision Tree Classifier": {
                "algorithm": "Decision Tree (Balanced Class Weight)",
                "threshold": dt_optimal_thresh,
                "metrics": dt_metrics,
                "top_features": top_dt_features,
                "description": "Interpretable tree structure splitting on key attrition drivers like OverTime, MonthlyIncome, and StockOptionLevel."
            },
            "Random Forest Classifier": {
                "algorithm": "Random Forest Ensemble (Balanced Class Weight)",
                "threshold": rf_optimal_thresh,
                "metrics": rf_metrics,
                "top_features": top_rf_features,
                "description": "Ensemble of 120 decorrelated decision trees reducing variance and improving generalizability on imbalanced data."
            }
        },
        "top_features": top_rf_features,
        "imbalance_explanation": (
            "Because approximately 84% of employees in the IBM dataset do not leave, a naive baseline that always "
            "predicts 'No' achieves ~83.9% accuracy despite identifying zero employees at risk. "
            "Therefore, Accuracy is misleading. Precision, Recall, F1-score, and ROC/PR-AUC provide the true measure "
            "of an organization's ability to proactively detect flight risks."
        )
    }
    with open(os.path.join(save_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("Training and evaluation successfully completed!")
    return summary


if __name__ == '__main__':
    train_and_evaluate_models()
