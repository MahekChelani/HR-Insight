"""
HR Insight - Model Performance Router
Exposes genuine machine learning evaluation metrics, confusion matrices, ROC/PR curves,
and comparative algorithm performance against the naive baseline.
"""
import os
import json
from fastapi import APIRouter, HTTPException
from ml.train import train_and_evaluate_models
from ml.predict import get_prediction_engine

router = APIRouter(prefix="/models", tags=["Model Performance"])


def _load_metrics_summary() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    summary_path = os.path.join(base_dir, 'saved_models', 'metrics_summary.json')
    if not os.path.exists(summary_path):
        # Auto-train if not yet trained
        return train_and_evaluate_models()
    with open(summary_path, 'r') as f:
        return json.load(f)


@router.get("/metrics")
def get_model_metrics():
    """Retrieve active best model's metrics, dataset splits, threshold, and top features."""
    summary = _load_metrics_summary()
    best_name = summary.get("selected_best_model", "Random Forest Classifier")
    best_details = summary.get("models", {}).get(best_name, {})
    return {
        "model_name": best_name,
        "selected_threshold": summary.get("selected_threshold", 0.48),
        "dataset_stats": summary.get("dataset_stats", {}),
        "metrics": best_details.get("metrics", {}),
        "top_features": summary.get("top_features", []),
        "imbalance_explanation": summary.get("imbalance_explanation", "")
    }


@router.get("/comparison")
def get_model_comparison():
    """Retrieve comparative analysis of Baseline, Decision Tree, and Random Forest."""
    summary = _load_metrics_summary()
    return {
        "selected_best_model": summary.get("selected_best_model"),
        "models": summary.get("models", {}),
        "top_features": summary.get("top_features", []),
        "imbalance_explanation": summary.get("imbalance_explanation", "")
    }


@router.post("/retrain")
def retrain_models():
    """Retrain ML models on the dataset and refresh the prediction engine."""
    try:
        summary = train_and_evaluate_models()
        # Force re-instantiation of prediction engine
        engine = get_prediction_engine()
        engine._load_model()
        return {
            "status": "success",
            "message": "Models successfully retrained and reloaded.",
            "best_model": summary.get("selected_best_model"),
            "threshold": summary.get("selected_threshold")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
