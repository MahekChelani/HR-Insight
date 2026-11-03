"""
Automated integration tests for FastAPI REST APIs, database persistence, and validation.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_api():
    """Test GET /api/health endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dataset_present"] is True
    assert data["model_loaded"] is True


def test_dashboard_summary_api():
    """Test GET /api/dashboard/summary calculates genuine metrics."""
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert data["kpis"]["total_employees"] == 1470
    assert data["kpis"]["attrition_count"] == 237
    assert data["kpis"]["attrition_rate"] == 16.12
    assert len(data["department_breakdown"]) == 3


def test_analytics_overview_api():
    """Test GET /api/analytics/overview with and without filters."""
    # Unfiltered
    res1 = client.get("/api/analytics/overview")
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["filter_summary"]["total"] == 1470

    # Filtered by Department=Sales
    res2 = client.get("/api/analytics/overview?department=Sales")
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["filter_summary"]["total"] == 446


def test_models_metrics_and_comparison_api():
    """Test model metrics and comparison endpoints."""
    res_m = client.get("/api/models/metrics")
    assert res_m.status_code == 200
    data_m = res_m.json()
    assert "Random Forest" in data_m["model_name"]
    assert "metrics" in data_m

    res_c = client.get("/api/models/comparison")
    assert res_c.status_code == 200
    data_c = res_c.json()
    assert "Decision Tree Classifier" in data_c["models"]
    assert "Random Forest Classifier" in data_c["models"]
    assert "Naive Majority Baseline" in data_c["models"]


def test_prediction_end_to_end_and_db_persistence():
    """Test POST /api/predict, check response, check DB retrieval, and delete record."""
    payload = {
        "employee_reference": "TEST-EMP-001",
        "model_choice": "random_forest",
        "Age": 29,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 700,
        "Department": "Sales",
        "DistanceFromHome": 22,
        "Education": 3,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1,
        "Gender": "Female",
        "HourlyRate": 75,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2700,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 4,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 3,
        "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0
    }

    # 1. Post prediction
    post_res = client.post("/api/predict", json=payload)
    assert post_res.status_code == 200
    pred = post_res.json()
    assert pred["predicted_class"] in ["Yes", "No"]
    assert "risk_probability" in pred
    assert pred["history_id"] is not None
    record_id = pred["history_id"]

    # 2. Retrieve history list
    list_res = client.get(f"/api/predictions?search=TEST-EMP-001")
    assert list_res.status_code == 200
    history = list_res.json()
    assert any(h["id"] == record_id for h in history)

    # 3. Retrieve individual prediction record
    get_res = client.get(f"/api/predictions/{record_id}")
    assert get_res.status_code == 200
    record = get_res.json()
    assert record["employee_reference"] == "TEST-EMP-001"
    assert record["input_features"]["Age"] == 29

    # 4. Delete prediction record
    del_res = client.delete(f"/api/predictions/{record_id}")
    assert del_res.status_code == 200

    # 5. Verify deleted
    get_after_del = client.get(f"/api/predictions/{record_id}")
    assert get_after_del.status_code == 404


def test_prediction_invalid_input_validation():
    """Verify backend rejects invalid or missing payload fields with 422 Unprocessable Entity."""
    # Missing required fields
    bad_payload = {
        "Age": 25,
        # Missing Department, OverTime, etc.
    }
    res = client.post("/api/predict", json=bad_payload)
    assert res.status_code == 422
