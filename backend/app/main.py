"""
HR Insight - Employee Attrition Prediction & Analytics System
FastAPI Backend Application Entrypoint
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import urlsplit

from app.database.session import Base, engine, SessionLocal
from app.database.migration import migrate_legacy_sqlite_data
from app.models.db_models import ModelMetadata
from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router
from app.api.analytics import router as analytics_router
from app.api.models import router as models_router
from app.api.predictions import router as predictions_router
from app.schemas.schemas import EmployeePredictionInput, PredictionResult
from app.api.predictions import create_prediction

logger = logging.getLogger(__name__)

is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "" if is_production else
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

if is_production and not frontend_origins:
    logger.warning("FRONTEND_ORIGINS is empty; browser requests will be blocked until configured.")
if is_production and any(
    origin == "*" or urlsplit(origin).scheme != "https" or not urlsplit(origin).netloc
    for origin in frontend_origins
):
    raise RuntimeError("Production FRONTEND_ORIGINS must contain HTTPS origins, not wildcards.")
if is_production and len(os.getenv("HR_INSIGHT_ADMIN_PASSWORD", "")) < 16:
    raise RuntimeError("HR_INSIGHT_ADMIN_PASSWORD must be configured with at least 16 characters in production.")

# Ensure database tables exist immediately on import
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event to ensure SQLite tables exist and initial metadata is registered."""
    Base.metadata.create_all(bind=engine)
    if is_production and engine.dialect.name == "postgresql":
        migrate_legacy_sqlite_data()
    
    # Pre-populate ModelMetadata if empty and metadata.json is available
    db = SessionLocal()
    try:
        count = db.query(ModelMetadata).count()
        if count == 0:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            meta_path = os.path.join(base_dir, 'saved_models', 'metadata.json')
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                entry = ModelMetadata(
                    model_name=meta_data.get("model_name", "Random Forest Classifier"),
                    algorithm=meta_data.get("algorithm", "Ensemble Random Forest"),
                    training_date=meta_data.get("training_date", "2026-11-01"),
                    evaluation_metrics=meta_data.get("evaluation_metrics", {}),
                    selected_threshold=meta_data.get("selected_threshold", 0.48)
                )
                db.add(entry)
                db.commit()
    except Exception as e:
        print(f"Notice during startup metadata population: {e}")
    finally:
        db.close()
        
    yield


app = FastAPI(
    title="HR Insight API",
    description="Enterprise Human Resource Attrition Prediction & Decision-Support System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")


# Alias POST /api/predict directly to comply with Section 11 specifications
@app.post("/api/predict", response_model=PredictionResult, tags=["Predictions"])
def predict_alias(payload: EmployeePredictionInput):
    """Direct alias endpoint for POST /api/predict requested in project requirements."""
    db = SessionLocal()
    try:
        return create_prediction(payload=payload, db=db)
    finally:
        db.close()


@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify backend operational readiness."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    model_path = os.path.join(base_dir, 'saved_models', 'best_model.joblib')

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database connectivity check failed.") from exc

    return {
        "status": "healthy",
        "service": "HR Insight – Employee Attrition Prediction API",
        "version": "1.0.0",
        "dataset_present": os.path.exists(data_path),
        "model_loaded": os.path.exists(model_path),
        "database": "connected",
        "database_engine": engine.dialect.name,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=not is_production,
    )
