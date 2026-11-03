"""
Helper script to start the FastAPI application server.
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    print(f"Starting HR Insight API server on port {port} ...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=not is_production,
    )
