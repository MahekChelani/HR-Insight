"""
Database session and SQLite engine initialization with SQLAlchemy ORM.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "hr_insight.db")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        raise RuntimeError("DATABASE_URL must be configured in production.")
    DATABASE_URL = f"sqlite:///{DB_PATH}"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

is_sqlite = DATABASE_URL.startswith("sqlite:")
if os.getenv("ENVIRONMENT", "development").lower() == "production" and is_sqlite:
    raise RuntimeError("Production DATABASE_URL must use PostgreSQL; SQLite storage is not persistent on Render.")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if is_sqlite else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for obtaining database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
