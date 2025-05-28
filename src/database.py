from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Konfigurasi database (gunakan SQLite untuk saat ini)
SQLALCHEMY_DATABASE_URL = "sqlite:///./abstract.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Basis untuk model SQLAlchemy
Base = declarative_base()