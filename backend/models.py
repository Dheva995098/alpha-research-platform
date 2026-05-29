"""
SQLAlchemy models for Alpha Research Platform.
Tables: Account, Simulation, Result, AlphaRegistry, LeaderboardAlpha, MLModelRecord
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON, inspect, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import settings
from .utils.time import utc_now

Base = declarative_base()


class Account(Base):
    """User BRAIN account with encrypted credentials."""
    __tablename__ = "accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # Placeholder for future multi-user support
    brain_email = Column(String, unique=True, index=True)
    brain_password_encrypted = Column(String)  # AES-256 encrypted
    daily_quota = Column(Integer, default=450)
    submissions_today = Column(Integer, default=0)
    last_quota_reset = Column(DateTime, default=utc_now)
    is_active = Column(Boolean, default=True)
    worker_enabled = Column(Boolean, default=True)
    max_running = Column(Integer, default=6)
    max_pending = Column(Integer, default=15)
    cooldown_until = Column(DateTime, nullable=True)
    last_worker_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    
    # Relationships
    simulations = relationship("Simulation", back_populates="account", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="account", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Account(id={self.id}, email={self.brain_email})>"


class Simulation(Base):
    """Alpha expression simulation record."""
    __tablename__ = "simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), index=True)
    brain_simulation_id = Column(String, unique=True, index=True)  # BRAIN API simulation ID
    expression_signature = Column(String, index=True, nullable=True)
    expression = Column(Text)  # FASTEXPR format
    settings = Column(JSON, nullable=True)  # BRAIN simulation settings used for this expression
    status = Column(String, default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)  # 0-100
    error_message = Column(String, nullable=True)
    submitted_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    account = relationship("Account", back_populates="simulations")
    result = relationship("Result", back_populates="simulation", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Simulation(id={self.id}, status={self.status})>"


class Result(Base):
    """Alpha backtest result from BRAIN."""
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), index=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), unique=True)
    brain_alpha_id = Column(String, unique=True, index=True)  # BRAIN alpha ID
    expression = Column(Text)
    sharpe = Column(Float, nullable=True)
    fitness = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)
    self_correlation = Column(Float, nullable=True)
    all_checks_passed = Column(Boolean, nullable=True)
    raw_metrics = Column(JSON, nullable=True)  # Full BRAIN response
    ml_pass_probability = Column(Float, nullable=True)  # Predicted by ranker
    final_score = Column(Float, nullable=True)  # Combined heuristic + ML score
    human_approved = Column(Boolean, default=False)
    submitted_to_brain = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    account = relationship("Account", back_populates="results")
    simulation = relationship("Simulation", back_populates="result")
    
    def __repr__(self):
        return f"<Result(id={self.id}, sharpe={self.sharpe}, fitness={self.fitness})>"


class AlphaRegistry(Base):
    """Global alpha fingerprint registry to avoid duplicate live work across accounts."""
    __tablename__ = "alpha_registry"

    id = Column(Integer, primary_key=True, index=True)
    expression_signature = Column(String, unique=True, index=True)
    expression = Column(Text)
    settings_signature = Column(String, index=True, nullable=True)
    first_account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    first_simulation_id = Column(Integer, nullable=True)
    status = Column(String, default="queued", index=True)
    source = Column(String, default="queue", index=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    def __repr__(self):
        return f"<AlphaRegistry(signature={self.expression_signature}, status={self.status})>"


class LeaderboardAlpha(Base):
    """Cached public WorldQuant leaderboard alphas for ML training."""
    __tablename__ = "leaderboard_alphas"
    
    id = Column(Integer, primary_key=True, index=True)
    expression = Column(Text, index=True)
    sharpe = Column(Float)
    fitness = Column(Float)
    turnover = Column(Float)
    self_correlation = Column(Float)
    passes_checks = Column(Boolean)
    scraped_at = Column(DateTime, default=utc_now)
    
    def __repr__(self):
        return f"<LeaderboardAlpha(sharpe={self.sharpe}, fitness={self.fitness})>"


class DataFieldRecord(Base):
    """Persisted BRAIN data-field metadata for field discovery and generation."""
    __tablename__ = "data_fields"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    dataset_id = Column(String, index=True, nullable=True)
    category = Column(String, index=True, nullable=True)
    field_type = Column(String, index=True, nullable=True)
    region = Column(String, default="USA", index=True)
    universe = Column(String, nullable=True, index=True)
    delay = Column(Integer, default=1)
    coverage = Column(Float, nullable=True)
    alpha_count = Column(Integer, nullable=True)
    user_count = Column(Integer, nullable=True)
    value_score = Column(Float, nullable=True)
    field_score = Column(Float, default=0.0, index=True)
    description = Column(Text, nullable=True)
    raw_metadata = Column(JSON, nullable=True)
    synced_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    def __repr__(self):
        return f"<DataFieldRecord(name={self.name}, dataset={self.dataset_id}, score={self.field_score})>"


class AlertConfig(Base):
    """User alert settings for Slack/Email."""
    __tablename__ = "alert_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    slack_webhook_url = Column(String, nullable=True)
    email_address = Column(String, nullable=True)
    alert_quota_exhaustion = Column(Boolean, default=True)
    alert_submission_failure = Column(Boolean, default=True)
    alert_ml_drift = Column(Boolean, default=True)
    alert_generation_anomaly = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)


class MLModelRecord(Base):
    """Persisted lightweight ML ranker metadata and weights."""
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, default="alpha_ranker")
    version = Column(String, default="baseline")
    model_type = Column(String, default="logistic")
    feature_names = Column(JSON)
    weights = Column(JSON)
    bias = Column(Float, default=0.0)
    metrics = Column(JSON, nullable=True)
    trained_on_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


# Database setup
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 30} if "sqlite" in settings.database_url else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    _ensure_lightweight_migrations()


def _ensure_lightweight_migrations():
    """Apply tiny SQLite-compatible schema additions for local development."""
    if "sqlite" not in settings.database_url:
        return
    with engine.begin() as connection:
        inspector = inspect(connection)
        table_names = set(inspector.get_table_names())
        if "simulations" in table_names:
            simulation_columns = {column["name"] for column in inspector.get_columns("simulations")}
            if "settings" not in simulation_columns:
                connection.execute(text("ALTER TABLE simulations ADD COLUMN settings JSON"))
            if "expression_signature" not in simulation_columns:
                connection.execute(text("ALTER TABLE simulations ADD COLUMN expression_signature VARCHAR"))
        if "accounts" in table_names:
            account_columns = {column["name"] for column in inspector.get_columns("accounts")}
            account_migrations = {
                "worker_enabled": "ALTER TABLE accounts ADD COLUMN worker_enabled BOOLEAN DEFAULT 1",
                "max_running": "ALTER TABLE accounts ADD COLUMN max_running INTEGER DEFAULT 6",
                "max_pending": "ALTER TABLE accounts ADD COLUMN max_pending INTEGER DEFAULT 15",
                "cooldown_until": "ALTER TABLE accounts ADD COLUMN cooldown_until DATETIME",
                "last_worker_error": "ALTER TABLE accounts ADD COLUMN last_worker_error TEXT",
            }
            for column_name, statement in account_migrations.items():
                if column_name not in account_columns:
                    connection.execute(text(statement))


def get_db():
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
