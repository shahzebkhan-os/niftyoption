import logging
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, BigInteger, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed, before_log
from src.config.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

class OptionChainSnapshot(Base):
    __tablename__ = 'option_chain_snapshots'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    expiry = Column(DateTime, nullable=False, index=True)
    strike = Column(Float, nullable=False)
    option_type = Column(String(2), nullable=False)  # 'CE' or 'PE'
    
    # Pricing & Liquidity
    ltp = Column(Float)
    volume = Column(BigInteger)
    oi = Column(BigInteger)
    oi_change = Column(BigInteger)
    bid_price = Column(Float)
    ask_price = Column(Float)
    bid_qty = Column(Integer)
    ask_qty = Column(Integer)
    
    # Greeks & Volatility
    iv = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    
    # Underlying Info (denormalized for query speed)
    symbol = Column(String(20), index=True)
    underlying_price = Column(Float)

    __table_args__ = (
        Index('idx_timestamp_expiry_strike', 'timestamp', 'expiry', 'strike'),
        UniqueConstraint('timestamp', 'expiry', 'strike', 'option_type', 'symbol', name='idx_unique_snapshot'),
    )

class RawDataLog(Base):
    __tablename__ = 'raw_data_logs'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    endpoint = Column(String)
    raw_data = Column(JSON)

# Robust Engine Creation
@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), before=before_log(logger, logging.INFO))
def get_engine():
    """
    Creates SQL Alchemy engine with retry logic.
    """
    return create_engine(
        settings.database_url,
        pool_size=20,
        max_overflow=0,
        pool_pre_ping=True
    )

_engine = None

def init_db():
    """Initializes the database tables."""
    global _engine
    try:
        if _engine is None:
            _engine = get_engine()
        Base.metadata.create_all(_engine)
        logger.info("Database tables created/verified successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}")
        raise

def get_session():
    """Returns a new database session."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    Session = sessionmaker(bind=_engine)
    return Session()

# Alias for compatibility
SessionLocal = sessionmaker(autocommit=False, autoflush=False)
