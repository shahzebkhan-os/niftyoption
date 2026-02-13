import logging
import asyncio
from sqlalchemy import text
from src.config.settings import settings
from src.data.database import init_db, get_session
from src.strategy.telegram_bot import TelegramManager
from src.utils.logger import setup_logging

logger = setup_logging()

async def verify_environment():
    logger.info("--- Environment Verification ---")
    logger.info(f"Symbols: {settings.SYMBOLS}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    
    if settings.GROWW_JWT_TOKEN and len(settings.GROWW_JWT_TOKEN) > 50:
        logger.info("✅ Groww JWT Token: Found (Valid Length)")
    else:
        logger.error("❌ Groww JWT Token: Missing or too short")

    if settings.GROWW_SECRET:
        logger.info("✅ Groww Secret: Found")
    else:
        logger.error("❌ Groww Secret: Missing")

async def verify_database():
    logger.info("--- Database Verification ---")
    try:
        init_db()
        session = get_session()
        # Simple query to check connection
        session.execute(text("SELECT 1"))
        session.close()
        logger.info("✅ Database Connectivity: SUCCESS")
    except Exception as e:
        logger.error(f"❌ Database Connectivity: FAILED - {e}")

async def verify_telegram():
    logger.info("--- Telegram Verification ---")
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("⚠️ Telegram Bot Token: Missing - Alerts will be log-only.")
        return

    if not settings.TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram Chat ID: Missing - Cannot send live messages. Please provide CHAT_ID in .env")
        return

    logger.info(f"Attempting to send test message to Chat ID: {settings.TELEGRAM_CHAT_ID}...")
    tm = TelegramManager(token=settings.TELEGRAM_BOT_TOKEN, chat_id=settings.TELEGRAM_CHAT_ID)
    
    test_data = {
        'regime': 'TEST_MODE',
        'symbol': 'VERIFY',
        'action': 'TEST_ALERT',
        'strike': 0,
        'confidence': 100.0,
        'ev': 0.0,
        'risk_level': 'NONE'
    }
    
    try:
        await tm.send_alert(test_data)
        logger.info("✅ Telegram Alert: SENT SUCCESSFULLY. Check your Telegram app.")
    except Exception as e:
        logger.error(f"❌ Telegram Alert: FAILED - {e}")

async def main():
    await verify_environment()
    await verify_database()
    await verify_telegram()
    logger.info("--- Verification Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
