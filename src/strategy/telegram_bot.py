import logging
import asyncio
import os
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

class TelegramManager:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.bot = None
        
        if self.token:
            self.bot = Bot(token=self.token)
        else:
            logger.warning("Telegram Token not provided. Alerts will only be logged.")

    async def send_message(self, message: str):
        """
        Sends a text message to the configured chat.
        """
        if not self.bot or not self.chat_id:
            logger.info(f"ALERTS (Log Only): {message}")
            return

        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_alert(self, alert_data: dict):
        """
        Formats and sends a structured alert.
        """
        # Format the message
        # E.g.
        # 🚨 SIGNAL DETECTED 🚨
        # Strategy: TRENDING_UP
        # Symbol: NIFTY
        # Action: BUY CALL
        # Strike: 18500
        # Expiry: 2023-10-26
        # Confidence: 85%
        # EV: 25.5
        
        msg = (
            f"🚨 *SIGNAL DETECTED* 🚨\n"
            f"Strategy: `{alert_data.get('regime', 'N/A')}`\n"
            f"Symbol: `{alert_data.get('symbol', 'N/A')}`\n"
            f"Action: *{alert_data.get('action', 'N/A')}*\n"
            f"Strike: {alert_data.get('strike', 'N/A')}\n"
            f"Confidence: {alert_data.get('confidence', 0):.1f}%\n"
            f"EV: {alert_data.get('ev', 0):.2f}\n"
            f"Risk Level: {alert_data.get('risk_level', 'N/A')}\n"
        )
        
        await self.send_message(msg)
