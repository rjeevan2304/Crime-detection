import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

logger = logging.getLogger("AlertManager")


class AlertManager:
    """
    Non-blocking, thread-safe alerting service with debounce cooldowns.
    Avoids stalling video streams during network calls to Twilio.
    """

    def __init__(
        self,
        cooldown_seconds: int = 60,
        max_workers: int = 2
    ):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.recipient_number = os.getenv("ALERT_RECIPIENT_PHONE")
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_times: Dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AlertWorker")

        self.client = None
        if self.account_sid and self.auth_token and not self.account_sid.startswith("Enter"):
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
            except Exception as e:
                logger.warning(f"Could not initialize Twilio client: {e}")

    def is_in_cooldown(self, crime_class: str) -> bool:
        last_time = self.last_alert_times.get(crime_class, 0.0)
        return (time.time() - last_time) < self.cooldown_seconds

    def trigger_alert(
        self,
        crime_class: str,
        confidence: float,
        snapshot_path: Optional[str] = None
    ) -> bool:
        """
        Dispatches alert asynchronously if cooldown has passed.
        Returns True if alert was queued, False if suppressed by cooldown.
        """
        now = time.time()
        if self.is_in_cooldown(crime_class):
            logger.info(f"Alert for '{crime_class}' suppressed by cooldown.")
            return False

        self.last_alert_times[crime_class] = now
        self.executor.submit(self._send_notification, crime_class, confidence, snapshot_path)
        return True

    def _send_notification(
        self,
        crime_class: str,
        confidence: float,
        snapshot_path: Optional[str]
    ):
        msg_body = (
            f"[CRIME ALERT] Detected: {crime_class.upper()} "
            f"with {confidence * 100:.1f}% confidence. "
        )
        if snapshot_path:
            msg_body += f"Snapshot: {snapshot_path}. "
        msg_body += "Please verify immediately."

        if self.client and self.twilio_number and self.recipient_number:
            try:
                message = self.client.messages.create(
                    body=msg_body,
                    from_=self.twilio_number,
                    to=self.recipient_number
                )
                logger.info(f"SMS alert sent successfully: SID {message.sid}")
            except Exception as e:
                logger.error(f"Twilio SMS sending failed: {e}")
        else:
            # Simulated alert for dev/testing environments without Twilio credentials
            logger.warning(
                f"[SMS SIMULATION - No Twilio credentials configured]\n"
                f"To: {self.recipient_number or 'NOT SET'}\n"
                f"Body: {msg_body}"
            )

    def shutdown(self):
        self.executor.shutdown(wait=False)
