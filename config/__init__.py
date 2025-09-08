from config.general import Settings
from config.logging import logging

settings = Settings()

logger = logging.getLogger(settings.APP_NAME)