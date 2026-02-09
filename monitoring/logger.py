import logging
import logging.handlers
import os
from datetime import datetime
from core.config import MarketConfig  # LogConfig removed - use MarketConfig or define LogConfig

def setup_logger(config: LogConfig) -> logging.Logger:
    """Configure le système de logging."""
    logger = logging.getLogger('market_maker')
    logger.setLevel(config.log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # File handler
    if config.log_file:
        os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=config.log_file,
            when='D',  # Daily rotation
            interval=1,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger