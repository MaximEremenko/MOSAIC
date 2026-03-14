# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:09:38 2024

@author: Maksim Eremenko
"""

# utilities/logger_config.py
import logging
import logging.config
import os
import sys
from pathlib import Path


logger = logging.getLogger(__name__)

def setup_logging(default_path='logging.conf', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    repo_root = Path(__file__).resolve().parents[2]
    path = Path(default_path)
    # Override path if environment variable is set
    value = os.getenv(env_key, None)
    if value:
        path = Path(value)
    elif not path.is_absolute():
        candidates = [Path.cwd() / path, repo_root / path]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    log_file = os.getenv("MOSAIC_LOG_FILE")

    try:
        if path.exists():
            # Load logging configuration from the file
            logging.config.fileConfig(path, disable_existing_loggers=False)
            logging.getLogger(__name__).info("Logging configuration loaded from %s", path)
        else:
            handlers = [logging.StreamHandler(sys.stdout)]
            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                handlers.append(logging.FileHandler(log_file, "a", "utf8"))
            logging.basicConfig(
                level=default_level,
                format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
                handlers=handlers,
            )
            logging.getLogger(__name__).info(
                "Basic logging configuration set with level %s", default_level
            )
    except Exception:
        logging.basicConfig(level=default_level)
        logging.getLogger(__name__).exception("Error in setting up logging")
        # Set up basic logging configuration as a fallback

