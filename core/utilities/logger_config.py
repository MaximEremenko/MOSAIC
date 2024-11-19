# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:09:38 2024

@author: Maksim Eremenko
"""

# utilities/logger_config.py
import os
import sys
import logging
import logging.config

def setup_logging(default_path='logging.conf', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = default_path
    # Override path if environment variable is set
    value = os.getenv(env_key, None)
    if value:
        path = value

    try:
        if os.path.exists(path):
            # Load logging configuration from the file
            logging.config.fileConfig(path, disable_existing_loggers=False)
            print(f"Logging configuration loaded from {path}")
        else:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname('../tests/config/app.log')
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # Set up basic logging configuration
            logging.basicConfig(
                level=default_level,
                format='%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('../tests/config/app.log', 'a', 'utf8'),
                ]
            )
            print(f"Basic logging configuration set with level {default_level}")
    except Exception as e:
        print(f"Error in setting up logging: {e}")
        # Set up basic logging configuration as a fallback
        logging.basicConfig(level=default_level)

