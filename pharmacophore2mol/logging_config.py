"""
Logging configuration for Pharmacophore2Mol CLI.

This module provides a centralized logging setup that works well with tqdm progress bars.
"""

import logging
import sys
from typing import Optional


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that uses tqdm.write() to avoid breaking progress bars.
    
    This ensures that log messages don't interfere with tqdm progress bar output.
    """
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            # Fallback to stderr if tqdm is not available
            self.handleError(record)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure logging for the CLI application.
    
    Args:
        verbose: If True, set logging level to DEBUG. Otherwise INFO.
        quiet: If True, suppress all output except errors.
    
    This function sets up:
    - Root logger configuration
    - Custom TqdmLoggingHandler to work with tqdm progress bars
    - Appropriate logging levels based on verbosity
    - Clean formatting for user-facing messages
    """
    # Determine log level
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Create formatter
    # For user-facing CLI, keep format simple and clean
    if verbose:
        # In verbose mode, show more details
        formatter = logging.Formatter(
            fmt='%(levelname)s - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        # In normal mode, just show the message
        formatter = logging.Formatter(
            fmt='%(message)s'
        )
    
    # Create and configure handler
    handler = TqdmLoggingHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add our custom handler
    root_logger.addHandler(handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('rdkit').setLevel(logging.WARNING)
    
    # Suppress RDKit C++ warnings about UFF
    # try:
    #     from rdkit import RDLogger
    #     RDLogger.DisableLog('rdApp.*')
    # except Exception:
    #     pass


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
