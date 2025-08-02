#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows CLI Compatibility Module
================================

Provides Windows-compatible CLI functions and utilities for the Schwabot system.
This module ensures consistent behavior across different platforms and handles
Windows-specific CLI quirks.
"""

import os
import sys
import logging
import codecs
from typing import Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def safe_print(*args, **kwargs) -> None:
    """
    Safe print function that handles Windows CLI encoding issues.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print
    """
    try:
        # Handle Windows encoding issues
        if sys.platform.startswith('win'):
            # Force UTF-8 encoding on Windows
            if 'encoding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
            
            # Handle Windows console color issues
            if 'colorama' in sys.modules:
                from colorama import init
                init()
        
        print(*args, **kwargs)
        
    except (UnicodeEncodeError, OSError) as e:
        # Fallback to basic print if encoding fails
        try:
            print(*args, **kwargs, file=sys.stderr)
        except:
            # Last resort: write to stderr
            sys.stderr.write(str(args) + '\n')
            sys.stderr.flush()

def safe_format_error(error: Exception) -> str:
    """
    Safely format error messages for Windows CLI.
    
    Args:
        error: Exception to format
        
    Returns:
        Formatted error string
    """
    try:
        error_str = str(error)
        if sys.platform.startswith('win'):
            # Replace problematic characters on Windows
            error_str = error_str.encode('ascii', errors='replace').decode('ascii')
        return error_str
    except Exception:
        return "Error occurred (details unavailable)"

def log_safe(message: str, level: str = "INFO") -> None:
    """
    Log message safely with Windows CLI compatibility.
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
    """
    try:
        # Clean message for Windows
        if sys.platform.startswith('win'):
            message = message.encode('ascii', errors='replace').decode('ascii')
        
        # Get logger and log
        log_func = getattr(logger, level.lower())
        log_func(message)
        
    except UnicodeEncodeError:
        # Fallback logging
        try:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(safe_message)
        except Exception:
            # Ultimate fallback
            sys.stderr.write(f"[{level}] {message}\n")
            sys.stderr.flush()

class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility handler for logging."""
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment."""
        return sys.platform.startswith('win')
    
    @staticmethod
    def setup_windows_console() -> None:
        """Setup Windows console for proper Unicode handling."""
        if sys.platform.startswith('win'):
            try:
                # Set console code page to UTF-8
                os.system('chcp 65001 > nul')
                
                # Setup colorama if available
                try:
                    from colorama import init
                    init()
                except ImportError:
                    pass
                    
            except Exception as e:
                # Silently fail if console setup fails
                pass
    
    @staticmethod
    def create_safe_logging_handler() -> logging.Handler:
        """Create a safe logging handler for Windows."""
        if sys.platform.startswith('win'):
            # Create handler with UTF-8 encoding
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            return handler
        else:
            # Standard handler for non-Windows
            return logging.StreamHandler()

def cli_handler(func):
    """
    Decorator to handle CLI functions with Windows compatibility.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            # Setup Windows console if needed
            if sys.platform.startswith('win'):
                WindowsCliCompatibilityHandler.setup_windows_console()
            
            # Execute function
            return func(*args, **kwargs)
            
        except UnicodeEncodeError as e:
            safe_print(f"Encoding error: {safe_format_error(e)}")
            return None
        except Exception as e:
            safe_print(f"Error: {safe_format_error(e)}")
            return None
    
    return wrapper

# Initialize Windows console on import
if sys.platform.startswith('win'):
    WindowsCliCompatibilityHandler.setup_windows_console() 