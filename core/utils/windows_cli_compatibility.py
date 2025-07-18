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
        if sys.platform.startswith('win'):
            # Handle Windows-specific error formatting
            error_str = str(error)
            # Remove any problematic characters
            error_str = error_str.replace('\x00', '')
            return error_str
        else:
            return str(error)
    except:
        return f"Error: {type(error).__name__}"

def log_safe(message: str, level: str = "INFO") -> None:
    """
    Safe logging function that handles Windows CLI issues.
    
    Args:
        message: Message to log
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    try:
        if level.upper() == "DEBUG":
            logger.debug(message)
        elif level.upper() == "INFO":
            logger.info(message)
        elif level.upper() == "WARNING":
            logger.warning(message)
        elif level.upper() == "ERROR":
            logger.error(message)
        elif level.upper() == "CRITICAL":
            logger.critical(message)
        else:
            logger.info(message)
    except Exception as e:
        # Fallback to safe_print if logging fails
        safe_print(f"[{level.upper()}] {message}")
        safe_print(f"Logging error: {safe_format_error(e)}")

class WindowsCliCompatibilityHandler:
    """Handler for Windows CLI compatibility issues."""
    
    def __init__(self):
        """Initialize the Windows CLI compatibility handler."""
        self.is_windows = sys.platform.startswith('win')
        self.encoding = 'utf-8' if self.is_windows else None
        
        # Initialize colorama on Windows if available
        if self.is_windows and 'colorama' in sys.modules:
            try:
                from colorama import init
                init()
            except ImportError:
                pass
    
    def setup_console(self) -> None:
        """Setup console for optimal Windows compatibility."""
        if self.is_windows:
            try:
                # Set console code page to UTF-8
                os.system('chcp 65001 > nul')
                
                # Enable virtual terminal processing if available
                if hasattr(os, 'system'):
                    os.system('echo off')
            except:
                pass
    
    def get_safe_path(self, path: Union[str, Path]) -> Path:
        """
        Get a safe path that works on Windows.
        
        Args:
            path: Path to convert
            
        Returns:
            Safe Path object
        """
        try:
            path_obj = Path(path)
            # Handle Windows path issues
            if self.is_windows:
                # Convert to absolute path if needed
                if not path_obj.is_absolute():
                    path_obj = path_obj.resolve()
                # Handle long path issues
                if len(str(path_obj)) > 260:
                    path_obj = Path("\\\\?\\" + str(path_obj))
            return path_obj
        except Exception as e:
            log_safe(f"Path conversion error: {safe_format_error(e)}", "WARNING")
            return Path(str(path))
    
    def safe_file_operation(self, operation: callable, *args, **kwargs) -> Any:
        """
        Safely perform file operations on Windows.
        
        Args:
            operation: File operation function to call
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
        """
        try:
            return operation(*args, **kwargs)
        except (OSError, IOError) as e:
            if self.is_windows:
                # Handle Windows-specific file issues
                error_msg = safe_format_error(e)
                if "Access is denied" in error_msg:
                    log_safe("File access denied - trying alternative method", "WARNING")
                    # Try alternative approach
                    try:
                        # Add retry logic or alternative path
                        return operation(*args, **kwargs)
                    except:
                        raise
            raise

# Global instance for easy access
cli_handler = WindowsCliCompatibilityHandler()

# Convenience functions
def setup_windows_cli():
    """Setup Windows CLI compatibility."""
    cli_handler.setup_console()

def get_safe_path(path: Union[str, Path]) -> Path:
    """Get a safe path for Windows."""
    return cli_handler.get_safe_path(path)

def safe_file_operation(operation: callable, *args, **kwargs) -> Any:
    """Safely perform file operations."""
    return cli_handler.safe_file_operation(operation, *args, **kwargs)

# Auto-setup on import
if __name__ != "__main__":
    setup_windows_cli() 