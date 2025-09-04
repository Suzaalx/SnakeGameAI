#!/usr/bin/env python3
"""
Logging Configuration for Snake Game AI

This module provides centralized logging configuration for the entire project,
including training logs, error handling, and performance monitoring.

Usage:
    from logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.error("Model failed to load", exc_info=True)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import traceback

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted

class GameMetricsFilter(logging.Filter):
    """Filter for game-specific metrics logging"""
    
    def filter(self, record):
        # Only allow records with 'metrics' in the logger name
        return 'metrics' in record.name.lower()

class TrainingLogger:
    """Specialized logger for training metrics and progress"""
    
    def __init__(self, name: str = "training"):
        self.logger = get_logger(f"training.{name}")
        self.start_time = None
        self.game_count = 0
        
    def start_training(self, config: dict):
        """Log training start with configuration"""
        self.start_time = datetime.now()
        self.logger.info("🚀 Training started")
        self.logger.info(f"Configuration: {config}")
        
    def log_game_result(self, game_num: int, score: int, record: int, 
                       epsilon: float, moves: int, duration: float = None):
        """Log individual game results"""
        self.game_count += 1
        
        # Create detailed game log
        game_info = {
            'game': game_num,
            'score': score,
            'record': record,
            'epsilon': f"{epsilon:.3f}",
            'moves': moves
        }
        
        if duration:
            game_info['duration'] = f"{duration:.2f}s"
            
        # Log with appropriate level based on performance
        if score >= record:
            self.logger.info(f"🏆 NEW RECORD! Game {game_num}: {game_info}")
        elif score >= 20:
            self.logger.info(f"🎯 HIGH SCORE! Game {game_num}: {game_info}")
        elif score >= 10:
            self.logger.info(f"✅ Good game {game_num}: {game_info}")
        else:
            self.logger.debug(f"📊 Game {game_num}: {game_info}")
    
    def log_model_save(self, filepath: str, score: int):
        """Log model saving events"""
        self.logger.info(f"💾 Model saved: {filepath} (Score: {score})")
    
    def log_training_stats(self, total_games: int, avg_score: float, 
                          max_score: int, training_time: float):
        """Log comprehensive training statistics"""
        self.logger.info("📊 Training Statistics:")
        self.logger.info(f"   Total Games: {total_games}")
        self.logger.info(f"   Average Score: {avg_score:.2f}")
        self.logger.info(f"   Best Score: {max_score}")
        self.logger.info(f"   Training Time: {training_time:.2f}s")
        
    def end_training(self):
        """Log training completion"""
        if self.start_time:
            duration = datetime.now() - self.start_time
            self.logger.info(f"🏁 Training completed in {duration}")
            self.logger.info(f"📈 Total games played: {self.game_count}")

def setup_logging(level: str = "INFO", 
                 console_output: bool = True,
                 file_output: bool = True,
                 metrics_file: bool = True) -> None:
    """Setup comprehensive logging configuration"""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(colored_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler for general logs
    if file_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"snake_ai_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always capture debug in files
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Separate handler for metrics
    if metrics_file:
        metrics_file = LOGS_DIR / "training_metrics.log"
        metrics_handler = logging.FileHandler(metrics_file)
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.setFormatter(simple_formatter)
        metrics_handler.addFilter(GameMetricsFilter())
        root_logger.addHandler(metrics_handler)
    
    # Log the setup
    logger = logging.getLogger("logger")
    logger.info(f"Logging initialized - Level: {level}")
    if file_output:
        logger.info(f"Log file: {log_file}")
    if metrics_file:
        logger.info(f"Metrics file: {metrics_file}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)

def log_exception(logger: logging.Logger, message: str = "An error occurred"):
    """Decorator to log exceptions with full traceback"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{message}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        return wrapper
    return decorator

def log_performance(logger: logging.Logger, operation: str):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"{operation} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{operation} failed after {duration:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"performance.{name}")
        self.metrics = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {'start': datetime.now()}
        
    def end_timer(self, operation: str, log_result: bool = True):
        """End timing and optionally log the result"""
        if operation in self.metrics and 'start' in self.metrics[operation]:
            duration = (datetime.now() - self.metrics[operation]['start']).total_seconds()
            self.metrics[operation]['duration'] = duration
            
            if log_result:
                self.logger.debug(f"{operation}: {duration:.3f}s")
            
            return duration
        return None
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")

# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()

# Export commonly used items
__all__ = [
    'setup_logging',
    'get_logger', 
    'TrainingLogger',
    'PerformanceMonitor',
    'log_exception',
    'log_performance'
]