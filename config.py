"""Configuration file for Snake Game AI

This module contains all configurable parameters for the Snake Game AI project.
Modify these values to experiment with different training configurations.
"""

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GameConfig:
    """Game-related configuration parameters"""
    # Display settings
    WINDOW_WIDTH: int = 640
    WINDOW_HEIGHT: int = 480
    BLOCK_SIZE: int = 20
    GAME_SPEED: int = 40
    
    # Game mechanics
    MAX_FRAMES_PER_GAME: int = 100  # Multiplied by snake length
    
    # Visual settings
    SHOW_GRID: bool = True
    ENABLE_GLOW_EFFECTS: bool = True
    
@dataclass
class TrainingConfig:
    """Training-related configuration parameters"""
    # Memory and batch settings
    MAX_MEMORY: int = 100_000
    BATCH_SIZE: int = 1000
    
    # Learning parameters
    LEARNING_RATE: float = 0.001
    GAMMA: float = 0.9  # Discount factor
    
    # Exploration parameters
    EPSILON_START: float = 0.8
    EPSILON_END: float = 0.0
    EPSILON_DECAY_GAMES: int = 80
    
    # Model architecture
    INPUT_SIZE: int = 11
    HIDDEN_SIZE: int = 256
    OUTPUT_SIZE: int = 3
    
    # Training control
    SAVE_MODEL_EVERY: int = 10  # Save model every N games
    PRINT_STATS_EVERY: int = 1  # Print statistics every N games
    
@dataclass
class ModelConfig:
    """Model-related configuration parameters"""
    # File paths
    MODEL_DIR: str = "model"
    BEST_MODEL_NAME: str = "model.pth"
    LATEST_MODEL_NAME: str = "model_latest.pth"
    
    # Model saving
    SAVE_ON_NEW_RECORD: bool = True
    SAVE_LATEST_ALWAYS: bool = True
    
@dataclass
class PlottingConfig:
    """Plotting and visualization configuration"""
    # Dashboard settings
    ENABLE_DASHBOARD: bool = True
    DASHBOARD_UPDATE_FREQUENCY: int = 1  # Update every N games
    RECENT_GAMES_WINDOW: int = 100
    
    # Plot styling
    FIGURE_SIZE: Tuple[int, int] = (15, 10)
    USE_DARK_THEME: bool = True
    
    # Performance
    MAX_PLOT_POINTS: int = 1000  # Limit points for performance
    
@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    LOG_FILE: str = "training.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console output
    VERBOSE: bool = True
    SHOW_PROGRESS_BAR: bool = True

class Config:
    """Main configuration class that combines all config sections"""
    
    def __init__(self):
        self.game = GameConfig()
        self.training = TrainingConfig()
        self.model = ModelConfig()
        self.plotting = PlottingConfig()
        self.logging = LoggingConfig()
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model.MODEL_DIR, exist_ok=True)
    
    def get_model_path(self, filename: str = None) -> str:
        """Get full path for model file"""
        if filename is None:
            filename = self.model.BEST_MODEL_NAME
        return os.path.join(self.model.MODEL_DIR, filename)
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'game': self.game.__dict__,
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'plotting': self.plotting.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_to_file(self, filepath: str = "config.json"):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_from_file(self, filepath: str = "config.json"):
        """Load configuration from JSON file"""
        import json
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            self.update_from_dict(config_dict)
        except FileNotFoundError:
            print(f"Configuration file {filepath} not found. Using default settings.")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filepath}. Using default settings.")

# Global configuration instance
config = Config()

# Color schemes for different themes
COLOR_SCHEMES = {
    'dark': {
        'background': (20, 25, 40),
        'grid': (40, 45, 60),
        'snake_head': (46, 204, 113),
        'snake_body': (39, 174, 96),
        'snake_outline': (27, 79, 114),
        'food': (231, 76, 60),
        'food_glow': (192, 57, 43),
        'text': (236, 240, 241),
        'score_bg': (52, 73, 94),
        'high_score': (241, 196, 15)
    },
    'light': {
        'background': (240, 248, 255),
        'grid': (220, 220, 220),
        'snake_head': (34, 139, 34),
        'snake_body': (50, 205, 50),
        'snake_outline': (0, 100, 0),
        'food': (220, 20, 60),
        'food_glow': (255, 69, 0),
        'text': (25, 25, 112),
        'score_bg': (230, 230, 250),
        'high_score': (255, 215, 0)
    },
    'neon': {
        'background': (10, 10, 10),
        'grid': (30, 30, 30),
        'snake_head': (0, 255, 127),
        'snake_body': (0, 255, 255),
        'snake_outline': (255, 255, 255),
        'food': (255, 20, 147),
        'food_glow': (255, 105, 180),
        'text': (255, 255, 255),
        'score_bg': (25, 25, 25),
        'high_score': (255, 255, 0)
    }
}

def get_color_scheme(theme: str = 'dark') -> dict:
    """Get color scheme for specified theme"""
    return COLOR_SCHEMES.get(theme, COLOR_SCHEMES['dark'])