import matplotlib.pyplot as plt
import matplotlib.style as style
from IPython import display
import numpy as np
from collections import deque
import seaborn as sns
import warnings
from typing import List, Optional, Dict, Any
warnings.filterwarnings('ignore')
from logger import get_logger, log_exception

# Set modern plotting style
plt.style.use('dark_background')
sns.set_palette("husl")
plt.ion()

class TrainingVisualizer:
    """Advanced visualization system for Snake AI training progress.
    
    Provides real-time plotting of training metrics including score progression,
    performance trends, recent game performance, and training statistics.
    
    Args:
        window_size (int): Size of the sliding window for recent performance tracking
    """
    
    def __init__(self, window_size: int = 100) -> None:
        # Initialize logging
        self.logger = get_logger("visualizer")
        self.logger.info("Initializing Training Visualizer")
        
        try:
            # Data storage - set window_size first
            self.window_size = window_size
            self.recent_scores = deque(maxlen=window_size)
            
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle('Snake AI Training Dashboard', fontsize=16, fontweight='bold')
            
            # Configure subplots
            self._setup_plots()
            
            # Colors
            self.colors = {
                'score': '#3498db',
                'mean': '#e74c3c', 
                'record': '#f1c40f',
                'recent': '#2ecc71',
                'grid': '#34495e'
            }
            
            self.logger.debug("Visualizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize visualizer: {e}")
            raise
        
    def _setup_plots(self) -> None:
        """Setup all subplot configurations.
        
        Configures the 2x2 subplot layout with appropriate titles, labels,
        and styling for the training dashboard.
        """
        # Score progression
        self.ax1.set_title('Score Progression', fontweight='bold', pad=20)
        self.ax1.set_xlabel('Game Number')
        self.ax1.set_ylabel('Score')
        self.ax1.grid(True, alpha=0.3)
        
        # Moving average
        self.ax2.set_title('Performance Trends', fontweight='bold', pad=20)
        self.ax2.set_xlabel('Game Number')
        self.ax2.set_ylabel('Average Score')
        self.ax2.grid(True, alpha=0.3)
        
        # Recent performance
        self.ax3.set_title(f'Recent {self.window_size} Games', fontweight='bold', pad=20)
        self.ax3.set_xlabel('Recent Games')
        self.ax3.set_ylabel('Score')
        self.ax3.grid(True, alpha=0.3)
        
        # Statistics
        self.ax4.set_title('Training Statistics', fontweight='bold', pad=20)
        self.ax4.axis('off')
        
    @log_exception(get_logger("visualizer"), "Error updating plots")
    def plot(self, scores: List[float], mean_scores: List[float], 
             game_number: Optional[int] = None, record: Optional[float] = None) -> None:
        """Enhanced plotting with multiple visualizations.
        
        Args:
            scores: List of all game scores
            mean_scores: List of running mean scores
            game_number: Current game number (optional)
            record: Current record score (optional)
        """
        try:
            # Validate inputs
            if not scores:
                self.logger.warning("Received empty scores list")
                return
                
            if mean_scores is None:
                self.logger.warning("Received None values for mean_scores")
                return
            
            # Clear all plots
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.clear()
                
            # Update recent scores
            self.recent_scores.append(scores[-1])
            
            # Plot 1: Score progression with highlights
            self._plot_score_progression(scores, record)
            
            # Plot 2: Performance trends
            self._plot_performance_trends(scores, mean_scores)
            
            # Plot 3: Recent performance
            self._plot_recent_performance()
            
            # Plot 4: Statistics panel
            self._plot_statistics(scores, mean_scores, game_number, record)
            
            # Refresh display
            self._refresh_display()
            
            # Log update every 10 games
            if len(scores) % 10 == 0:
                self.logger.debug(f"Plots updated - Game {len(scores)}, Score: {scores[-1]}")
                
        except Exception as e:
            self.logger.error(f"Failed to update plots: {e}")
            # Continue without crashing the training
        
    def _plot_score_progression(self, scores, record):
        """Plot score progression with record highlights"""
        self.ax1.set_title('Score Progression', fontweight='bold', pad=20)
        self.ax1.set_xlabel('Game Number')
        self.ax1.set_ylabel('Score')
        
        # Main score line
        x = range(1, len(scores) + 1)
        self.ax1.plot(x, scores, color=self.colors['score'], linewidth=1.5, alpha=0.8)
        
        # Highlight record scores
        if record and record > 0:
            record_games = [i for i, score in enumerate(scores) if score == record]
            if record_games:
                self.ax1.scatter([i+1 for i in record_games], [scores[i] for i in record_games], 
                               color=self.colors['record'], s=100, zorder=5, 
                               label=f'Record: {record}', marker='*')
        
        # Current score annotation
        if scores:
            self.ax1.annotate(f'{scores[-1]}', 
                            xy=(len(scores), scores[-1]), 
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['score'], alpha=0.7),
                            fontweight='bold')
        
        self.ax1.grid(True, alpha=0.3)
        if record and record > 0:
            self.ax1.legend()
        
    def _plot_performance_trends(self, scores, mean_scores):
        """Plot performance trends with smoothing"""
        self.ax2.set_title('Performance Trends', fontweight='bold', pad=20)
        self.ax2.set_xlabel('Game Number')
        self.ax2.set_ylabel('Average Score')
        
        x = range(1, len(mean_scores) + 1)
        
        # Mean score line
        self.ax2.plot(x, mean_scores, color=self.colors['mean'], linewidth=2, 
                     label='Mean Score')
        
        # Smoothed trend (if enough data)
        if len(mean_scores) > 10:
            window = min(20, len(mean_scores) // 5)
            smoothed = np.convolve(mean_scores, np.ones(window)/window, mode='valid')
            smooth_x = range(window, len(mean_scores) + 1)
            self.ax2.plot(smooth_x, smoothed, color=self.colors['recent'], 
                         linewidth=3, alpha=0.8, label='Trend')
        
        # Current mean annotation
        if mean_scores:
            self.ax2.annotate(f'{mean_scores[-1]:.1f}', 
                            xy=(len(mean_scores), mean_scores[-1]), 
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['mean'], alpha=0.7),
                            fontweight='bold')
        
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
    def _plot_recent_performance(self):
        """Plot recent performance as bar chart"""
        self.ax3.set_title(f'Recent {len(self.recent_scores)} Games', fontweight='bold', pad=20)
        self.ax3.set_xlabel('Recent Games')
        self.ax3.set_ylabel('Score')
        
        if self.recent_scores:
            x = range(len(self.recent_scores))
            bars = self.ax3.bar(x, self.recent_scores, color=self.colors['recent'], alpha=0.7)
            
            # Highlight best recent score
            max_idx = np.argmax(self.recent_scores)
            bars[max_idx].set_color(self.colors['record'])
            
            # Add trend line
            if len(self.recent_scores) > 3:
                z = np.polyfit(x, self.recent_scores, 1)
                p = np.poly1d(z)
                self.ax3.plot(x, p(x), color=self.colors['score'], linewidth=2, alpha=0.8)
        
        self.ax3.grid(True, alpha=0.3)
        
    def _plot_statistics(self, scores, mean_scores, game_number, record):
        """Plot statistics panel"""
        self.ax4.clear()
        self.ax4.set_title('Training Statistics', fontweight='bold', pad=20)
        self.ax4.axis('off')
        
        if not scores:
            return
            
        # Calculate statistics
        current_score = scores[-1]
        current_mean = mean_scores[-1] if mean_scores else 0
        max_score = max(scores)
        min_score = min(scores)
        std_score = np.std(scores)
        
        # Recent improvement
        recent_improvement = 0
        if len(scores) > 10:
            recent_avg = np.mean(scores[-10:])
            older_avg = np.mean(scores[-20:-10]) if len(scores) > 20 else np.mean(scores[:-10])
            recent_improvement = recent_avg - older_avg
        
        # Create statistics text
        stats_text = f"""
        🎮 Current Game: {game_number or len(scores)}
        📊 Current Score: {current_score}
        📈 Average Score: {current_mean:.1f}
        🏆 Record Score: {record or max_score}
        📉 Lowest Score: {min_score}
        📊 Score Std Dev: {std_score:.1f}
        🔥 Recent Trend: {recent_improvement:+.1f}
        
        🎯 Games Played: {len(scores)}
        ⭐ Success Rate: {(len([s for s in scores if s > current_mean]) / len(scores) * 100):.1f}%
        """
        
        self.ax4.text(0.1, 0.9, stats_text, transform=self.ax4.transAxes, 
                     fontsize=12, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', alpha=0.8))
        
    def _refresh_display(self):
        """Refresh the display"""
        plt.tight_layout()
        try:
            # Try to use IPython display if available
            get_ipython()
            display.clear_output(wait=True)
            display.display(self.fig)
        except NameError:
            # Fallback to regular matplotlib
            plt.draw()
            plt.pause(0.01)

def simple_plot(scores, mean_scores):
    """Simple plotting function for basic score visualization"""
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)

# Global visualizer instance
visualizer = TrainingVisualizer()

@log_exception(get_logger("visualizer"), "Error in main plot function")
def plot(scores: List[float], mean_scores: List[float], 
         game_number: Optional[int] = None, record: Optional[float] = None) -> None:
    """Enhanced plotting function with dashboard.
    
    Args:
        scores: List of all game scores
        mean_scores: List of running mean scores
        game_number: Current game number (optional)
        record: Current record score (optional)
    """
    try:
        visualizer.plot(scores, mean_scores, game_number, record)
        
    except Exception as e:
        logger = get_logger("visualizer")
        logger.error(f"Plot function failed: {e}")
        # Don't crash training if plotting fails

# Legacy function for compatibility
@log_exception(get_logger("visualizer"), "Error in simple plot function")
def simple_plot(scores: List[float], mean_scores: List[float]) -> None:
    """Simple plotting for basic use.
    
    Args:
        scores: List of all game scores
        mean_scores: List of running mean scores
    """
    try:
        plt.clf()
        plt.title("Training Progress")
        plt.xlabel('Games')
        plt.ylabel('Score')
        
        if scores and len(scores) > 0:
            plt.plot(scores, label='Score', color='#3498db')
        
        if mean_scores and len(mean_scores) > 0:
            plt.plot(mean_scores, label='Average', color='#e74c3c')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if scores and len(scores) > 0:
            plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        if mean_scores and len(mean_scores) > 0:
            plt.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.1f}')
        
        plt.show(block=False)
        plt.pause(0.1)
        
    except Exception as e:
        logger = get_logger("visualizer")
        logger.error(f"Simple plot function failed: {e}")
        # Don't crash if plotting fails