#!/usr/bin/env python3
"""
Model Performance Evaluation and Metrics Dashboard

This module provides comprehensive evaluation tools for the Snake AI model,
including performance metrics, statistical analysis, and evaluation dashboards.

Usage:
    from evaluation import ModelEvaluator, PerformanceMetrics
    
    evaluator = ModelEvaluator(agent)
    metrics = evaluator.evaluate_model(num_games=100)
    evaluator.generate_report(metrics)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from collections import defaultdict, deque
import time
import statistics

from logger import get_logger, log_exception
from snake_gameai import SnakeGameAI
from agent import Agent


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""
    
    # Basic statistics
    total_games: int = 0
    total_score: int = 0
    average_score: float = 0.0
    median_score: float = 0.0
    max_score: int = 0
    min_score: int = 0
    std_deviation: float = 0.0
    
    # Performance categories
    games_over_10: int = 0
    games_over_20: int = 0
    games_over_30: int = 0
    games_over_50: int = 0
    
    # Efficiency metrics
    average_moves_per_game: float = 0.0
    average_moves_per_food: float = 0.0
    efficiency_ratio: float = 0.0  # score / moves
    
    # Time metrics
    total_evaluation_time: float = 0.0
    average_game_duration: float = 0.0
    
    # Advanced metrics
    score_distribution: Dict[str, int] = field(default_factory=dict)
    performance_trend: List[float] = field(default_factory=list)
    consistency_score: float = 0.0
    improvement_rate: float = 0.0
    
    # Game-specific metrics
    collision_types: Dict[str, int] = field(default_factory=dict)
    survival_rates: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'basic_stats': {
                'total_games': self.total_games,
                'total_score': self.total_score,
                'average_score': self.average_score,
                'median_score': self.median_score,
                'max_score': self.max_score,
                'min_score': self.min_score,
                'std_deviation': self.std_deviation
            },
            'performance_categories': {
                'games_over_10': self.games_over_10,
                'games_over_20': self.games_over_20,
                'games_over_30': self.games_over_30,
                'games_over_50': self.games_over_50
            },
            'efficiency_metrics': {
                'average_moves_per_game': self.average_moves_per_game,
                'average_moves_per_food': self.average_moves_per_food,
                'efficiency_ratio': self.efficiency_ratio
            },
            'time_metrics': {
                'total_evaluation_time': self.total_evaluation_time,
                'average_game_duration': self.average_game_duration
            },
            'advanced_metrics': {
                'score_distribution': self.score_distribution,
                'performance_trend': self.performance_trend,
                'consistency_score': self.consistency_score,
                'improvement_rate': self.improvement_rate
            },
            'game_metrics': {
                'collision_types': self.collision_types,
                'survival_rates': self.survival_rates
            }
        }


class ModelEvaluator:
    """Comprehensive model evaluation system.
    
    Provides tools for evaluating model performance, generating metrics,
    and creating detailed evaluation reports with visualizations.
    """
    
    def __init__(self, agent: Optional[Agent] = None, model_path: Optional[str] = None):
        """Initialize the model evaluator.
        
        Args:
            agent: Pre-initialized agent instance
            model_path: Path to saved model file
        """
        self.logger = get_logger("evaluator")
        self.logger.info("Initializing Model Evaluator")
        
        if agent is not None:
            self.agent = agent
        elif model_path is not None:
            self.agent = Agent()
            self.agent.load_model(model_path)
        else:
            raise ValueError("Either agent or model_path must be provided")
        
        # Evaluation settings
        self.game_speed = 1000  # Fast evaluation
        self.show_games = False
        
        # Results storage
        self.evaluation_history: List[PerformanceMetrics] = []
        
        # Create evaluation directory
        self.eval_dir = Path("evaluation_results")
        self.eval_dir.mkdir(exist_ok=True)
        
    @log_exception(get_logger("evaluator"), "Error during model evaluation")
    def evaluate_model(self, num_games: int = 100, 
                      show_progress: bool = True) -> PerformanceMetrics:
        """Evaluate model performance over multiple games.
        
        Args:
            num_games: Number of games to evaluate
            show_progress: Whether to show progress updates
            
        Returns:
            PerformanceMetrics object containing evaluation results
        """
        self.logger.info(f"Starting model evaluation: {num_games} games")
        start_time = time.time()
        
        # Initialize tracking variables
        scores = []
        moves_per_game = []
        game_durations = []
        collision_types = defaultdict(int)
        
        # Performance tracking
        recent_scores = deque(maxlen=20)  # For trend analysis
        
        try:
            for game_num in range(num_games):
                if show_progress and (game_num + 1) % 10 == 0:
                    self.logger.info(f"Evaluation progress: {game_num + 1}/{num_games}")
                
                # Run single game evaluation
                game_start = time.time()
                score, moves, collision_type = self._evaluate_single_game()
                game_duration = time.time() - game_start
                
                # Store results
                scores.append(score)
                moves_per_game.append(moves)
                game_durations.append(game_duration)
                collision_types[collision_type] += 1
                recent_scores.append(score)
                
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(
                scores, moves_per_game, game_durations, 
                collision_types, time.time() - start_time
            )
            
            # Store evaluation in history
            self.evaluation_history.append(metrics)
            
            self.logger.info(f"Evaluation completed: Avg Score = {metrics.average_score:.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def _evaluate_single_game(self) -> Tuple[int, int, str]:
        """Evaluate a single game and return score, moves, and collision type.
        
        Returns:
            Tuple of (score, total_moves, collision_type)
        """
        game = SnakeGameAI()
        total_moves = 0
        collision_type = "timeout"
        
        while True:
            # Get current state
            state_old = self.agent.get_state(game)
            
            # Get action from agent
            final_move = self.agent.get_action(state_old)
            
            # Perform move
            reward, done, score = game.play_step(final_move)
            total_moves += 1
            
            if done:
                # Determine collision type
                if game.is_collision(game.head):
                    if (game.head.x < 0 or game.head.x >= game.w or 
                        game.head.y < 0 or game.head.y >= game.h):
                        collision_type = "boundary"
                    else:
                        collision_type = "self"
                else:
                    collision_type = "timeout"
                break
        
        return score, total_moves, collision_type
    
    def _calculate_metrics(self, scores: List[int], moves_per_game: List[int],
                          game_durations: List[float], collision_types: Dict[str, int],
                          total_time: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.
        
        Args:
            scores: List of game scores
            moves_per_game: List of moves per game
            game_durations: List of game durations
            collision_types: Dictionary of collision type counts
            total_time: Total evaluation time
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        # Basic statistics
        metrics.total_games = len(scores)
        metrics.total_score = sum(scores)
        metrics.average_score = statistics.mean(scores) if scores else 0
        metrics.median_score = statistics.median(scores) if scores else 0
        metrics.max_score = max(scores) if scores else 0
        metrics.min_score = min(scores) if scores else 0
        metrics.std_deviation = statistics.stdev(scores) if len(scores) > 1 else 0
        
        # Performance categories
        metrics.games_over_10 = sum(1 for s in scores if s >= 10)
        metrics.games_over_20 = sum(1 for s in scores if s >= 20)
        metrics.games_over_30 = sum(1 for s in scores if s >= 30)
        metrics.games_over_50 = sum(1 for s in scores if s >= 50)
        
        # Efficiency metrics
        metrics.average_moves_per_game = statistics.mean(moves_per_game) if moves_per_game else 0
        
        # Calculate moves per food (approximate)
        total_food = sum(scores)  # Each score point = 1 food
        total_moves = sum(moves_per_game)
        metrics.average_moves_per_food = total_moves / total_food if total_food > 0 else 0
        metrics.efficiency_ratio = total_food / total_moves if total_moves > 0 else 0
        
        # Time metrics
        metrics.total_evaluation_time = total_time
        metrics.average_game_duration = statistics.mean(game_durations) if game_durations else 0
        
        # Score distribution
        metrics.score_distribution = self._calculate_score_distribution(scores)
        
        # Performance trend (moving average)
        window_size = min(10, len(scores))
        if len(scores) >= window_size:
            metrics.performance_trend = [
                statistics.mean(scores[i:i+window_size]) 
                for i in range(len(scores) - window_size + 1)
            ]
        
        # Consistency score (inverse of coefficient of variation)
        if metrics.average_score > 0:
            cv = metrics.std_deviation / metrics.average_score
            metrics.consistency_score = 1 / (1 + cv)  # Higher is more consistent
        
        # Improvement rate (trend slope)
        if len(metrics.performance_trend) > 1:
            x = list(range(len(metrics.performance_trend)))
            y = metrics.performance_trend
            metrics.improvement_rate = np.polyfit(x, y, 1)[0]  # Slope of linear fit
        
        # Collision types
        metrics.collision_types = dict(collision_types)
        
        # Survival rates
        total_games = len(scores)
        metrics.survival_rates = {
            'score_0_5': sum(1 for s in scores if 0 <= s <= 5) / total_games,
            'score_6_15': sum(1 for s in scores if 6 <= s <= 15) / total_games,
            'score_16_30': sum(1 for s in scores if 16 <= s <= 30) / total_games,
            'score_30_plus': sum(1 for s in scores if s > 30) / total_games
        }
        
        return metrics
    
    def _calculate_score_distribution(self, scores: List[int]) -> Dict[str, int]:
        """Calculate score distribution in ranges.
        
        Args:
            scores: List of game scores
            
        Returns:
            Dictionary with score range counts
        """
        distribution = {
            '0-5': 0, '6-10': 0, '11-20': 0, '21-30': 0, 
            '31-50': 0, '51-100': 0, '100+': 0
        }
        
        for score in scores:
            if score <= 5:
                distribution['0-5'] += 1
            elif score <= 10:
                distribution['6-10'] += 1
            elif score <= 20:
                distribution['11-20'] += 1
            elif score <= 30:
                distribution['21-30'] += 1
            elif score <= 50:
                distribution['31-50'] += 1
            elif score <= 100:
                distribution['51-100'] += 1
            else:
                distribution['100+'] += 1
        
        return distribution
    
    @log_exception(get_logger("evaluator"), "Error generating evaluation dashboard")
    def generate_dashboard(self, metrics: PerformanceMetrics, 
                          save_path: Optional[str] = None) -> None:
        """Generate comprehensive evaluation dashboard with visualizations.
        
        Args:
            metrics: Performance metrics to visualize
            save_path: Optional path to save the dashboard
        """
        self.logger.info("Generating evaluation dashboard")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Snake AI Model Evaluation Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Score Distribution
        ax1 = plt.subplot(3, 4, 1)
        self._plot_score_distribution(ax1, metrics)
        
        # 2. Performance Categories
        ax2 = plt.subplot(3, 4, 2)
        self._plot_performance_categories(ax2, metrics)
        
        # 3. Efficiency Metrics
        ax3 = plt.subplot(3, 4, 3)
        self._plot_efficiency_metrics(ax3, metrics)
        
        # 4. Collision Types
        ax4 = plt.subplot(3, 4, 4)
        self._plot_collision_types(ax4, metrics)
        
        # 5. Performance Trend
        ax5 = plt.subplot(3, 4, (5, 6))
        self._plot_performance_trend(ax5, metrics)
        
        # 6. Survival Rates
        ax6 = plt.subplot(3, 4, 7)
        self._plot_survival_rates(ax6, metrics)
        
        # 7. Key Statistics
        ax7 = plt.subplot(3, 4, 8)
        self._plot_key_statistics(ax7, metrics)
        
        # 8. Model Comparison (if history available)
        ax8 = plt.subplot(3, 4, (9, 12))
        self._plot_evaluation_history(ax8)
        
        plt.tight_layout()
        
        # Save dashboard
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.eval_dir / f"evaluation_dashboard_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Dashboard saved to: {save_path}")
        
        plt.show()
    
    def _plot_score_distribution(self, ax, metrics: PerformanceMetrics) -> None:
        """Plot score distribution histogram."""
        ranges = list(metrics.score_distribution.keys())
        counts = list(metrics.score_distribution.values())
        
        bars = ax.bar(ranges, counts, color='skyblue', alpha=0.7)
        ax.set_title('Score Distribution', fontweight='bold')
        ax.set_xlabel('Score Range')
        ax.set_ylabel('Number of Games')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
    
    def _plot_performance_categories(self, ax, metrics: PerformanceMetrics) -> None:
        """Plot performance category achievements."""
        categories = ['10+', '20+', '30+', '50+']
        counts = [metrics.games_over_10, metrics.games_over_20, 
                 metrics.games_over_30, metrics.games_over_50]
        percentages = [c / metrics.total_games * 100 for c in counts]
        
        bars = ax.bar(categories, percentages, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        ax.set_title('Achievement Rates', fontweight='bold')
        ax.set_xlabel('Score Threshold')
        ax.set_ylabel('Percentage of Games (%)')
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom')
    
    def _plot_efficiency_metrics(self, ax, metrics: PerformanceMetrics) -> None:
        """Plot efficiency metrics."""
        metrics_names = ['Avg Moves\nper Game', 'Avg Moves\nper Food', 'Efficiency\nRatio']
        values = [metrics.average_moves_per_game, metrics.average_moves_per_food, 
                 metrics.efficiency_ratio * 100]  # Convert ratio to percentage
        
        bars = ax.bar(metrics_names, values, color=['lightcoral', 'lightgreen', 'lightblue'], alpha=0.7)
        ax.set_title('Efficiency Metrics', fontweight='bold')
        ax.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
    
    def _plot_collision_types(self, ax, metrics: PerformanceMetrics) -> None:
        """Plot collision types pie chart."""
        if metrics.collision_types:
            labels = list(metrics.collision_types.keys())
            sizes = list(metrics.collision_types.values())
            colors = ['lightcoral', 'lightskyblue', 'lightgreen']
            
            ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
            ax.set_title('Collision Types', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No collision data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Collision Types', fontweight='bold')
    
    def _plot_performance_trend(self, ax, metrics: PerformanceMetrics) -> None:
        """Plot performance trend over time."""
        if metrics.performance_trend:
            ax.plot(metrics.performance_trend, marker='o', linewidth=2, markersize=4)
            ax.set_title('Performance Trend (Moving Average)', fontweight='bold')
            ax.set_xlabel('Game Window')
            ax.set_ylabel('Average Score')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(metrics.performance_trend) > 1:
                x = list(range(len(metrics.performance_trend)))
                z = np.polyfit(x, metrics.performance_trend, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.3f})')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data for trend', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Trend', fontweight='bold')
    
    def _plot_survival_rates(self, ax, metrics: PerformanceMetrics) -> None:
        """Plot survival rates by score range."""
        ranges = list(metrics.survival_rates.keys())
        rates = [r * 100 for r in metrics.survival_rates.values()]  # Convert to percentage
        
        bars = ax.bar(ranges, rates, color='gold', alpha=0.7)
        ax.set_title('Survival Rates by Score', fontweight='bold')
        ax.set_xlabel('Score Range')
        ax.set_ylabel('Percentage of Games (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{rate:.1f}%', ha='center', va='bottom')
    
    def _plot_key_statistics(self, ax, metrics: PerformanceMetrics) -> None:
        """Plot key statistics as text."""
        ax.axis('off')
        
        stats_text = f"""
        KEY STATISTICS
        
        Total Games: {metrics.total_games}
        Average Score: {metrics.average_score:.2f}
        Median Score: {metrics.median_score:.2f}
        Max Score: {metrics.max_score}
        Std Deviation: {metrics.std_deviation:.2f}
        
        Consistency Score: {metrics.consistency_score:.3f}
        Improvement Rate: {metrics.improvement_rate:.4f}
        
        Avg Game Duration: {metrics.average_game_duration:.2f}s
        Total Eval Time: {metrics.total_evaluation_time:.1f}s
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _plot_evaluation_history(self, ax) -> None:
        """Plot evaluation history comparison."""
        if len(self.evaluation_history) > 1:
            avg_scores = [m.average_score for m in self.evaluation_history]
            max_scores = [m.max_score for m in self.evaluation_history]
            consistency_scores = [m.consistency_score for m in self.evaluation_history]
            
            x = list(range(1, len(self.evaluation_history) + 1))
            
            ax.plot(x, avg_scores, 'o-', label='Average Score', linewidth=2)
            ax.plot(x, max_scores, 's-', label='Max Score', linewidth=2)
            
            # Secondary y-axis for consistency
            ax2 = ax.twinx()
            ax2.plot(x, consistency_scores, '^-', color='red', label='Consistency', linewidth=2)
            ax2.set_ylabel('Consistency Score', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title('Evaluation History Comparison', fontweight='bold')
            ax.set_xlabel('Evaluation Session')
            ax.set_ylabel('Score')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No evaluation history available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Evaluation History', fontweight='bold')
    
    def generate_report(self, metrics: PerformanceMetrics, 
                       save_path: Optional[str] = None) -> str:
        """Generate detailed text report of evaluation results.
        
        Args:
            metrics: Performance metrics to report
            save_path: Optional path to save the report
            
        Returns:
            String containing the full report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
========================================
SNAKE AI MODEL EVALUATION REPORT
========================================
Generated: {timestamp}
Total Games Evaluated: {metrics.total_games}
Evaluation Duration: {metrics.total_evaluation_time:.2f} seconds

========================================
BASIC PERFORMANCE STATISTICS
========================================
Total Score: {metrics.total_score}
Average Score: {metrics.average_score:.2f}
Median Score: {metrics.median_score:.2f}
Maximum Score: {metrics.max_score}
Minimum Score: {metrics.min_score}
Standard Deviation: {metrics.std_deviation:.2f}

========================================
PERFORMANCE CATEGORIES
========================================
Games with Score ≥ 10: {metrics.games_over_10} ({metrics.games_over_10/metrics.total_games*100:.1f}%)
Games with Score ≥ 20: {metrics.games_over_20} ({metrics.games_over_20/metrics.total_games*100:.1f}%)
Games with Score ≥ 30: {metrics.games_over_30} ({metrics.games_over_30/metrics.total_games*100:.1f}%)
Games with Score ≥ 50: {metrics.games_over_50} ({metrics.games_over_50/metrics.total_games*100:.1f}%)

========================================
EFFICIENCY METRICS
========================================
Average Moves per Game: {metrics.average_moves_per_game:.2f}
Average Moves per Food: {metrics.average_moves_per_food:.2f}
Efficiency Ratio (Score/Moves): {metrics.efficiency_ratio:.4f}

========================================
ADVANCED METRICS
========================================
Consistency Score: {metrics.consistency_score:.3f}
Improvement Rate: {metrics.improvement_rate:.4f}
Average Game Duration: {metrics.average_game_duration:.3f} seconds

========================================
SCORE DISTRIBUTION
========================================
"""
        
        for range_name, count in metrics.score_distribution.items():
            percentage = count / metrics.total_games * 100
            report += f"{range_name}: {count} games ({percentage:.1f}%)\n"
        
        report += f"""
========================================
COLLISION ANALYSIS
========================================
"""
        
        for collision_type, count in metrics.collision_types.items():
            percentage = count / metrics.total_games * 100
            report += f"{collision_type.title()} Collisions: {count} ({percentage:.1f}%)\n"
        
        report += f"""
========================================
SURVIVAL ANALYSIS
========================================
"""
        
        for range_name, rate in metrics.survival_rates.items():
            report += f"{range_name}: {rate*100:.1f}%\n"
        
        report += f"""
========================================
RECOMMENDations
========================================
"""
        
        # Generate recommendations based on metrics
        recommendations = self._generate_recommendations(metrics)
        for rec in recommendations:
            report += f"• {rec}\n"
        
        report += "\n========================================\n"
        
        # Save report if path provided
        if save_path is None:
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.eval_dir / f"evaluation_report_{timestamp_file}.txt"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Evaluation report saved to: {save_path}")
        return report
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate recommendations based on evaluation metrics.
        
        Args:
            metrics: Performance metrics to analyze
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Performance-based recommendations
        if metrics.average_score < 10:
            recommendations.append("Consider increasing training duration or adjusting hyperparameters")
            recommendations.append("Model may benefit from more exploration during training")
        
        if metrics.consistency_score < 0.5:
            recommendations.append("High score variance detected - consider stabilizing training")
            recommendations.append("Review reward function and training stability")
        
        if metrics.efficiency_ratio < 0.1:
            recommendations.append("Low efficiency detected - model may be making unnecessary moves")
            recommendations.append("Consider optimizing pathfinding behavior")
        
        # Collision-based recommendations
        if metrics.collision_types.get('boundary', 0) > metrics.total_games * 0.3:
            recommendations.append("High boundary collision rate - improve boundary awareness")
        
        if metrics.collision_types.get('self', 0) > metrics.total_games * 0.4:
            recommendations.append("High self-collision rate - improve self-avoidance strategy")
        
        # Improvement-based recommendations
        if metrics.improvement_rate < 0:
            recommendations.append("Negative improvement trend detected - review training stability")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory across all metrics")
            recommendations.append("Consider fine-tuning for specific performance goals")
        
        return recommendations
    
    def save_metrics(self, metrics: PerformanceMetrics, 
                    filename: Optional[str] = None) -> str:
        """Save metrics to JSON file.
        
        Args:
            metrics: Performance metrics to save
            filename: Optional filename for the saved metrics
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_metrics_{timestamp}.json"
        
        filepath = self.eval_dir / filename
        
        # Add metadata
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'evaluator_version': '1.0',
                'total_games': metrics.total_games
            },
            'metrics': metrics.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {filepath}")
        return str(filepath)
    
    def load_metrics(self, filepath: str) -> PerformanceMetrics:
        """Load metrics from JSON file.
        
        Args:
            filepath: Path to the metrics file
            
        Returns:
            PerformanceMetrics object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics_data = data['metrics']
        
        # Reconstruct PerformanceMetrics object
        metrics = PerformanceMetrics()
        
        # Basic stats
        basic = metrics_data['basic_stats']
        metrics.total_games = basic['total_games']
        metrics.total_score = basic['total_score']
        metrics.average_score = basic['average_score']
        metrics.median_score = basic['median_score']
        metrics.max_score = basic['max_score']
        metrics.min_score = basic['min_score']
        metrics.std_deviation = basic['std_deviation']
        
        # Performance categories
        perf = metrics_data['performance_categories']
        metrics.games_over_10 = perf['games_over_10']
        metrics.games_over_20 = perf['games_over_20']
        metrics.games_over_30 = perf['games_over_30']
        metrics.games_over_50 = perf['games_over_50']
        
        # Efficiency metrics
        eff = metrics_data['efficiency_metrics']
        metrics.average_moves_per_game = eff['average_moves_per_game']
        metrics.average_moves_per_food = eff['average_moves_per_food']
        metrics.efficiency_ratio = eff['efficiency_ratio']
        
        # Time metrics
        time_m = metrics_data['time_metrics']
        metrics.total_evaluation_time = time_m['total_evaluation_time']
        metrics.average_game_duration = time_m['average_game_duration']
        
        # Advanced metrics
        adv = metrics_data['advanced_metrics']
        metrics.score_distribution = adv['score_distribution']
        metrics.performance_trend = adv['performance_trend']
        metrics.consistency_score = adv['consistency_score']
        metrics.improvement_rate = adv['improvement_rate']
        
        # Game metrics
        game_m = metrics_data['game_metrics']
        metrics.collision_types = game_m['collision_types']
        metrics.survival_rates = game_m['survival_rates']
        
        return metrics


def main():
    """Example usage of the evaluation system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Snake AI Model')
    parser.add_argument('--model', type=str, default='model/model.pth',
                       help='Path to the model file')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games to evaluate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    from logger import setup_logging
    setup_logging()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path=args.model)
        
        # Run evaluation
        print(f"Evaluating model: {args.model}")
        print(f"Number of games: {args.games}")
        
        metrics = evaluator.evaluate_model(num_games=args.games)
        
        # Generate dashboard
        evaluator.generate_dashboard(metrics)
        
        # Generate report
        report = evaluator.generate_report(metrics)
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Average Score: {metrics.average_score:.2f}")
        print(f"Max Score: {metrics.max_score}")
        print(f"Games ≥ 20: {metrics.games_over_20} ({metrics.games_over_20/metrics.total_games*100:.1f}%)")
        print(f"Consistency: {metrics.consistency_score:.3f}")
        
        # Save metrics
        evaluator.save_metrics(metrics)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())