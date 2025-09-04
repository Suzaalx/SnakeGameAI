#!/usr/bin/env python3
"""
Snake Game AI Demo Script

This script demonstrates the trained Snake AI model in action.
It loads a pre-trained model and runs the game in demo mode with
enhanced visualization and statistics.

Usage:
    python demo.py [--model MODEL_PATH] [--games NUM_GAMES] [--speed SPEED]

Example:
    python demo.py --model model/model.pth --games 5 --speed 20
"""

import argparse
import sys
import os
import time
import pygame
from typing import Optional, List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from snake_gameai import SnakeGameAI
from agent import Agent
from config import config, get_color_scheme
from Helper import simple_plot
import matplotlib.pyplot as plt
from logger import get_logger, log_exception, setup_logging

class DemoRunner:
    """Handles running demo games with statistics and visualization"""
    
    def __init__(self, model_path: str, speed: int = 20, theme: str = 'dark'):
        # Initialize logging
        self.logger = get_logger("demo")
        self.logger.info(f"Initializing Demo Runner: {model_path}")
        
        self.model_path = model_path
        self.speed = speed
        self.theme = theme
        self.colors = get_color_scheme(theme)
        
        try:
            # Initialize agent and load model
            self.agent = Agent()
            self.load_model()
            
            # Statistics
            self.demo_scores: List[int] = []
            self.demo_stats: Dict = {
                'total_games': 0,
                'total_score': 0,
                'max_score': 0,
                'min_score': float('inf'),
                'avg_score': 0.0,
                'games_over_10': 0,
                'games_over_20': 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize demo runner: {e}")
            raise
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.agent.load_model(self.model_path)
                print(f"✅ Successfully loaded model from {self.model_path}")
                self.logger.info(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("Available models:")
                model_dir = os.path.dirname(self.model_path) or 'model'
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith('.pth'):
                            print(f"  - {os.path.join(model_dir, file)}")
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.logger.error(f"Error loading model: {e}")
            return False
    
    @log_exception(get_logger("demo"), "Error running demo game")
    def run_demo_game(self, game_number: int, show_game: bool = True) -> int:
        """Run a single demo game"""
        try:
            # Create game with custom speed
            game = SnakeGameAI()
            game.clock = pygame.time.Clock()
            
            # Set game display info
            game.game_number = game_number
            game.record = self.demo_stats['max_score']
            
            score = 0
            moves = 0
            max_moves = 1000  # Prevent infinite games
            start_time = time.time()
            
            # Disable epsilon (no random moves in demo)
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
            
            try:
                while moves < max_moves:
                    # Get current state
                    state = self.agent.get_state(game)
                    
                    # Validate state
                    if state is None or len(state) != 11:
                        self.logger.error(f"Invalid state received: {state}")
                        break
                    
                    # Get action from trained model
                    action = self.agent.get_action(state)
                    
                    # Validate action
                    if action is None or len(action) != 3:
                        self.logger.error(f"Invalid action received: {action}")
                        break
                    
                    # Perform action
                    reward, done, score = game.play_step(action)
                    moves += 1
                    
                    if show_game:
                        # Custom clock speed for demo
                        game.clock.tick(self.speed)
                    
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return score
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                # Pause/unpause
                                self._wait_for_space()
                            elif event.key == pygame.K_ESCAPE:
                                return score
                    
                    if done:
                        break
                
                if moves >= max_moves:
                    self.logger.warning(f"Game terminated due to move limit: {max_moves}")
                
                # Calculate game duration
                duration = time.time() - start_time
                
                # Print game summary
                print(f"🎮 Game {game_number}: Score {score}, Moves {moves}, Duration {duration:.1f}s")
                self.logger.debug(f"Game {game_number} completed with score {score}")
                
                return score
                
            finally:
                # Restore original epsilon
                self.agent.epsilon = original_epsilon
                
        except Exception as e:
            self.logger.error(f"Error in demo game {game_number}: {e}")
            return 0  # Return 0 score on error
    
    def _wait_for_space(self):
        """Wait for spacebar to continue (pause functionality)"""
        print("⏸️  Game paused. Press SPACE to continue or ESC to quit...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                        print("▶️  Game resumed")
                    elif event.key == pygame.K_ESCAPE:
                        waiting = False
                elif event.type == pygame.QUIT:
                    waiting = False
            time.sleep(0.1)
    
    def update_statistics(self, score: int):
        """Update demo statistics"""
        self.demo_scores.append(score)
        self.demo_stats['total_games'] += 1
        self.demo_stats['total_score'] += score
        self.demo_stats['max_score'] = max(self.demo_stats['max_score'], score)
        self.demo_stats['min_score'] = min(self.demo_stats['min_score'], score)
        self.demo_stats['avg_score'] = self.demo_stats['total_score'] / self.demo_stats['total_games']
        
        if score >= 10:
            self.demo_stats['games_over_10'] += 1
        if score >= 20:
            self.demo_stats['games_over_20'] += 1
    
    def print_final_statistics(self):
        """Print comprehensive demo statistics"""
        stats = self.demo_stats
        
        print("\n" + "="*60)
        print("🏆 DEMO RESULTS SUMMARY")
        print("="*60)
        print(f"📊 Total Games Played: {stats['total_games']}")
        print(f"🎯 Average Score: {stats['avg_score']:.2f}")
        print(f"🏅 Highest Score: {stats['max_score']}")
        print(f"📉 Lowest Score: {stats['min_score']}")
        print(f"📈 Total Score: {stats['total_score']}")
        print(f"🎖️  Games with Score ≥ 10: {stats['games_over_10']} ({stats['games_over_10']/stats['total_games']*100:.1f}%)")
        print(f"🏆 Games with Score ≥ 20: {stats['games_over_20']} ({stats['games_over_20']/stats['total_games']*100:.1f}%)")
        
        if len(self.demo_scores) > 1:
            import numpy as np
            std_dev = np.std(self.demo_scores)
            print(f"📊 Score Standard Deviation: {std_dev:.2f}")
        
        print("="*60)
    
    def plot_demo_results(self):
        """Plot demo results"""
        if not self.demo_scores:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Score progression
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(self.demo_scores) + 1), self.demo_scores, 'o-', linewidth=2, markersize=6)
        plt.title('Demo Game Scores', fontweight='bold')
        plt.xlabel('Game Number')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        plt.subplot(2, 2, 2)
        plt.hist(self.demo_scores, bins=max(5, len(set(self.demo_scores))), alpha=0.7, edgecolor='black')
        plt.title('Score Distribution', fontweight='bold')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative average
        plt.subplot(2, 2, 3)
        cumulative_avg = [sum(self.demo_scores[:i+1])/(i+1) for i in range(len(self.demo_scores))]
        plt.plot(range(1, len(cumulative_avg) + 1), cumulative_avg, 'g-', linewidth=2)
        plt.title('Cumulative Average Score', fontweight='bold')
        plt.xlabel('Game Number')
        plt.ylabel('Average Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics
        plt.subplot(2, 2, 4)
        metrics = ['Avg Score', 'Max Score', 'Games ≥10', 'Games ≥20']
        values = [
            self.demo_stats['avg_score'],
            self.demo_stats['max_score'],
            self.demo_stats['games_over_10'],
            self.demo_stats['games_over_20']
        ]
        bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        plt.title('Performance Metrics', fontweight='bold')
        plt.ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @log_exception(get_logger("demo"), "Error running demo")
    def run_demo(self, num_games: int = 5, show_plots: bool = True) -> None:
        """Run complete demo session"""
        try:
            print(f"🚀 Starting Snake AI Demo - {num_games} games")
            print(f"⚙️  Speed: {self.speed} FPS")
            print(f"🎨 Theme: {self.theme}")
            print("\n⌨️  Controls:")
            print("   SPACE - Pause/Resume")
            print("   ESC   - Skip current game")
            print("   Close window - Exit demo\n")
            
            self.logger.info(f"Starting demo: {num_games} games at {self.speed} FPS")
            
            try:
                for game_num in range(1, num_games + 1):
                    print(f"\n🎮 Starting Game {game_num}/{num_games}...")
                    
                    try:
                        score = self.run_demo_game(game_num)
                        self.update_statistics(score)
                        
                    except Exception as e:
                        self.logger.error(f"Error in game {game_num}: {e}")
                        print(f"❌ Game {game_num} failed: {e}")
                        continue
                    
                    # Brief pause between games
                    if game_num < num_games:
                        time.sleep(1)
                
                # Show final results
                self.print_final_statistics()
                
                if show_plots and len(self.demo_scores) > 0:
                    print("\n📊 Generating performance plots...")
                    self.plot_demo_results()
                    
            except KeyboardInterrupt:
                print("\n⏹️  Demo interrupted by user")
                self.logger.info("Demo interrupted by user")
            except Exception as e:
                print(f"\n❌ Demo error: {e}")
                self.logger.error(f"Demo failed: {e}")
                raise
            finally:
                pygame.quit()
                print("\n👋 Demo completed. Thank you for watching!")
                
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
            raise

def main():
    """Main demo function"""
    # Setup logging for demo
    setup_logging(log_level="INFO", log_file="demo.log")
    logger = get_logger("demo_main")
    
    parser = argparse.ArgumentParser(
        description="Snake Game AI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                                    # Run with default settings
  python demo.py --games 10 --speed 30             # 10 games at 30 FPS
  python demo.py --model model/model_latest.pth     # Use specific model
  python demo.py --theme neon --speed 15            # Neon theme, slower speed
        """
    )
    
    parser.add_argument('--model', '-m', 
                       default='model/model.pth',
                       help='Path to trained model file (default: model/model.pth)')
    parser.add_argument('--games', '-g', 
                       type=int, default=5,
                       help='Number of demo games to run (default: 5)')
    parser.add_argument('--speed', '-s', 
                       type=int, default=20,
                       help='Game speed in FPS (default: 20)')
    parser.add_argument('--theme', '-t', 
                       choices=['dark', 'light', 'neon'], default='dark',
                       help='Visual theme (default: dark)')
    parser.add_argument('--no-plots', 
                       action='store_true',
                       help='Disable performance plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(log_level="DEBUG")
    
    logger.info(f"Starting Snake AI Demo with args: {vars(args)}")
    
    try:
        # Validate arguments
        if args.games <= 0:
            raise ValueError("Number of games must be positive")
        
        if args.speed <= 0 or args.speed > 100:
            raise ValueError("Speed must be between 1 and 100 FPS")
        
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        
        # Initialize and run demo
        demo = DemoRunner(args.model, args.speed, args.theme)
        demo.run_demo(args.games, not args.no_plots)
        logger.info("Demo completed successfully")
        return 0
        
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(error_msg)
        print(f"❌ Error: {error_msg}")
        print("💡 Make sure you have trained a model first using: python agent.py")
        return 1
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Error running demo: {e}")
        print("💡 Check your model file and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())