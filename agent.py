import torch
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Union, Dict, Any
from snake_gameai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from Helper import plot
from logger import get_logger, TrainingLogger, log_exception, PerformanceMonitor
import time
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    """Deep Q-Learning Agent for Snake Game AI.
    
    This agent uses a Deep Q-Network (DQN) to learn optimal actions for playing Snake.
    It implements experience replay, epsilon-greedy exploration, and target network updates.
    
    Attributes:
        n_game (int): Number of games played
        epsilon (float): Exploration rate for epsilon-greedy policy
        gamma (float): Discount factor for future rewards
        memory (deque): Experience replay buffer
        model (Linear_QNet): Neural network for Q-value approximation
        trainer (QTrainer): Training component for the neural network
        logger: Logger instance for debugging and monitoring
        training_logger: Specialized logger for training metrics
        performance_monitor: Monitor for performance tracking
    """

    def __init__(self) -> None:
        """Initialize the Agent with default hyperparameters and components."""
        self.n_game: int = 0
        self.epsilon: float = 0  # randomness
        self.gamma: float = 0.9  # discount rate
        self.memory: deque = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model: Linear_QNet = Linear_QNet(11, 256, 3)
        self.trainer: QTrainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Initialize logging
        self.logger = get_logger("agent")
        self.training_logger = TrainingLogger("main")
        self.performance_monitor = PerformanceMonitor("agent")
        
        self.logger.info("Agent initialized with DQN model")
        self.logger.debug(f"Model architecture: 11 -> 256 -> 3")
        self.logger.debug(f"Memory capacity: {MAX_MEMORY}")
        self.logger.debug(f"Learning rate: {LR}")
        self.logger.debug(f"Gamma (discount): {self.gamma}")
        self.record = 0

    @log_exception(get_logger("agent"), "Error in get_state")
    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        """Extract the current state representation from the game.
        
        The state is an 11-dimensional vector containing:
        - Danger detection (3 dimensions): straight, right, left relative to current direction
        - Current direction (4 dimensions): left, right, up, down (one-hot encoded)
        - Food location (4 dimensions): relative position of food to snake head
        
        Args:
            game (SnakeGameAI): The current game instance
            
        Returns:
            np.ndarray: 11-dimensional state vector as integers (0 or 1)
            
        Raises:
            Exception: If state extraction fails, returns zero vector
        """
        try:
            head = game.snake[0]
            point_l = Point(head.x - 20, head.y)
            point_r = Point(head.x + 20, head.y)
            point_u = Point(head.x, head.y - 20)
            point_d = Point(head.x, head.y + 20)
            
            dir_l = game.direction == Direction.LEFT
            dir_r = game.direction == Direction.RIGHT
            dir_u = game.direction == Direction.UP
            dir_d = game.direction == Direction.DOWN

            state = [
                # Danger straight
                (dir_r and game.is_collision(point_r)) or 
                (dir_l and game.is_collision(point_l)) or 
                (dir_u and game.is_collision(point_u)) or 
                (dir_d and game.is_collision(point_d)),

                # Danger right
                (dir_u and game.is_collision(point_r)) or 
                (dir_d and game.is_collision(point_l)) or 
                (dir_l and game.is_collision(point_u)) or 
                (dir_r and game.is_collision(point_d)),

                # Danger left
                (dir_d and game.is_collision(point_r)) or 
                (dir_u and game.is_collision(point_l)) or 
                (dir_r and game.is_collision(point_u)) or 
                (dir_l and game.is_collision(point_d)),
                
                # Move direction
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                
                # Food location 
                game.food.x < game.head.x,  # food left
                game.food.x > game.head.x,  # food right
                game.food.y < game.head.y,  # food up
                game.food.y > game.head.y  # food down
                ]

            return np.array(state, dtype=int)
        except Exception as e:
            self.logger.error(f"Failed to compute game state: {e}")
            # Return safe default state
            return np.zeros(11, dtype=int)

    def remember(self, state: np.ndarray, action: List[int], reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay memory.
        
        Args:
            state (np.ndarray): Current state
            action (List[int]): Action taken (one-hot encoded)
            reward (float): Reward received
            next_state (np.ndarray): Resulting state
            done (bool): Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from file.
        
        Args:
            model_path (str): Path to the model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.logger.info(f"Model successfully loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    @log_exception(get_logger("agent"), "Error in long memory training")
    def train_long_memory(self) -> None:
        """Train the model on a batch of experiences from replay memory.
        
        Uses experience replay to break correlation between consecutive experiences
        and improve learning stability. Samples a random batch from memory if enough
        experiences are available.
        
        Raises:
            Exception: If batch training fails
        """
        try:
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            else:
                mini_sample = self.memory

            if mini_sample:  # Only train if we have samples
                states, actions, rewards, next_states, dones = zip(*mini_sample)
                self.trainer.train_step(states, actions, rewards, next_states, dones)
                self.logger.debug(f"Long memory training completed with {len(mini_sample)} samples")
        except Exception as e:
            self.logger.error(f"Long memory training failed: {e}")
            raise

    @log_exception(get_logger("agent"), "Error in short memory training")
    def train_short_memory(self, state: np.ndarray, action: List[int], reward: float,
                          next_state: np.ndarray, done: bool) -> None:
        """Train the model on a single experience (online learning).
        
        Args:
            state (np.ndarray): Current state
            action (List[int]): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Resulting state
            done (bool): Whether episode ended
            
        Raises:
            Exception: If training fails
        """
        try:
            self.trainer.train_step(state, action, reward, next_state, done)
        except Exception as e:
            self.logger.error(f"Short memory training failed: {e}")
            raise

    def get_action(self, state: np.ndarray, train: bool = True) -> List[int]:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state representation
            train (bool): Whether in training mode (enables exploration)
            
        Returns:
            List[int]: One-hot encoded action [straight, right, left]
        """
        # Epsilon-greedy exploration (only during training)
        if train:
            self.epsilon = max(10, 80 - self.n_game)  # Decay epsilon with minimum
        else:
            self.epsilon = 0  # No exploration during inference
            
        final_move = [0, 0, 0]
        
        if train and random.randint(0, 200) < self.epsilon:
            # Random exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploit learned policy
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def load_model_from_folder(self, file_name: str = 'model.pth') -> None:
        """Load a model from the default model folder.
        
        Args:
            file_name (str): Name of the model file to load
        """
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()
            print(f"Loaded model from {file_path}")

def train() -> None:
    """Main training loop for the Snake AI agent.
    
    Runs continuous training episodes, logging performance metrics,
    saving models, and plotting results. Handles graceful shutdown
    on keyboard interrupt.
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()
    
    # Set initial game statistics for UI
    game.game_number = 0
    game.record = 0
    
    # Start training logging
    training_config = {
        'max_memory': MAX_MEMORY,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'model_architecture': '11->256->3',
        'gamma': agent.gamma
    }
    agent.training_logger.start_training(training_config)
    
    # Performance monitoring
    training_start_time = time.time()
    game_start_time = None
    
    try:
        while True:
            # Start game timing
            if game_start_time is None:
                game_start_time = time.time()
            
            # get old state
            agent.performance_monitor.start_timer("get_state")
            state_old = agent.get_state(game)
            agent.performance_monitor.end_timer("get_state", log_result=False)

            # get move
            agent.performance_monitor.start_timer("get_action")
            final_move = agent.get_action(state_old)
            agent.performance_monitor.end_timer("get_action", log_result=False)

            # Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Train short memory and remember
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Calculate game duration
                game_duration = time.time() - game_start_time if game_start_time else 0
                game_start_time = None
                
                # Game over logic
                game.reset()
                agent.n_game += 1
                
                agent.performance_monitor.start_timer("long_memory_training")
                agent.train_long_memory()
                agent.performance_monitor.end_timer("long_memory_training")

                # Check for new record
                is_new_record = score > record
                if is_new_record:
                    record = score
                    model_path = agent.model.save()
                    agent.training_logger.log_model_save(model_path, score)
                agent.model.save('model_latest.pth')

                # Log game result
                moves = len(game.snake) - 1  # Approximate moves based on snake length
                agent.training_logger.log_game_result(
                    agent.n_game, score, record, agent.epsilon, moves, game_duration
                )

                # Update game statistics for UI
                game.game_number = agent.n_game
                game.record = record

                print(f'Game: {agent.n_game}, Score: {score}, Record: {record}')

                # Enhanced plotting with additional metrics
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_game
                plot_mean_scores.append(mean_score)
                
                # Plot results
                try:
                    plot(plot_scores, plot_mean_scores, agent.n_game, record)
                except Exception as e:
                    agent.logger.warning(f"Plotting failed: {e}")
                
                # Log periodic statistics
                if agent.n_game % 50 == 0:
                    training_time = time.time() - training_start_time
                    agent.training_logger.log_training_stats(
                        agent.n_game, mean_score, record, training_time
                    )
                    agent.performance_monitor.log_memory_usage()
                
                # Reset game start time for next game
                game_start_time = time.time()
                
    except KeyboardInterrupt:
        agent.logger.info("Training interrupted by user")
        training_time = time.time() - training_start_time
        agent.training_logger.log_training_stats(
            agent.n_game, total_score / max(agent.n_game, 1), record, training_time
        )
        agent.training_logger.end_training()
    except Exception as e:
        agent.logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        agent.logger.info("Training session ended")



if __name__ == "__main__":
    train()