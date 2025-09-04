import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from typing import Union, Tuple, Optional, Dict, Any
from logger import get_logger, log_exception

class Linear_QNet(nn.Module):
    """Deep Q-Network implementation using PyTorch.
    
    A neural network for Q-learning that maps states to Q-values for each possible action.
    Uses a simple two-layer fully connected architecture with ReLU activation.
    
    Args:
        input_size (int): Size of the input state vector
        hidden_size (int): Number of neurons in the hidden layer
        output_size (int): Number of possible actions (output Q-values)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Initialize logging
        self.logger = get_logger("model")
        self.logger.info(f"Q-Network initialized: {input_size} -> {hidden_size} -> {output_size}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization.
        
        Uses Xavier uniform initialization for weights and zero initialization for biases
        to ensure proper gradient flow during training.
        """
        try:
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)
            self.logger.debug("Weights initialized using Xavier uniform")
        except Exception as e:
            self.logger.error(f"Weight initialization failed: {e}")

    @log_exception(get_logger("model"), "Error in forward pass")
    def forward(self, x: Union[torch.Tensor, list, tuple]) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor or array-like object
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        try:
            # Ensure input is the right type and shape
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            # Handle single sample vs batch
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            # Return safe default output
            batch_size = x.size(0) if x.dim() > 1 else 1
            return torch.zeros(batch_size, 3)

    def save(self, file_name: str = 'model.pth') -> bool:
        """Save the model state to a file.
        
        Args:
            file_name: Name of the file to save the model to
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        """Save model with error handling and logging"""
        try:
            model_folder_path = './model'
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
                self.logger.debug(f"Created model directory: {model_folder_path}")

            file_path = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_path)
            self.logger.info(f"Model saved successfully: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load(self, file_path: str) -> bool:
        """Load model state from a file.
        
        Args:
            file_path: Path to the saved model file
            
        Returns:
            bool: True if load was successful, False otherwise
        """
        """Load model with error handling"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
            
            self.load_state_dict(torch.load(file_path))
            self.logger.info(f"Model loaded successfully: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

class QTrainer:
    """Q-Learning trainer for the Deep Q-Network.
    
    Handles the training process including loss calculation, backpropagation,
    and optimizer updates for the Q-learning algorithm.
    
    Args:
        model (Linear_QNet): The Q-network to train
        lr (float): Learning rate for the optimizer
        gamma (float): Discount factor for future rewards
    """
    
    def __init__(self, model: Linear_QNet, lr: float, gamma: float) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Initialize logging
        self.logger = get_logger("trainer")
        self.logger.info(f"Q-Trainer initialized: lr={lr}, gamma={gamma}")
        
        # Training statistics
        self.training_steps = 0
        self.total_loss = 0.0

    @log_exception(get_logger("trainer"), "Error in training step")
    def train_step(self, state: Union[list, torch.Tensor], action: Union[list, torch.Tensor], 
                   reward: Union[float, list, torch.Tensor], next_state: Union[list, torch.Tensor], 
                   done: Union[bool, list, tuple]) -> None:
        """Perform one training step using Q-learning algorithm.
        
        Args:
            state: Current state(s) of the environment
            action: Action(s) taken in the current state
            reward: Reward(s) received for the action
            next_state: Next state(s) after taking the action
            done: Whether the episode(s) is/are finished
        """
        try:
            # Convert inputs to tensors with proper error handling
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            
            # Handle single sample case
            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )

            # Validate tensor shapes
            if state.size(1) != 11:  # Expected input size
                self.logger.error(f"Invalid state shape: {state.shape}, expected (batch, 11)")
                return

            # 1: Get predicted Q values with current state
            pred = self.model(state)
            
            # 2: Calculate target Q values
            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    with torch.no_grad():  # Don't track gradients for target calculation
                        next_q_values = self.model(next_state[idx])
                        Q_new = reward[idx] + self.gamma * torch.max(next_q_values)

                # Update target for the action taken
                action_idx = torch.argmax(action[idx]).item()
                target[idx][action_idx] = Q_new

            # 3: Compute loss and update model
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"Invalid loss detected: {loss.item()}")
                return
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update training statistics
            self.training_steps += 1
            self.total_loss += loss.item()
            
            # Log training progress periodically
            if self.training_steps % 1000 == 0:
                avg_loss = self.total_loss / 1000
                self.logger.debug(f"Training step {self.training_steps}, avg loss: {avg_loss:.6f}")
                self.total_loss = 0.0
                
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            raise
    
    def get_training_stats(self) -> Dict[str, Union[int, float]]:
        """Get current training statistics.
        
        Returns:
            Dict containing training steps, learning rate, and gamma values
        """
        return {
            'training_steps': self.training_steps,
            'learning_rate': self.lr,
            'gamma': self.gamma
        }
         

