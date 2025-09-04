import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from typing import List, Tuple, Optional, Union
from logger import get_logger, log_exception
import math
pygame.init()

# Modern font setup
font_large = pygame.font.Font(None, 36)
font_medium = pygame.font.Font(None, 24)
font_small = pygame.font.Font(None, 18)

# Reset 
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision


class Direction(Enum):
    """Enumeration for movement directions in the game."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Point = namedtuple('Point', 'x y')

# Game settings
BLOCK_SIZE = 20
SPEED = 40

# Modern color palette
BACKGROUND = (20, 25, 40)  # Dark blue-gray
GRID_COLOR = (40, 45, 60)  # Slightly lighter for grid
SNAKE_HEAD = (46, 204, 113)  # Emerald green
SNAKE_BODY = (39, 174, 96)   # Darker green
SNAKE_OUTLINE = (27, 79, 114) # Dark blue outline
FOOD_COLOR = (231, 76, 60)    # Red
FOOD_GLOW = (192, 57, 43)     # Darker red for glow effect
TEXT_COLOR = (236, 240, 241)  # Light gray
SCORE_BG = (52, 73, 94)       # Dark blue-gray for score background
HIGH_SCORE_COLOR = (241, 196, 15)  # Gold for high score

# Legacy colors for compatibility
WHITE = TEXT_COLOR
RED = FOOD_COLOR
BLUE1 = SNAKE_HEAD
BLUE2 = SNAKE_BODY
BLACK = BACKGROUND

class SnakeGameAI:
    """Snake Game AI Environment.
    
    A Pygame-based Snake game implementation designed for AI training.
    Provides state representation, action handling, and reward calculation
    for reinforcement learning algorithms.
    
    Args:
        w (int): Width of the game window in pixels
        h (int): Height of the game window in pixels
    """
    
    def __init__(self, w: int = 640, h: int = 480) -> None:
        self.w=w
        self.h=h
        
        # Initialize logging
        self.logger = get_logger("game")
        self.logger.info(f"Initializing Snake Game AI: {w}x{h}")
        
        try:
            #init display
            self.display = pygame.display.set_mode((self.w,self.h))
            pygame.display.set_caption('Snake AI - Deep Q-Learning')
            self.clock = pygame.time.Clock()
            self.logger.debug("Pygame display initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame display: {e}")
            raise
        
        #init game state
        self.reset()
        
        # Game statistics for UI
        self.game_number = 0
        self.record = 0
        
        # Performance tracking
        self.frame_count = 0
        self.total_moves = 0
    @log_exception(get_logger("game"), "Error during game reset")
    def reset(self) -> None:
        """Reset the game to initial state.
        
        Initializes snake position, direction, score, and places food.
        Called at the start of each new game episode.
        """
        try:
            self.direction = Direction.RIGHT
            self.head = Point(self.w/2,self.h/2)
            self.snake = [self.head,
                          Point(self.head.x-BLOCK_SIZE,self.head.y),
                          Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
            self.score = 0
            self.food = None
            self._place__food()
            self.frame_iteration = 0
            self.total_moves = 0
            
            self.logger.debug(f"Game reset - Starting position: {self.head}")
        except Exception as e:
            self.logger.error(f"Game reset failed: {e}")
            raise
      

    def _place__food(self) -> None:
        """Place food at random location, avoiding snake body.
        
        Attempts to place food at a random valid position that doesn't
        overlap with the snake's body. Includes safety mechanism to prevent
        infinite loops in case the board is nearly full.
        """
        max_attempts = 100  # Prevent infinite recursion
        attempts = 0
        
        while attempts < max_attempts:
            try:
                x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
                y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
                self.food = Point(x,y)
                
                if self.food not in self.snake:
                    self.logger.debug(f"Food placed at: {self.food}")
                    return
                    
                attempts += 1
            except Exception as e:
                self.logger.error(f"Error placing food: {e}")
                break
        
        # Fallback: place food at a safe default location
        self.food = Point(BLOCK_SIZE, BLOCK_SIZE)
        self.logger.warning(f"Food placement fallback used after {attempts} attempts")


    @log_exception(get_logger("game"), "Error in game step")
    def play_step(self, action: List[int]) -> Tuple[int, bool, int]:
        """Execute one game step with the given action.
        
        Args:
            action: List of 3 integers representing [straight, right, left] action
            
        Returns:
            Tuple containing:
                - reward (int): Reward for this step (-10 for game over, +10 for food, 0 otherwise)
                - game_over (bool): Whether the game has ended
                - score (int): Current game score
        """
        try:
            self.frame_iteration+=1
            self.frame_count += 1
            self.total_moves += 1
            
            # 1. Collect the user input
            for event in pygame.event.get():
                if(event.type == pygame.QUIT):
                    self.logger.info("Game quit by user")
                    pygame.quit()
                    quit()
                
            # 2. Move
            self._move(action)
            self.snake.insert(0,self.head)

            # 3. Check if game Over
            reward = 0  # eat food: +10 , game over: -10 , else: 0
            game_over = False 
            
            # Check for collision or timeout
            if self.is_collision():
                game_over=True
                reward = -10
                self.logger.debug(f"Game over - Collision at {self.head}")
                return reward,game_over,self.score
            
            # Timeout check (prevent infinite loops)
            if self.frame_iteration > 100*len(self.snake):
                game_over=True
                reward = -10
                self.logger.debug(f"Game over - Timeout after {self.frame_iteration} frames")
                return reward,game_over,self.score
                
            # 4. Place new Food or just move
            if(self.head == self.food):
                self.score+=1
                reward=10
                self.logger.debug(f"Food eaten! Score: {self.score}")
                self._place__food()
                
            else:
                self.snake.pop()
            
            # 5. Update UI and clock
            self._update_ui()
            self.clock.tick(SPEED)
            # 6. Return game Over and Display Score
            
            return reward,game_over,self.score
            
        except Exception as e:
            self.logger.error(f"Error in play_step: {e}")
            # Return safe defaults
            return -10, True, self.score

    def _update_ui(self):
        # Fill background
        self.display.fill(BACKGROUND)
        
        # Draw subtle grid
        self._draw_grid()
        
        # Draw food with glow effect
        self._draw_food()
        
        # Draw snake with improved design
        self._draw_snake()
        
        # Draw modern UI elements
        self._draw_ui_elements()
        
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw a subtle grid background"""
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.w, y), 1)
    
    def _draw_food(self):
        """Draw food with glow effect"""
        # Glow effect (larger circle)
        glow_radius = BLOCK_SIZE // 2 + 4
        food_center = (self.food.x + BLOCK_SIZE//2, self.food.y + BLOCK_SIZE//2)
        pygame.draw.circle(self.display, FOOD_GLOW, food_center, glow_radius)
        
        # Main food (smaller circle)
        food_radius = BLOCK_SIZE // 2 - 2
        pygame.draw.circle(self.display, FOOD_COLOR, food_center, food_radius)
        
        # Highlight for 3D effect
        highlight_pos = (food_center[0] - 3, food_center[1] - 3)
        pygame.draw.circle(self.display, (255, 255, 255), highlight_pos, 3)
    
    def _draw_snake(self):
        """Draw snake with modern design"""
        for i, pt in enumerate(self.snake):
            if i == 0:  # Head
                # Head with rounded corners and outline
                head_rect = pygame.Rect(pt.x + 1, pt.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
                pygame.draw.rect(self.display, SNAKE_OUTLINE, head_rect, border_radius=8)
                
                inner_rect = pygame.Rect(pt.x + 3, pt.y + 3, BLOCK_SIZE - 6, BLOCK_SIZE - 6)
                pygame.draw.rect(self.display, SNAKE_HEAD, inner_rect, border_radius=6)
                
                # Eyes
                eye_size = 3
                left_eye = (pt.x + 6, pt.y + 6)
                right_eye = (pt.x + BLOCK_SIZE - 9, pt.y + 6)
                pygame.draw.circle(self.display, (255, 255, 255), left_eye, eye_size)
                pygame.draw.circle(self.display, (255, 255, 255), right_eye, eye_size)
                pygame.draw.circle(self.display, (0, 0, 0), left_eye, 1)
                pygame.draw.circle(self.display, (0, 0, 0), right_eye, 1)
            else:  # Body
                # Body segments with gradient effect
                body_rect = pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
                pygame.draw.rect(self.display, SNAKE_OUTLINE, body_rect, border_radius=4)
                
                inner_rect = pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8)
                pygame.draw.rect(self.display, SNAKE_BODY, inner_rect, border_radius=3)
    
    def _draw_ui_elements(self):
        """Draw modern UI elements"""
        # Score panel background
        score_panel = pygame.Rect(10, 10, 200, 60)
        pygame.draw.rect(self.display, SCORE_BG, score_panel, border_radius=10)
        pygame.draw.rect(self.display, SNAKE_OUTLINE, score_panel, 2, border_radius=10)
        
        # Score text
        score_text = font_medium.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.display.blit(score_text, (20, 20))
        
        # Game counter
        if hasattr(self, 'game_number'):
            game_text = font_small.render(f"Game: {self.game_number}", True, TEXT_COLOR)
            self.display.blit(game_text, (20, 45))
        
        # High score indicator (if available)
        if hasattr(self, 'record') and self.record > 0:
            record_text = font_small.render(f"Record: {self.record}", True, HIGH_SCORE_COLOR)
            self.display.blit(record_text, (120, 45))

    def _move(self, action: List[int]) -> None:
        """Move snake based on action [straight, right, left].
        
        Args:
            action: List of 3 integers where [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn
        """
        try:
            # Action
            # [1,0,0] -> Straight
            # [0,1,0] -> Right Turn 
            # [0,0,1] -> Left Turn

            clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
            idx = clock_wise.index(self.direction)
            
            # Validate action format
            if not isinstance(action, (list, np.ndarray)) or len(action) != 3:
                self.logger.error(f"Invalid action format: {action}")
                action = [1, 0, 0]  # Default to straight
            
            if np.array_equal(action,[1,0,0]):
                new_dir = clock_wise[idx]
            elif np.array_equal(action,[0,1,0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx] # right Turn
            else:
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx] # Left Turn
            self.direction = new_dir

            x = self.head.x
            y = self.head.y
            if(self.direction == Direction.RIGHT):
                x+=BLOCK_SIZE
            elif(self.direction == Direction.LEFT):
                x-=BLOCK_SIZE
            elif(self.direction == Direction.DOWN):
                y+=BLOCK_SIZE
            elif(self.direction == Direction.UP):
                y-=BLOCK_SIZE
            self.head = Point(x,y)
            
        except Exception as e:
            self.logger.error(f"Error in move: {e}")
            # Keep current position as fallback

    def is_collision(self, pt: Optional[Point] = None) -> bool:
        """Check if point collides with boundaries or snake body.
        
        Args:
            pt: Point to check for collision. If None, uses current head position.
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        if(pt is None):
            pt = self.head
        
        try:
            #hit boundary
            if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
                self.logger.debug(f"Boundary collision at {pt}")
                return True
            
            # hits itself
            if(pt in self.snake[1:]):
                self.logger.debug(f"Self collision at {pt}")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error in collision detection: {e}")
            return True  # Safe default: assume collision
