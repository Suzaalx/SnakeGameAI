import torch
import numpy as np
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet

class Agent:
    def __init__(self, model_path='model.pth'):
        self.model = Linear_QNet(11, 256, 3)
        self.load_model(model_path)

    def load_model(self, file_name):
        model_folder_path = './model'
        file_path = f"{model_folder_path}/{file_name}"
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        print(f"✅ Loaded model from {file_path}")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def play(model_name='model.pth'):
    agent = Agent(model_path=model_name)
    game = SnakeGameAI()

    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)
        reward, done, score = game.play_step(final_move)

        if done:
            print(f"🍎 Game Over! Final Score: {score}")
            game.reset()

if __name__ == "__main__":
    play('model.pth')  # You can change this to 'model_latest.pth' or any custom file