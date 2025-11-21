import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FluentFusionEnv(gym.Env):
    """
    Custom Gym environment for FluentFusion language learning platform.
    Simulates an agent navigating through language lessons in a 2D grid.
    Goal: Reach the final lesson position while maximizing learning rewards.
    """

    def __init__(self):
        super(FluentFusionEnv, self).__init__()

        # Action space: 4 discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Observation space: position (x, y) in grid, proficiency level
        self.observation_space = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)

        # Grid size
        self.grid_size = 10

        # Initialize state: [x, y, proficiency]
        self.state = np.array([0.0, 0.0, 0.0])

        # Max steps per episode
        self.max_steps = 100
        self.current_step = 0

        # Lesson positions (simplified)
        self.lesson_positions = [(i, j) for i in range(self.grid_size + 1) for j in range(self.grid_size + 1)]
        self.goal_position = (self.grid_size, self.grid_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0])
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        old_x, old_y, prof = self.state

        # Calculate old distance to goal
        old_distance = abs(self.grid_size - old_x) + abs(self.grid_size - old_y)

        if action == 0:  # up
            y = min(old_y + 1, self.grid_size)
            x = old_x
        elif action == 1:  # down
            y = max(old_y - 1, 0)
            x = old_x
        elif action == 2:  # left
            x = max(old_x - 1, 0)
            y = old_y
        elif action == 3:  # right
            x = min(old_x + 1, self.grid_size)
            y = old_y

        # Update proficiency based on lesson completion
        if (int(x), int(y)) in self.lesson_positions:
            prof = min(prof + 0.1, 10.0)  # Increase proficiency for completing lessons

        self.state = np.array([x, y, prof])

        # Calculate new distance to goal
        new_distance = abs(self.grid_size - x) + abs(self.grid_size - y)

        # Reward: shaped reward for progress towards goal, negative for steps
        reward = -0.01  # small step penalty
        reward += (old_distance - new_distance) * 0.01  # reward for reducing distance
        if (int(x), int(y)) == self.goal_position:
            reward += 1.0  # goal reward

        # Terminal conditions
        done = (int(x), int(y)) == self.goal_position or self.current_step >= self.max_steps
        self.current_step += 1

        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        # Basic text rendering
        x, y, prof = self.state
        print(f"Position: ({x:.1f}, {y:.1f}), Proficiency: {prof:.1f}")

    def close(self):
        pass