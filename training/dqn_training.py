"""
DQN Training Script for FluentFusion RL Environment
Uses Stable Baselines3 DQN algorithm for value-based reinforcement learning.
"""

import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from environment.custom_env import FluentFusionEnv

class TrainingCallback(BaseCallback):
    """
    Custom callback for logging training progress.
    """
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log every 1000 steps
        if self.n_calls % 1000 == 0:
            print(f"Step {self.n_calls}: Mean reward: {self.locals['infos'][-1] if self.locals['infos'] else 'N/A'}")
        return True

def train_dqn_model(learning_rate=1e-3, buffer_size=10000, batch_size=32, gamma=0.99,
                   exploration_fraction=0.1, total_timesteps=10000):
    """
    Train DQN model with specified hyperparameters.
    """
    # Create environment
    env = FluentFusionEnv()

    # Create DQN model
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    # Create callback
    callback = TrainingCallback()

    # Train the model
    print(f"Training DQN with lr={learning_rate}, buffer={buffer_size}, batch={batch_size}")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the model
    os.makedirs("models/dqn", exist_ok=True)
    model_path = f"models/dqn/dqn_lr{learning_rate}_buf{buffer_size}_batch{batch_size}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN model for FluentFusion RL Environment')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help='Exploration fraction')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total training timesteps')

    args = parser.parse_args()

    # Training run with command line args
    model = train_dqn_model(
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        total_timesteps=args.total_timesteps
    )

    # Test the trained model
    env = FluentFusionEnv()
    obs, _ = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            break