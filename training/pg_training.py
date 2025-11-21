"""
Policy Gradient Training Scripts for FluentFusion RL Environment
Includes REINFORCE, PPO, and A2C implementations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
import os
from environment.custom_env import FluentFusionEnv

# REINFORCE Implementation
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def train_reinforce(learning_rate=1e-3, gamma=0.99, num_episodes=1000):
    """
    Train using REINFORCE algorithm.
    """
    env = FluentFusionEnv()
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_probs = policy(obs_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            obs, reward, done, _, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        # Compute loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {sum(rewards)}")

    # Save model
    os.makedirs("models/pg", exist_ok=True)
    torch.save(policy.state_dict(), "models/pg/reinforce_model.pth")
    print("REINFORCE model saved")

    return policy

def train_ppo(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, total_timesteps=50000):
    """
    Train using PPO algorithm with Stable Baselines3.
    """
    env = FluentFusionEnv()
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("models/pg/ppo_model")
    print("PPO model saved")
    return model

def train_a2c(learning_rate=7e-4, n_steps=5, gamma=0.99, total_timesteps=10000):
    """
    Train using A2C algorithm with Stable Baselines3.
    """
    env = FluentFusionEnv()
    model = A2C(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("models/pg/a2c_model")
    print("A2C model saved")
    return model

if __name__ == "__main__":
    # Train all models
    print("Training REINFORCE...")
    reinforce_policy = train_reinforce()

    print("Training PPO...")
    ppo_model = train_ppo()

    print("Training A2C...")
    a2c_model = train_a2c()

    print("All policy gradient models trained and saved.")