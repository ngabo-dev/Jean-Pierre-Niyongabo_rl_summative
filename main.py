"""
Main entry point for running the best performing RL model in the FluentFusion environment.
Loads the trained model and demonstrates it with visualization.
"""

import time
from stable_baselines3 import PPO
from environment.custom_env import FluentFusionEnv
from environment.rendering import Renderer

def run_best_model(model_path="models/pg/ppo_model", episodes=5):
    """
    Run the best performing model (PPO) with visualization.
    """
    # Load environment and model
    env = FluentFusionEnv()
    model = PPO.load(model_path)

    # Initialize renderer
    renderer = Renderer()

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            # Render
            if not renderer.render(obs):
                break  # User closed window

            # Small delay for visualization
            time.sleep(0.1)

            print(f"Step {steps}: Action {action}, Reward {reward:.3f}, Total {total_reward:.3f}")

        print(f"Episode {episode + 1} finished: Total Reward {total_reward:.3f}, Steps {steps}")

    renderer.close()
    env.close()

def demonstrate_random_agent(episodes=1):
    """
    Demonstrate random actions in the environment (for comparison).
    """
    env = FluentFusionEnv()
    renderer = Renderer()

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Random Episode {episode + 1} ---")

        while not done:
            # Random action
            action = env.action_space.sample()

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            if not renderer.render(obs):
                break

            time.sleep(0.2)

            print(f"Step {steps}: Action {action}, Reward {reward:.3f}, Total {total_reward:.3f}")

        print(f"Random Episode {episode + 1} finished: Total Reward {total_reward:.3f}, Steps {steps}")

    renderer.close()
    env.close()

if __name__ == "__main__":
    print("FluentFusion RL Agent Demonstration")
    print("===================================")

    # First demonstrate random agent
    print("\nDemonstrating random agent...")
    demonstrate_random_agent()

    # Then run trained model
    print("\nRunning best performing model (PPO)...")
    try:
        run_best_model()
    except FileNotFoundError:
        print("Model not found. Please train the models first using training scripts.")