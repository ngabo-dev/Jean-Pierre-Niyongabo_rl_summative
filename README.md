# FluentFusion RL Summative

Reinforcement Learning implementation for the FluentFusion language learning platform using Stable Baselines3 and custom Gymnasium environment.

## Project Overview

This project implements a custom RL environment based on the FluentFusion language learning platform. The agent learns to navigate through language lessons in a 2D grid, optimizing for lesson completion and proficiency improvement.

### Environment Description

- **State Space**: Agent position (x, y) and current proficiency level
- **Action Space**: 4 discrete actions (up, down, left, right)
- **Reward Structure**:
  - +1.0 for reaching the goal (final lesson)
  - Shaped reward: +0.01 * distance reduction towards goal
  - +0.1 for completing lessons (increasing proficiency)
  - -0.01 step penalty
- **Terminal Conditions**: Goal reached or max steps exceeded

### Agent-Environment Diagram

```
+-------------------+
| FluentFusion RL   |
| Environment       |
+-------------------+
         |
         | State: [x, y, proficiency]
         v
    +---------+
    |  Agent  | <--- Actions: [up, down, left, right]
    | (Blue   |
    | Circle) |
    +---------+
         |
         | Observations
         v
+-------------------+
| 2D Grid World     |
| - Yellow circles: |
|   Lesson positions|
| - Green square:   |
|   Goal position   |
| - Grid navigation |
+-------------------+
         |
         | Rewards
         v
    +---------+
    | Learning|
    | Algorithm|
    | (DQN/PPO/|
    | etc.)    |
    +---------+
```

### Implemented Algorithms

1. **DQN (Deep Q-Network)** - Value-based method
2. **REINFORCE** - Policy gradient method
3. **PPO (Proximal Policy Optimization)** - Policy gradient method
4. **A2C (Advantage Actor-Critic)** - Policy gradient method

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   └── rendering.py             # Pygame visualization
├── training/
│   ├── dqn_training.py          # DQN training script
│   └── pg_training.py           # Policy gradient training scripts
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for best model
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ngabo-dev/Jean-Pierre-Niyongabo_rl_summative.git
   cd fluentfusion-rl-summative
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

Train individual algorithms with customizable hyperparameters:

```bash
# Train DQN with custom parameters
python training/dqn_training.py --learning_rate 0.001 --buffer_size 50000 --batch_size 64 --total_timesteps 20000

# Train specific PG algorithm
python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --total_timesteps 50000

# Train all PG methods
python training/pg_training.py --algorithm all
```

### Running the Best Model

Run the demonstration with the best performing model:

```bash
python main.py
```

This will first show a random agent demonstration, then run the trained PPO model with visualization.

### Hyperparameter Tuning

At least 10 runs per algorithm with different parameter combinations are recommended. Below are suggested hyperparameter combinations for tuning:

#### DQN Hyperparameter Combinations

| Run | learning_rate | buffer_size | batch_size | gamma | exploration_fraction | total_timesteps |
|-----|---------------|-------------|------------|-------|----------------------|-----------------|
| 1   | 0.001         | 10000       | 32         | 0.99  | 0.1                  | 10000           |
| 2   | 0.0005        | 20000       | 64         | 0.95  | 0.2                  | 15000           |
| 3   | 0.002         | 5000        | 16         | 0.99  | 0.05                 | 8000            |
| 4   | 0.0015        | 15000       | 128        | 0.98  | 0.15                 | 12000           |
| 5   | 0.0008        | 25000       | 32         | 0.97  | 0.12                 | 20000           |
| 6   | 0.003         | 8000        | 64         | 0.99  | 0.08                 | 10000           |
| 7   | 0.0012        | 30000       | 256        | 0.96  | 0.18                 | 25000           |
| 8   | 0.0003        | 40000       | 32         | 0.95  | 0.25                 | 30000           |
| 9   | 0.0025        | 12000       | 128        | 0.98  | 0.06                 | 15000           |
| 10  | 0.0006        | 35000       | 64         | 0.97  | 0.14                 | 18000           |

Example command: `python training/dqn_training.py --learning_rate 0.001 --buffer_size 10000 --batch_size 32 --gamma 0.99 --exploration_fraction 0.1 --total_timesteps 10000`

#### Policy Gradient Methods Hyperparameter Combinations

For PPO and A2C (use --algorithm ppo or --algorithm a2c):

| Run | learning_rate | n_steps | batch_size | n_epochs | gamma | total_timesteps |
|-----|---------------|---------|------------|----------|-------|-----------------|
| 1   | 0.0003        | 2048    | 64         | 10       | 0.99  | 50000           |
| 2   | 0.0001        | 1024    | 32         | 5        | 0.95  | 30000           |
| 3   | 0.0005        | 4096    | 128        | 15       | 0.98  | 70000           |
| 4   | 0.0007        | 2048    | 256        | 8        | 0.97  | 40000           |
| 5   | 0.0002        | 1536    | 64         | 12       | 0.96  | 60000           |
| 6   | 0.001         | 3072    | 32         | 6        | 0.99  | 45000           |
| 7   | 0.0004        | 2048    | 128        | 10       | 0.98  | 55000           |
| 8   | 0.0006        | 1024    | 64         | 14       | 0.95  | 35000           |
| 9   | 0.0008        | 4096    | 256        | 9        | 0.97  | 65000           |
| 10  | 0.00025       | 2560    | 128        | 11       | 0.96  | 50000           |

Example command: `python training/pg_training.py --algorithm ppo --learning_rate 0.0003 --n_steps 2048 --batch_size 64 --n_epochs 10 --gamma 0.99 --total_timesteps 50000`

For REINFORCE (use --algorithm reinforce):

| Run | learning_rate | gamma | num_episodes |
|-----|---------------|-------|--------------|
| 1   | 0.001         | 0.99  | 1000         |
| 2   | 0.0005        | 0.95  | 1500         |
| 3   | 0.002         | 0.98  | 800          |
| 4   | 0.0015        | 0.97  | 1200         |
| 5   | 0.0008        | 0.96  | 2000         |
| 6   | 0.003         | 0.99  | 600          |
| 7   | 0.0012        | 0.98  | 1800         |
| 8   | 0.0003        | 0.95  | 2500         |
| 9   | 0.0025        | 0.97  | 900          |
| 10  | 0.0006        | 0.96  | 1600         |

Example command: `python training/pg_training.py --algorithm reinforce --reinforce_lr 0.001 --gamma 0.99 --reinforce_episodes 1000`

## Visualization

The environment uses Pygame for 2D visualization:
- Blue circle: Agent
- Yellow circles: Lesson positions
- Green square: Goal position
- Grid lines for navigation

## Evaluation Metrics

- Average reward per episode
- Steps per episode
- Convergence time
- Proficiency level achieved

## Dependencies

- gymnasium: RL environment framework
- stable-baselines3: RL algorithms
- torch: Neural network framework
- pygame: Visualization
- numpy: Numerical computations
- matplotlib: Plotting (for analysis)
- tensorboard: Training monitoring

## Results Documentation

Compare the performance of all four algorithms in terms of:
- Learning curves
- Final performance metrics
- Hyperparameter sensitivity
- Computational efficiency

Document findings in the final report with clear graphs and analysis.