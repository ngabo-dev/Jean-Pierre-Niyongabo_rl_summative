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
   git clone <repository-url>
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

Modify the hyperparameters in the training scripts and run multiple times to compare performance. At least 10 runs per algorithm with different parameter combinations are recommended.

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