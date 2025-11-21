#!/bin/bash

# Script to run all hyperparameter tuning experiments
# Starting from DQN Run 4

echo "Starting Hyperparameter Tuning Experiments..."

# Activate virtual environment
source venv/bin/activate
export PYTHONPATH=.

# DQN Experiments (Runs 4-10)
echo "Running DQN Experiments..."

echo "DQN Run 4: lr=0.0015, buf=15000, batch=128, gamma=0.98, exp=0.15, steps=12000"
python training/dqn_training.py --learning_rate 0.0015 --buffer_size 15000 --batch_size 128 --gamma 0.98 --exploration_fraction 0.15 --total_timesteps 12000

echo "DQN Run 5: lr=0.0008, buf=25000, batch=32, gamma=0.97, exp=0.12, steps=20000"
python training/dqn_training.py --learning_rate 0.0008 --buffer_size 25000 --batch_size 32 --gamma 0.97 --exploration_fraction 0.12 --total_timesteps 20000

echo "DQN Run 6: lr=0.003, buf=8000, batch=64, gamma=0.99, exp=0.08, steps=10000"
python training/dqn_training.py --learning_rate 0.003 --buffer_size 8000 --batch_size 64 --gamma 0.99 --exploration_fraction 0.08 --total_timesteps 10000

echo "DQN Run 7: lr=0.0012, buf=30000, batch=256, gamma=0.96, exp=0.18, steps=25000"
python training/dqn_training.py --learning_rate 0.0012 --buffer_size 30000 --batch_size 256 --gamma 0.96 --exploration_fraction 0.18 --total_timesteps 25000

echo "DQN Run 8: lr=0.0003, buf=40000, batch=32, gamma=0.95, exp=0.25, steps=30000"
python training/dqn_training.py --learning_rate 0.0003 --buffer_size 40000 --batch_size 32 --gamma 0.95 --exploration_fraction 0.25 --total_timesteps 30000

echo "DQN Run 9: lr=0.0025, buf=12000, batch=128, gamma=0.98, exp=0.06, steps=15000"
python training/dqn_training.py --learning_rate 0.0025 --buffer_size 12000 --batch_size 128 --gamma 0.98 --exploration_fraction 0.06 --total_timesteps 15000

echo "DQN Run 10: lr=0.0006, buf=35000, batch=64, gamma=0.97, exp=0.14, steps=18000"
python training/dqn_training.py --learning_rate 0.0006 --buffer_size 35000 --batch_size 64 --gamma 0.97 --exploration_fraction 0.14 --total_timesteps 18000

# Policy Gradient Experiments
echo "Running Policy Gradient Experiments..."

# PPO Runs 1-10
for i in {1..10}
do
    case $i in
        1) params="--learning_rate 0.0003 --n_steps 2048 --batch_size 64 --n_epochs 10 --gamma 0.99 --total_timesteps 50000" ;;
        2) params="--learning_rate 0.0001 --n_steps 1024 --batch_size 32 --n_epochs 5 --gamma 0.95 --total_timesteps 30000" ;;
        3) params="--learning_rate 0.0005 --n_steps 4096 --batch_size 128 --n_epochs 15 --gamma 0.98 --total_timesteps 70000" ;;
        4) params="--learning_rate 0.0007 --n_steps 2048 --batch_size 256 --n_epochs 8 --gamma 0.97 --total_timesteps 40000" ;;
        5) params="--learning_rate 0.0002 --n_steps 1536 --batch_size 64 --n_epochs 12 --gamma 0.96 --total_timesteps 60000" ;;
        6) params="--learning_rate 0.001 --n_steps 3072 --batch_size 32 --n_epochs 6 --gamma 0.99 --total_timesteps 45000" ;;
        7) params="--learning_rate 0.0004 --n_steps 2048 --batch_size 128 --n_epochs 10 --gamma 0.98 --total_timesteps 55000" ;;
        8) params="--learning_rate 0.0006 --n_steps 1024 --batch_size 64 --n_epochs 14 --gamma 0.95 --total_timesteps 35000" ;;
        9) params="--learning_rate 0.0008 --n_steps 4096 --batch_size 256 --n_epochs 9 --gamma 0.97 --total_timesteps 65000" ;;
        10) params="--learning_rate 0.00025 --n_steps 2560 --batch_size 128 --n_epochs 11 --gamma 0.96 --total_timesteps 50000" ;;
    esac
    echo "PPO Run $i: $params"
    python training/pg_training.py --algorithm ppo $params
done

# A2C Runs 1-10
for i in {1..10}
do
    case $i in
        1) params="--learning_rate 0.0003 --n_steps 2048 --batch_size 64 --n_epochs 10 --gamma 0.99 --total_timesteps 50000" ;;
        2) params="--learning_rate 0.0001 --n_steps 1024 --batch_size 32 --n_epochs 5 --gamma 0.95 --total_timesteps 30000" ;;
        3) params="--learning_rate 0.0005 --n_steps 4096 --batch_size 128 --n_epochs 15 --gamma 0.98 --total_timesteps 70000" ;;
        4) params="--learning_rate 0.0007 --n_steps 2048 --batch_size 256 --n_epochs 8 --gamma 0.97 --total_timesteps 40000" ;;
        5) params="--learning_rate 0.0002 --n_steps 1536 --batch_size 64 --n_epochs 12 --gamma 0.96 --total_timesteps 60000" ;;
        6) params="--learning_rate 0.001 --n_steps 3072 --batch_size 32 --n_epochs 6 --gamma 0.99 --total_timesteps 45000" ;;
        7) params="--learning_rate 0.0004 --n_steps 2048 --batch_size 128 --n_epochs 10 --gamma 0.98 --total_timesteps 55000" ;;
        8) params="--learning_rate 0.0006 --n_steps 1024 --batch_size 64 --n_epochs 14 --gamma 0.95 --total_timesteps 35000" ;;
        9) params="--learning_rate 0.0008 --n_steps 4096 --batch_size 256 --n_epochs 9 --gamma 0.97 --total_timesteps 65000" ;;
        10) params="--learning_rate 0.00025 --n_steps 2560 --batch_size 128 --n_epochs 11 --gamma 0.96 --total_timesteps 50000" ;;
    esac
    echo "A2C Run $i: $params"
    python training/pg_training.py --algorithm a2c $params
done

# REINFORCE Runs 1-10
for i in {1..10}
do
    case $i in
        1) params="--reinforce_lr 0.001 --gamma 0.99 --reinforce_episodes 1000" ;;
        2) params="--reinforce_lr 0.0005 --gamma 0.95 --reinforce_episodes 1500" ;;
        3) params="--reinforce_lr 0.002 --gamma 0.98 --reinforce_episodes 800" ;;
        4) params="--reinforce_lr 0.0015 --gamma 0.97 --reinforce_episodes 1200" ;;
        5) params="--reinforce_lr 0.0008 --gamma 0.96 --reinforce_episodes 2000" ;;
        6) params="--reinforce_lr 0.003 --gamma 0.99 --reinforce_episodes 600" ;;
        7) params="--reinforce_lr 0.0012 --gamma 0.98 --reinforce_episodes 1800" ;;
        8) params="--reinforce_lr 0.0003 --gamma 0.95 --reinforce_episodes 2500" ;;
        9) params="--reinforce_lr 0.0025 --gamma 0.97 --reinforce_episodes 900" ;;
        10) params="--reinforce_lr 0.0006 --gamma 0.96 --reinforce_episodes 1600" ;;
    esac
    echo "REINFORCE Run $i: $params"
    python training/pg_training.py --algorithm reinforce $params
done

echo "All experiments completed!"