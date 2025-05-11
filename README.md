# Reinforcement Learning Coursework

This repository contains implementations of various reinforcement learning algorithms applied to classic control and robotics tasks.

## Projects Overview

### 1. REINFORCE Algorithm Implementation
- Custom implementation of REINFORCE algorithm for CartPole-v1
- Features:
  - Policy Network with dropout for better generalization
  - Achieved 100% success rate in evaluation
  - Training visualization with matplotlib
  - Performance tracking and plotting

### 2. Stable-Baselines3 Implementations

#### CartPole with A2C
- Implementation using SB3's A2C algorithm
- Integration with Weights & Biases for experiment tracking
- Model published on HuggingFace Hub
- Training configuration:
  - 25,000 timesteps
  - MlpPolicy

#### Panda Robot Reaching Task
- DDPG implementation for PandaReachJointsDense-v3
- Features:
  - HER (Hindsight Experience Replay) buffer
  - Integration with panda_gym environment
  - Weights & Biases monitoring
  - Model evaluation in rendered environment

## Key Findings

1. REINFORCE Implementation:
   - Achieved optimal performance on CartPole
   - Demonstrated stable learning curve

2. DDPG on Panda Robot:
   - Best performance achieved with 5000 timesteps
   - Interesting observation: Performance degradation after 5000 timesteps
   - Possible overfitting with longer training periods

## Technologies Used

- PyTorch for custom implementations
- Stable-Baselines3 for standardized algorithms
- Weights & Biases for experiment tracking
- Gymnasium for environments
- panda_gym for robotics simulations
- HuggingFace Hub for model sharing

## Project Structure

```
.
├── reinforce_cartpole.py      # Custom REINFORCE implementation
├── evaluate_cartpole.py       # Evaluation script for REINFORCE
├── a2c_sb3_cartpole.py       # SB3 A2C implementation
├── ddpg_sb3_panda_reach.py   # DDPG implementation for Panda robot
├── a2c_sb3_panda_reach_test.py # Testing script for Panda environment
└── training_curve.png        # Learning curve visualization
```

## Results

The project achieved several key milestones:
- Successful implementation of REINFORCE with 100% evaluation success rate
- Working integration with modern RL tools (SB3, W&B)
- Interesting findings regarding training duration impact on performance
- Practical experience with both classic control and robotics environments

## Links

- [A2C CartPole Model on HuggingFace](https://huggingface.co/AxelLabrousse/a2c_cartpole)
- Training visualizations and detailed metrics available in W&B reports