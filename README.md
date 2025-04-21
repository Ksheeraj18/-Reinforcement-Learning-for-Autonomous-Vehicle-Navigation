README.md

# Autonomous Vehicle Obstacle Avoidance using Reinforcement Learning

## Project Overview
A PyTorch + PyGame implementation of a self-learning car that navigates through dynamically generated obstacles using Deep Q-Learning (DQN).

## Key Features
- 🚗 Vehicle control via neural network (steer left/right/straight)
- 🧠 Double DQN architecture with experience replay
- 🎮 Interactive GUI for real-time demonstration
- ⚡ GPU-accelerated training (CUDA support)

## File Structure
project/
├── models/ # Saved model weights
├── src/
│ ├── train.py # Training script
│ └── demo.py # Demonstration GUI
├── assets/ # Graphical assets
├── requirements.txt # Dependencies
└── README.md # This file


## Installation
```bash
git clone https://github.com/yourusername/rl-car-demo.git
cd rl-car-demo
pip install -r requirements.txt
Usage
```
1. Training the Model:

bash
python src/train.py
(Press Ctrl+C to stop training when rewards stabilize)

2. Running the Demo:

bash
python src/demo.py
(Click anywhere to add obstacles)

Technical Specifications
State Space: 5 dimensions (normalized positions + speed)

Action Space: 3 discrete actions (left/right/straight)

Reward Function:


reward = survival_bonus - center_penalty - collision_penalty
Neural Network: 3-layer MLP (128 units/layer)
