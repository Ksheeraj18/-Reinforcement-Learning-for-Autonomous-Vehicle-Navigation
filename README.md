# Autonomous Vehicle Navigation using Deep Reinforcement Learning

This project implements an autonomous vehicle navigation system using Deep Q-Network (DQN) reinforcement learning. The system learns to navigate a vehicle from a starting point to a goal position while avoiding obstacles in a grid-based environment.

## Overview

The agent learns through reinforcement learning to make intelligent navigation decisions, receiving rewards for reaching the goal and penalties for colliding with obstacles or taking unnecessary steps. The project uses PyTorch for the neural network implementation and Matplotlib for visualization.

## Features

- Custom grid-based environment with adjustable complexity
- Deep Q-Network (DQN) agent with experience replay
- Real-time visualization using Matplotlib
- Multiple operation modes: training, demo, and interactive
- GPU acceleration support for faster training
- Configurable reward system and neural network architecture

## Project Structure

- `agent.py`: Implementation of the DQN agent
- `environment.py`: Definition of the navigation environment
- `gui.py`: Visualization interface
- `training.py`: Training loop logic
- `main.py`: Main entry point with argument parsing

## Requirements

See `requirements.txt` for a list of dependencies.

## Usage

### Training the Model

To train a new model:

```bash
python main.py --mode train --episodes 1000
```

This will train the agent for 1000 episodes and save the model as `final_model.pt`.

### Running a Demo

To run a demonstration using a trained model:

```bash
python main.py --mode demo --load_model final_model.pt
```

### Interactive Mode

To run in interactive mode where obstacles are randomly added during navigation:

```bash
python main.py --mode interactive --load_model final_model.pt
```

## Command Line Arguments

- `--mode`: Operating mode (train, demo, interactive)
- `--episodes`: Number of episodes for training
- `--load_model`: Path to a pre-trained model file

## How It Works

1. **Environment**: A grid environment with the vehicle, goal, and obstacles
2. **State**: Vehicle position, goal position, and distance sensor readings in 8 directions
3. **Actions**: Move up, right, down, or left
4. **Rewards**:
   - +100 for reaching the goal
   - -10 for hitting an obstacle
   - -0.1 for each step (encourages efficiency)

5. **Neural Network**: A 3-layer fully connected network that maps states to Q-values
6. **Learning Algorithm**: DQN with experience replay and epsilon-greedy exploration

## Training Results

The training script generates a graph showing episode scores and average scores over time, saved as `training_results.png`.

## License

[MIT License](LICENSE)
