import numpy as np
import argparse
from environment import VehicleEnv
from agent import DQNAgent
from gui import VehicleGUI
from training import train_agent
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='RL for Autonomous Vehicle Navigation')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'demo', 'interactive'],
                        help='Mode to run: train, demo, or interactive')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model')
    args = parser.parse_args()
    
    # Create environment
    env = VehicleEnv(grid_size=50)
    
    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Load model if specified
    if args.load_model:
        try:
            agent.load(args.load_model)
            print(f"Loaded model from {args.load_model}")
        except FileNotFoundError:
            print(f"Model file {args.load_model} not found. Running with untrained agent.")
    
    if args.mode == 'train':
        # Train the agent
        print(f"Training agent for {args.episodes} episodes...")
        scores, avg_scores = train_agent(env, agent, episodes=args.episodes)
        
        # Plot training results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Episode Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.subplot(1, 2, 2)
        plt.plot(avg_scores)
        plt.title('Average Scores (last 100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Score')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
        
        print("Training complete. Model saved as 'final_model.pt'")
        
    elif args.mode == 'demo':
        # Run demonstration
        gui = VehicleGUI(env, agent)
        gui.run_demo(episodes=5)
        
    elif args.mode == 'interactive':
        # Run interactive mode
        gui = VehicleGUI(env, agent)
        gui.run_interactive()

if __name__ == "__main__":
    main()