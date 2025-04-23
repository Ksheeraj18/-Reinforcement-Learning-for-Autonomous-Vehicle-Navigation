import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Modified GUI for local execution that uses matplotlib instead of pygame
class VehicleGUI:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
        # Colors
        self.WHITE = (1.0, 1.0, 1.0)
        self.BLACK = (0.0, 0.0, 0.0)
        self.RED = (1.0, 0.0, 0.0)
        self.GREEN = (0.0, 1.0, 0.0)
        self.BLUE = (0.0, 0.0, 1.0)
        
        # Setup the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.img = None
        
    def draw_environment(self):
        # Create a grid
        grid = np.ones((self.env.grid_size, self.env.grid_size, 3))  # White background
        
        # Draw obstacles
        for obs in self.env.obstacles:
            grid[obs[0], obs[1]] = self.BLACK  # Black obstacles
        
        # Draw goal
        grid[self.env.goal_pos[0], self.env.goal_pos[1]] = self.GREEN  # Green goal
        
        # Draw vehicle
        grid[self.env.vehicle_pos[0], self.env.vehicle_pos[1]] = self.BLUE  # Blue vehicle
        
        # Update the plot
        if self.img is None:
            self.img = self.ax.imshow(grid)
            self.ax.set_title('Autonomous Vehicle Navigation')
            self.ax.axis('off')
        else:
            self.img.set_data(grid)
            
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def run_demo(self, episodes=5):
        """Run a demonstration of the trained agent"""
        for e in range(episodes):
            state = self.env.reset()
            self.draw_environment()
            time.sleep(1)  # Pause to show initial state
            
            done = False
            steps = 0
            max_steps = 100  # Prevent infinite loops
            
            while not done and steps < max_steps:
                # Get action from agent
                action = self.agent.act(state, training=False)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                
                # Update display
                self.draw_environment()
                time.sleep(0.2)  # Control speed of simulation
                
                steps += 1
                
            # Show result
            if np.array_equal(self.env.vehicle_pos, self.env.goal_pos):
                print(f"Episode {e+1}: Goal reached in {steps} steps!")
            else:
                print(f"Episode {e+1}: Failed to reach goal.")
            
            time.sleep(1)
    
    def run_interactive(self):
        """
        Run in interactive mode with random obstacles added periodically
        """
        state = self.env.reset()
        self.draw_environment()
        
        steps = 0
        max_steps = 200
        
        while steps < max_steps:
            # Get action from agent
            action = self.agent.act(state, training=False)
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            
            # Occasionally add a random obstacle
            if steps % 10 == 0:
                x, y = np.random.randint(0, self.env.grid_size, 2)
                if self.env.add_obstacle(x, y):
                    print(f"Added obstacle at ({x}, {y})")
            
            # Update display
            self.draw_environment()
            time.sleep(0.2)  # Control speed of simulation
            
            steps += 1
            
            if done:
                if np.array_equal(self.env.vehicle_pos, self.env.goal_pos):
                    print("Goal reached!")
                else:
                    print("Hit an obstacle!")
                    
                # Reset after a short delay
                time.sleep(1)
                state = self.env.reset()
                steps = 0