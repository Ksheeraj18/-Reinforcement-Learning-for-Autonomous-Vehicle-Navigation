import numpy as np
import gym
from gym import spaces

class VehicleEnv(gym.Env):
    def __init__(self, grid_size=50):
        super(VehicleEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.obstacles = []
        self.vehicle_pos = np.array([5, 5])
        self.goal_pos = np.array([grid_size-5, grid_size-5])
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: vehicle position (x,y), goal position (x,y), 
        # and obstacle information (flattened grid or distance sensors)
        self.observation_space = spaces.Box(
            low=0, high=grid_size, 
            shape=(2 + 2 + 8,),  # vehicle pos + goal pos + 8 distance sensors
            dtype=np.float32
        )
        
        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size))
        self.reset()
        
    def reset(self):
        # Reset vehicle position
        self.vehicle_pos = np.array([5, 5])
        
        # Clear grid
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Place goal
        self.grid[self.goal_pos[0]][self.goal_pos[1]] = 2
        
        # Place obstacles
        self.obstacles = []
        for _ in range(10):  # Random initial obstacles
            pos = np.random.randint(0, self.grid_size, size=2)
            if not np.array_equal(pos, self.vehicle_pos) and not np.array_equal(pos, self.goal_pos):
                self.obstacles.append(pos)
                self.grid[pos[0]][pos[1]] = 1
        
        return self._get_observation()
    
    def add_obstacle(self, x, y):
        """Add an obstacle at the specified position"""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            pos = np.array([x, y])
            if not np.array_equal(pos, self.vehicle_pos) and not np.array_equal(pos, self.goal_pos):
                self.obstacles.append(pos)
                self.grid[x][y] = 1
                return True
        return False
    
    def _get_observation(self):
        # Get distance to obstacles in 8 directions
        sensors = self._get_sensor_readings()
        
        # Combine vehicle position, goal position, and sensor readings
        obs = np.concatenate([
            self.vehicle_pos,
            self.goal_pos,
            sensors
        ]).astype(np.float32)
        
        return obs
    
    def _get_sensor_readings(self):
        # Directions: N, NE, E, SE, S, SW, W, NW
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        readings = []
        for dx, dy in directions:
            # Get distance to nearest obstacle in this direction
            distance = 1.0  # Default normalized distance
            for i in range(1, self.grid_size):
                x, y = self.vehicle_pos[0] + i*dx, self.vehicle_pos[1] + i*dy
                if (x < 0 or x >= self.grid_size or 
                    y < 0 or y >= self.grid_size or
                    self.grid[int(x)][int(y)] == 1):
                    distance = i / self.grid_size  # Normalize
                    break
            readings.append(distance)
            
        return np.array(readings)
    
    def step(self, action):
        # Move the vehicle based on action
        if action == 0:  # Up
            self.vehicle_pos[0] = max(0, self.vehicle_pos[0] - 1)
        elif action == 1:  # Right
            self.vehicle_pos[1] = min(self.grid_size - 1, self.vehicle_pos[1] + 1)
        elif action == 2:  # Down
            self.vehicle_pos[0] = min(self.grid_size - 1, self.vehicle_pos[0] + 1)
        elif action == 3:  # Left
            self.vehicle_pos[1] = max(0, self.vehicle_pos[1] - 1)
        
        # Check if vehicle hit an obstacle
        hit_obstacle = False
        for obs in self.obstacles:
            if np.array_equal(self.vehicle_pos, obs):
                hit_obstacle = True
                break
        
        # Check if vehicle reached the goal
        reached_goal = np.array_equal(self.vehicle_pos, self.goal_pos)
        
        # Calculate reward
        if hit_obstacle:
            reward = -10
            done = True
        elif reached_goal:
            reward = 100
            done = True
        else:
            # Reward based on distance to goal
            prev_dist = np.linalg.norm(np.array(self.vehicle_pos) - np.array(self.goal_pos))
            reward = -0.1  # Small penalty for each step to encourage efficiency
            done = False
        
        # Get observation
        observation = self._get_observation()
        
        return observation, reward, done, {}