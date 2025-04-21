import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Car Training")
clock = pygame.time.Clock()

# Hyperparameters
BATCH_SIZE = 256
LR = 0.0001
GAMMA = 0.95
MEMORY_SIZE = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998
TARGET_UPDATE = 10

# Setup GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Environment Classes
class Car:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 50
        self.speed = 5
        self.width = 40
        self.height = 60

class Obstacle:
    def __init__(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = 0
        self.width = 30
        self.height = 30

def get_state(car, obstacles):
    state = [
        car.x / WIDTH,  # Normalized position
        car.speed / 10.0
    ]
    if obstacles:
        nearest = min(obstacles, key=lambda o: o.y)
        state.extend([
            nearest.x / WIDTH,
            (nearest.y - car.y) / HEIGHT,
            nearest.width / WIDTH
        ])
    else:
        state.extend([0.5, 1.0, 0.1])  # Default values
    return np.array(state, dtype=np.float32)

def train():
    # Initialize models
    policy_net = DQN(5, 3).to(device)
    target_net = DQN(5, 3).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    
    car = Car()
    obstacles = []
    episode = 0
    running = True
    
    while running:
        # Reset environment
        car.reset()
        obstacles = []
        episode_reward = 0
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
            
            if not running:
                break
            
            # Get current state
            state = get_state(car, obstacles)
            
            # Select action
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()
            
            # Execute action
            if action == 0:  # Left
                car.x = max(0, car.x - car.speed)
            elif action == 1:  # Right
                car.x = min(WIDTH - car.width, car.x + car.speed)
            
            # Spawn obstacles
            if random.random() < 0.03:
                obstacles.append(Obstacle())
            
            # Update obstacles
            new_obstacles = []
            collision = False
            for obstacle in obstacles:
                obstacle.y += 3
                if obstacle.y > HEIGHT:
                    continue
                new_obstacles.append(obstacle)
                
                # Check collision
                if (car.x < obstacle.x + obstacle.width and
                    car.x + car.width > obstacle.x and
                    car.y < obstacle.y + obstacle.height and
                    car.y + car.height > obstacle.y):
                    collision = True
            
            obstacles = new_obstacles
            
            # Calculate reward
            reward = 1.0  # Survival bonus
            reward -= abs(car.x - WIDTH/2) * 0.02  # Center penalty
            if collision:
                reward = -5.0
                done = True
            
            episode_reward += reward
            
            # Store experience
            next_state = get_state(car, obstacles)
            memory.append((state, action, reward, next_state, done))
            
            # Train if enough samples
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                
                states = torch.FloatTensor([s for s, _, _, _, _ in batch]).to(device)
                actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(device)
                rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(device)
                next_states = torch.FloatTensor([ns for _, _, _, ns, _ in batch]).to(device)
                dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(device)
                
                # Double DQN
                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1)
                    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1))
                    target_q = rewards + (1 - dones) * GAMMA * next_q.squeeze()
                
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            
            # Render
            screen.fill((255, 255, 255))
            pygame.draw.rect(screen, (0, 128, 255), (car.x, car.y, car.width, car.height))
            for obstacle in obstacles:
                pygame.draw.rect(screen, (255, 0, 0), (obstacle.x, obstacle.y, obstacle.width, obstacle.height))
            pygame.display.flip()
            clock.tick(60)
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        print(f"Episode {episode}: Reward {episode_reward:.1f}, Epsilon {epsilon:.2f}")
        episode += 1
        
        # Save model periodically
        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), "model.pth")
    
    pygame.quit()

if __name__ == "__main__":
    train()