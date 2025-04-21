import pygame
import random
import torch
import numpy as np

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Car Demo")
clock = pygame.time.Clock()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(torch.nn.Module):
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

model = DQN(5, 3).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

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
        car.x / WIDTH,
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
        state.extend([0.5, 1.0, 0.1])
    return np.array(state, dtype=np.float32)

def main():
    car = Car()
    obstacles = []
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                obstacles.append(Obstacle())
        
        # Get state and make decision
        state = get_state(car, obstacles)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = model(state_tensor).argmax().item()
        
        # Execute action
        if action == 0:
            car.x = max(0, car.x - car.speed)
        elif action == 1:
            car.x = min(WIDTH - car.width, car.x + car.speed)
        
        # Update obstacles
        new_obstacles = []
        for obstacle in obstacles:
            obstacle.y += 3
            if obstacle.y <= HEIGHT:
                new_obstacles.append(obstacle)
        obstacles = new_obstacles
        
        # Spawn new obstacles occasionally
        if random.random() < 0.02:
            obstacles.append(Obstacle())
        
        # Render
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 128, 255), (car.x, car.y, car.width, car.height))
        for obstacle in obstacles:
            pygame.draw.rect(screen, (255, 0, 0), (obstacle.x, obstacle.y, obstacle.width, obstacle.height))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()