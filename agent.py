import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Check if CUDA is available and force GPU usage if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Set PyTorch to use the GPU
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU instead.")

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Increased network size for better learning
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Increased memory size for GPU
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Create model and move to GPU if available
        self.model = DQNModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Print model device to confirm
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Process entire batch at once for GPU efficiency
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([s[0] for s in minibatch]).to(device)
        actions = torch.LongTensor([s[1] for s in minibatch]).to(device)
        rewards = torch.FloatTensor([s[2] for s in minibatch]).to(device)
        next_states = torch.FloatTensor([s[3] for s in minibatch]).to(device)
        dones = torch.FloatTensor([s[4] for s in minibatch]).to(device)
        
        # Current Q values
        curr_q = self.model(states)
        curr_q_values = curr_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        next_q_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_q_values = torch.max(self.model(next_states), dim=1)[0]
        
        # Target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(curr_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name):
        torch.save(self.model.state_dict(), name)
    
    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))
        self.model.eval()