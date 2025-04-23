import numpy as np
from tqdm import tqdm
import time

def train_agent(env, agent, episodes=1000, batch_size=128, save_interval=100):
    """Train the agent using DQN with larger batch size for better GPU utilization"""
    scores = []
    avg_scores = []
    
    start_time = time.time()
    
    for e in tqdm(range(episodes), desc="Training Progress"):
        state = env.reset()
        
        done = False
        score = 0
        steps = 0
        max_steps = 500  # Prevent infinite loops
        
        while not done and steps < max_steps:
            # Get action from agent
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            steps += 1
            
            # Train agent with larger batch size for GPU efficiency
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
        # End of episode
        scores.append(score)
        
        # Calculate average score over last 100 episodes
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Print progress every 10 episodes
        if (e + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {e+1}/{episodes}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}, Time: {elapsed:.2f}s")
            start_time = time.time()
        
        # Save model periodically
        if (e + 1) % save_interval == 0:
            agent.save(f"model_checkpoint_{e+1}.pt")
            
    # Save final model
    agent.save("final_model.pt")
    
    return scores, avg_scores