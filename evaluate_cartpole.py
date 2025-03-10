import gymnasium as gym
import torch
import numpy as np

from reinforce_cartpole import PolicyNetwork  # Assuming this is your policy class

def evaluate_policy(policy, env, n_episodes=100):
    successes = 0
    

    for episode in range(n_episodes):
        state , _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            
            action, _ = policy.act(state)
            
            # Take action in environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        # In CartPole, reaching 200 steps is considered a success
        if episode_reward >= 195:  # Using 195 as success threshold
            successes += 1
    
    success_rate = (successes / n_episodes) * 100
    return success_rate

def main():
    # Create environment
    env = gym.make('CartPole-v1')
    
    policy = PolicyNetwork()
    
    # Load the saved model
    policy.load_state_dict(torch.load('reinforce_cartpole.pth'))
    policy.eval()  # Set to evaluation mode
    
    # Evaluate policy
    success_rate = evaluate_policy(policy, env)
    print(f"Success rate over {100} episodes: {success_rate:.2f}%")
    
    env.close()

if __name__ == "__main__":
    main()