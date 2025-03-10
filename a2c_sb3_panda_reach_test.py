import gymnasium as gym
from stable_baselines3 import DDPG
import time

import panda_gym


# Create the environment
env = gym.make("PandaReachJointsDense-v3", render_mode="human")

# Load the trained model
model = DDPG.load("ddpg_reachjointsdense2.zip", env=env)

# Number of test episodes
n_episodes = 20

for episode in range(n_episodes):
    observation, info = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    
    print(f"Episode {episode + 1}")
    
    while not (done or truncated):
        # Get action from trained model
        action, _ = model.predict(observation, deterministic=True)
        
        # Execute action in environment
        observation, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        time.sleep(0.1)  # Add small delay to better visualize the movement
        
    print(f"Episode reward: {episode_reward}")

env.close()