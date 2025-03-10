import gymnasium as gym
import torch
from tqdm import tqdm
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(4, 128)
        self.fc2 = torch.nn.Linear(128, 2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def main():
    # Create the environment
    env = gym.make("CartPole-v1")
    
    # Hyperparameters
    nb_episodes = 500
    gamma = 0.99
    lr = 5e-3

    # Initialize the agent and the optimizer
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Initialize list to store episode returns
    episode_returns = []
    running_mean = 0

    with tqdm(range(nb_episodes), desc='Training', unit='episode') as pbar:
        for episode in pbar:
            # Reset the environment
            obs, info = env.reset()
            done = False

            # Reset the buffer
            rewards = []
            log_probs = []

            while not done:
                action, log_prob = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                log_probs.append(log_prob)

            # Calculate total episode return
            episode_return = sum(rewards)
            episode_returns.append(episode_return)
            
            # Update running mean
            running_mean = sum(episode_returns[-100:]) / min(len(episode_returns), 100)
            
            # Update progress bar
            pbar.set_postfix({
                'episode return': f'{episode_return:.2f}',
                'mean return (100)': f'{running_mean:.2f}'
            })

            # Continue with returns calculation and policy update
            returns = []
            R = 0
            for r in rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)

            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

    # Plot the results before closing
    plt.figure(figsize=(10,5))
    plt.plot(episode_returns)
    plt.title('Episode Returns over Time')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.show()

    # Save the model
    torch.save(policy.state_dict(), "reinforce_cartpole.pth")
    
    env.close()


if __name__ == "__main__":
    main()
