
import torch
import torch.nn as nn
import torch.optim as optim
from CleanupEnv import CleanupEnv

class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        n_input_channels = observation_space.shape[1]
        n_actions = action_space.nvec.sum()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[0][None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, observation_space):
        super(ValueNetwork, self).__init__()
        n_input_channels = observation_space.shape[1]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[0][None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x



import numpy as np
from torch.distributions import Categorical

def train(env, policy_net, value_net, policy_optimizer, value_optimizer, n_episodes=1000, gamma=0.99):
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32)
            action_probs = policy_net(obs)
            value = value_net(obs)
            
            dist = Categorical(logits=action_probs)
            action = dist.sample()
            
            obs, reward, terminated, truncated, info = env.step(action.numpy())
            done = terminated or truncated
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()
        
        advantage = returns - values
        
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        print(f"Episode {episode + 1}: Total Reward: {sum(rewards)}")

# Create environment
env = CleanupEnv()

# Initialize networks and optimizers
policy_net = PolicyNetwork(env.observation_space, env.action_space)
value_net = ValueNetwork(env.observation_space)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

# Train the agent
train(env, policy_net, value_net, policy_optimizer, value_optimizer)
