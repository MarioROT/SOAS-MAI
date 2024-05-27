import numpy as np
import torch
import gymnasium as gym
from environment import CleanupEnv
from agent_architecture import AgentNetwork


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium.spaces import Box, Discrete


import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
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

        # Calculate the output size
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations).view(observations.size(0), -1))

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256)
        )

class CleanupEnvWithAgent(CleanupEnv):
    def __init__(self):
        super(CleanupEnvWithAgent, self).__init__()
        self.agent = AgentNetwork((3, 15, 15), 8)
        self.hx = torch.zeros((len(self.agents), 128))  # Initial hidden state of LSTM
        self.cx = torch.zeros((len(self.agents), 128))  # Initial cell state of LSTM
        self.decayed_rewards = [0] * len(self.agents)
        
        # Define the observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(3, 15, 15), dtype=np.uint8)
        self.action_space = Discrete(8)  # 8 possible actions

    def step(self, actions):
        rewards = []
        for i, agent in enumerate(self.agents):
            reward = 0
            action = actions[i]  # Get the action for the current agent
            # Update agent position based on action
            if action == 0:  # Move left
                agent['pos'] = (agent['pos'][0], max(agent['pos'][1] - 1, 0))
            elif action == 1:  # Move right
                agent['pos'] = (agent['pos'][0], min(agent['pos'][1] + 1, self.grid_size[1] - 1))
            elif action == 2:  # Move up
                agent['pos'] = (max(agent['pos'][0] - 1, 0), agent['pos'][1])
            elif action == 3:  # Move down
                agent['pos'] = (min(agent['pos'][0] + 1, self.grid_size[0] - 1), agent['pos'][1])
            elif action == 4:  # Rotate left
                agent['direction'] = (agent['direction'] - 1) % 4
            elif action == 5:  # Clean
                if agent['pos'] in self.waste:
                    self.waste.remove(agent['pos'])
                    reward += 10
            elif action == 6:  # Rotate right
                agent['direction'] = (agent['direction'] + 1) % 4
            elif action == 7:  # Tag
                reward -= 1  # Cost for tagging
                for j, other_agent in enumerate(self.agents):
                    if i != j and agent['pos'] == other_agent['pos']:
                        other_agent['reward'] -= 50  # Penalty for being tagged

            agent['reward'] += reward
            rewards.append(reward)

        # Update decayed rewards
        for i in range(len(self.agents)):
            self.decayed_rewards[i] = 0.975 * self.decayed_rewards[i] + rewards[i]
        
        # Check if agents are in apple field and collect apples
        for agent in self.agents:
            if agent['pos'] in self.apples:
                self.apples.remove(agent['pos'])
                rewards[self.agents.index(agent)] += 1
                agent['reward'] += 1

        # Update the grid
        self._update_grid()

        # Update apple spawning based on the cleanliness of the aquifer
        self._spawn_apples()

        # Check if episode is done
        done = False  # Example: Can be based on a condition such as all apples are collected

        # Return the observation, reward, done, and additional info
        return self._get_observation(), rewards, done, {}

    def _get_observation(self):
        # Resize or crop self.grid to fit into (15, 15)
        resized_grid = self.grid[:15, :15]  # Adjust this as needed
        
        # Ensure the observation is returned with the shape (3, 15, 15)
        observation = np.zeros((3, 15, 15), dtype=np.uint8)
        observation[0] = resized_grid
        observation[1] = resized_grid
        observation[2] = resized_grid
        return observation

    def get_agent_action(self, observation, last_action, last_extrinsic_reward, last_intrinsic_reward):
        # Ensure observation has 3 channels
        if len(observation.shape) == 2:
            observation = np.stack((observation,)*3, axis=-1)
        observation = observation.transpose((2, 0, 1))  # Convert to (C, H, W) format

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        last_action = torch.tensor([last_action], dtype=torch.float32).view(1, -1)
        last_extrinsic_reward = torch.tensor([last_extrinsic_reward], dtype=torch.float32).view(1, -1)
        last_intrinsic_reward = torch.tensor([last_intrinsic_reward], dtype=torch.float32).view(1, -1)

        policy, value_extrinsic, value_intrinsic, self.hx, self.cx = self.agent(
            observation, last_action, last_extrinsic_reward, last_intrinsic_reward, self.hx, self.cx
        )
        action = policy.argmax(dim=1).item()  # Select the action with the highest probability
        return action

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        super().reset()
        self.hx = torch.zeros((len(self.agents), 128))
        self.cx = torch.zeros((len(self.agents), 128))
        return self._get_observation(), {}

# Create a DummyVecEnv for parallel environment
env = DummyVecEnv([lambda: CleanupEnvWithAgent()])

# Initialize the PPO model with the custom policy
model = PPO(CustomActorCriticPolicy, env, verbose=1)

# Training loop modified to handle multi-agent actions
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_cleanup")

# Load the trained model
model = PPO.load("ppo_cleanup")

# Evaluate the trained model
obs, info = env.reset()

# Initialize last actions and rewards
num_agents = len(env.envs[0].agents)
last_action = [0] * num_agents
last_extrinsic_reward = [0.0] * num_agents
last_intrinsic_reward = [0.0] * num_agents

for step in range(1000):
    actions = []
    for i in range(num_agents):
        action = env.envs[0].get_agent_action(
            obs[i],
            last_action[i],
            last_extrinsic_reward[i],
            last_intrinsic_reward[i]
        )
        actions.append(action)
        
    obs, rewards, dones, info = env.step(actions)
    env.render()

    # Update last actions and rewards
    last_action = actions
    last_extrinsic_reward = rewards
    last_intrinsic_reward = [env.envs[0].get_intrinsic_reward(o).item() for o in obs]
