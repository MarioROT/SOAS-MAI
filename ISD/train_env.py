
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gymnasium.spaces import Box, Discrete


import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from CleanupEnv import CleanupEnv

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[1]  # Adjusted to handle multi-agent observations

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
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[0][None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.shape) == 5:
            observations = observations[0]
        return self.linear(self.cnn(observations).view(observations.size(0), -1))

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256)
        )

# Create a DummyVecEnv for parallel environment
env = DummyVecEnv([lambda: CleanupEnv()])

# Initialize the PPO model with the custom policy
model = PPO(CustomActorCriticPolicy, env, verbose=1)

# Train the PPO model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_cleanup")

# Load the trained model
model = PPO.load("ppo_cleanup")

# Evaluate the trained model
obs = env.reset()

num_agents = 2  # Adjust based on your actual number of agents
for step in range(1000):
    actions = model.predict(obs, deterministic=True)[0]
    obs, rewards, dones, infos = env.step(actions)
    env.render()
