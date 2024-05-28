import pygame
import torch
import numpy as np
from environment import CleanupEnv
from agent_architecture import AgentNetwork
import os

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()

# Initialize the environment
env = CleanupEnv()

# Simulation parameters
steps_per_episode = 1000  # Set a fixed number of steps per episode

# Directory where the models are saved
load_dir = 'trained_models'

# Define the input shape and number of actions
input_shape = (3, 15, 15)  # 3 channels, 15x15 input size
num_actions = 8  # Example number of actions, adjust as necessary

# Load the model parameters
population_size = 50
population = [AgentNetwork(input_shape, num_actions) for _ in range(population_size)]
for idx, agent in enumerate(population):
    agent.load_state_dict(torch.load(os.path.join(load_dir, f'agent_{idx}.pth')))

def random_matchmaking(population):
    return np.random.choice(population, size=5, replace=False)

def compute_intrinsic_reward(agent, hx):
    # Example intrinsic reward based on the intrinsic value head
    intrinsic_reward = agent.value_head_intrinsic(hx)
    return intrinsic_reward.item()

actions_map = {0:"M Left", 1:"M Right", 2:"M UP", 3:"M Down", 4:"R Left", 5:"R Right", 6:"Clean", 7:"Tag"}
# Matchmaking
agents = random_matchmaking(population)

# Reset the environment
observations = env.reset()

# Initialize LSTM states and last actions/rewards
hx = [torch.zeros(1, 128) for _ in agents]
cx = [torch.zeros(1, 128) for _ in agents]
last_actions = [torch.zeros(1) for _ in agents]
last_extrinsic_rewards = [torch.zeros(1) for _ in agents]
last_intrinsic_rewards = [torch.zeros(1) for _ in agents]

for step in range(steps_per_episode):
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            exit()
    
    actions_log = f""
    actions = []
    for idx, agent in enumerate(agents):
        obs = torch.tensor(observations[idx], dtype=torch.float32).unsqueeze(0)
        if obs.size(0) == 1:  # If the observation has 1 channel, convert it to 3 channels
            obs = obs.repeat(1, 3, 1, 1)
        action_probs, hx[idx], cx[idx] = agent(
            obs, 
            last_actions[idx], 
            last_extrinsic_rewards[idx], 
            last_intrinsic_rewards[idx], 
            hx[idx], 
            cx[idx]
        )
        action = torch.multinomial(torch.nn.functional.softmax(action_probs, dim=-1), 1)
        actions_log += f"A[{idx}]={actions_map[action[0][0].numpy().tolist()]} | "
        actions.append(action.item())

    print(actions_log)
    
    # Step the environment
    observations, rewards, done, _ = env.step(actions)
    
    # Render the environment
    env.render()
    
    # Update last actions and rewards
    for idx in range(len(agents)):
        last_actions[idx] = torch.tensor([actions[idx]], dtype=torch.float32)
        last_extrinsic_rewards[idx] = torch.tensor([rewards[idx]], dtype=torch.float32)
        last_intrinsic_rewards[idx] = torch.tensor([compute_intrinsic_reward(agent, hx[idx])], dtype=torch.float32)

    # Control the frame rate
    clock.tick(30)  # Limit to 30 frames per second

# Close the environment
env.close()
pygame.quit()

