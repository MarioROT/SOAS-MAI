
import torch.optim as optim
from environment import CleanupEnv
import torch
import torch.nn.functional as F
import numpy as np
from agent_architecture import AgentNetwork
import copy
import os
import pygame

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device used: {device}")

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()

# Initialize the environment
env = CleanupEnv()

# Training parameters
num_episodes = 10
steps_per_episode = 250 # Set a fixed number of steps per episode
learning_rate = 0.04 # 0.0004
gamma = 0.99  # Discount factor
epsilon = 1e-5
entropy_cost = 0.01

# Define the input shape and number of actions
input_shape = (3, 15, 15)  # 3 channels, 15x15 input size
num_actions = 8  

# Initialize the population
population_size = 50
population = [AgentNetwork(input_shape, num_actions).to(device) for _ in range(population_size)]
optimizers = [optim.RMSprop(agent.parameters(), lr=learning_rate, eps=epsilon) for agent in population]

def random_matchmaking(population):
    return np.random.choice(population, size=5, replace=False)

def compute_loss(agent, rewards, hx):
    # Example loss calculation: Mean squared error between predicted and actual rewards
    predicted_rewards = agent.value_head_extrinsic(hx)
    rewards_tensor = torch.tensor([rewards], dtype=torch.float32).unsqueeze(1).to(device) # Ensure shape is [1, 1]
    loss = F.mse_loss(predicted_rewards, rewards_tensor)
    return loss

def compute_fitnesses(population):
    return np.random.rand(len(population))

def evolve_population(population, fitnesses):
    top_indices = np.argsort(fitnesses)[-int(0.2 * len(fitnesses)):]  # Top 20%
    top_agents = [population[i] for i in top_indices]
    new_population = []
    for _ in range(len(population)):
        parent = np.random.choice(top_agents)
        child = copy.deepcopy(parent)
        mutate(child)
        new_population.append(child)
    return new_population

def mutate(agent):
    mutation_rate = 0.1
    with torch.no_grad():
        for param in agent.parameters():
            param.add_(mutation_rate * torch.randn(param.size()).to(device))

def compute_intrinsic_reward(agent, hx):
    # Example intrinsic reward based on the intrinsic value head
    intrinsic_reward = agent.value_head_intrinsic(hx)
    return intrinsic_reward.item()

# Training loop
for episode in range(num_episodes):
    if episode % 10 == 0:
        print(f"-----> Episode {episode}")
    # Matchmaking
    agents = random_matchmaking(population)
    
    # Reset the environment
    observations = env.reset()
    
    # Initialize LSTM states and last actions/rewards
    hx = [torch.zeros(1, 128, requires_grad=True).to(device) for _ in agents]
    cx = [torch.zeros(1, 128, requires_grad=True).to(device) for _ in agents]
    last_actions = [torch.zeros(1).to(device) for _ in agents]
    last_extrinsic_rewards = [torch.zeros(1).to(device) for _ in agents]
    last_intrinsic_rewards = [torch.zeros(1).to(device) for _ in agents]
    
    for step in range(steps_per_episode):
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()

        if step % 10 == 0:
            print(f"---> Step {step}")
        actions = []
        for idx, agent in enumerate(agents):
            obs = torch.tensor(observations[idx], dtype=torch.float32).unsqueeze(0).to(device)
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
            actions.append(action.item())
        
        # Step the environment
        observations, rewards, done, _ = env.step(actions)

        env.render()
        
        # Update agents' networks
        for idx, agent in enumerate(agents):
            optimizer = optimizers[idx]
            optimizer.zero_grad()
            loss = compute_loss(agent, rewards[idx], hx[idx])
            # loss.backward(retain_graph=step < steps_per_episode - 1)
            loss.backward(retain_graph=True)
            optimizer.step()
        
        # Update last actions and rewards
        for idx in range(len(agents)):
            last_actions[idx] = torch.tensor([actions[idx]], dtype=torch.float32).to(device)
            last_extrinsic_rewards[idx] = torch.tensor([rewards[idx]], dtype=torch.float32).to(device)
            last_intrinsic_rewards[idx] = torch.tensor([compute_intrinsic_reward(agent, hx[idx])], dtype=torch.float32).to(device)

        # Control the frame rate
        clock.tick(30)  # Limit to 30 frames per second

    # Evolutionary process
    fitnesses = compute_fitnesses(population)
    population = evolve_population(population, fitnesses)

import torch
import os

# Directory to save the models
save_dir = 'trained_models'
os.makedirs(save_dir, exist_ok=True)

# Save the model parameters
for idx, agent in enumerate(population):
    torch.save(agent.state_dict(), os.path.join(save_dir, f'agent_{idx}.pth'))

