import numpy as np
import pygame
import torch
from environment import CleanupEnv
from agent_architecture import AgentNetwork

# Assuming the CleanupEnv class and AgentNetwork class are already defined

class CleanupEnvWithAgent(CleanupEnv):
    def __init__(self):
        super(CleanupEnvWithAgent, self).__init__()
        self.agent = AgentNetwork((3, 15, 15), 8)
        self.hx = torch.zeros(1, 128)  # Initial hidden state of LSTM
        self.cx = torch.zeros(1, 128)  # Initial cell state of LSTM
        self.decayed_rewards = [0] * len(self.agents)

    def step(self, actions):
        rewards = []
        for i, agent in enumerate(self.agents):
            reward = 0
            # Update agent position based on action
            if actions[i] == 0:  # Move left
                agent['pos'] = (agent['pos'][0], max(agent['pos'][1] - 1, 0))
            elif actions[i] == 1:  # Move right
                agent['pos'] = (agent['pos'][0], min(agent['pos'][1] + 1, self.grid_size[1] - 1))
            elif actions[i] == 2:  # Move up
                agent['pos'] = (max(agent['pos'][0] - 1, 0), agent['pos'][1])
            elif actions[i] == 3:  # Move down
                agent['pos'] = (min(agent['pos'][0] + 1, self.grid_size[0] - 1), agent['pos'][1])
            elif actions[i] == 4:  # Rotate left
                agent['direction'] = (agent['direction'] - 1) % 4
            elif actions[i] == 5:  # Rotate right
                agent['direction'] = (agent['direction'] + 1) % 4
            elif actions[i] == 6:  # Clean
                if agent['pos'] in self.waste:
                    self.waste.remove(agent['pos'])
                    # reward += 1
            elif actions[i] == 7:  # Tag
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

        # Update waste spawning based on the cleanliness of the aquifer
        self._spawn_waste()

        # Check if episode is done
        done = False  # Example: Can be based on a condition such as all apples are collected

        # Return the observation, reward, done, and additional info
        return self._get_observation(), rewards, done, {}

    def get_agent_action(self, observation, last_action, last_extrinsic_reward, last_intrinsic_reward):
        # Ensure observation has 3 channels
        if len(observation.shape) == 2:
            observation = np.stack((observation,)*3, axis=-1)
            # observation = np.stack((observation,), axis=-1)
        observation = observation.transpose((2, 0, 1))  # Convert to (C, H, W) format

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        last_action = torch.tensor([last_action], dtype=torch.float32).view(1, -1)
        last_extrinsic_reward = torch.tensor([last_extrinsic_reward], dtype=torch.float32).view(1, -1)
        last_intrinsic_reward = torch.tensor([last_intrinsic_reward], dtype=torch.float32).view(1, -1)

        policy, value_extrinsic, value_intrinsic, self.hx, self.cx = self.agent(
            observation, last_action, last_extrinsic_reward, last_intrinsic_reward, self.hx, self.cx
        )
        action = policy.argmax().item()  # Select the action with the highest probability
        return action

# Example usage
env = CleanupEnvWithAgent()

# Number of episodes to simulate
num_episodes = 5

# Simulate multiple episodes
for episode in range(num_episodes):
    observations = env.reset()
    done = False
    total_rewards = [0] * len(env.agents)
    last_action = [0] * len(env.agents)
    last_extrinsic_reward = [0] * len(env.agents)
    last_intrinsic_reward = [0] * len(env.agents)
    
    print(f"Episode {episode + 1}")
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        actions = []
        for i, observation in enumerate(observations):
            action = env.get_agent_action(
                observation,
                last_action[i],
                last_extrinsic_reward[i],
                last_intrinsic_reward[i]
            )
            actions.append(action)
        
        observations, rewards, done, info = env.step(actions)
        
        reward_log = f""
        for i, reward in enumerate(rewards):
            reward_log += f"| A[{i}]={reward} "

        print(reward_log)

        
        total_rewards = [total_rewards[i] + rewards[i] for i in range(len(rewards))]
        last_action = actions
        last_extrinsic_reward = rewards
        last_intrinsic_reward = [env.agent.get_intrinsic_reward(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item() for obs in observations]
        
        env.render()
    
    print(f"Total rewards for Episode {episode + 1}: {total_rewards}\n")

env.close()
