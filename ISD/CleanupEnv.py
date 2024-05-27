import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, MultiDiscrete

class CleanupEnv(Env):
    def __init__(self):
        super(CleanupEnv, self).__init__()
        self.grid_size = (15, 15)
        self.num_agents = 2  # Define the number of agents
        self.agents = [{'pos': (0, 0), 'direction': 0, 'reward': 0} for _ in range(self.num_agents)]
        self.waste = set()
        self.apples = set()
        self.grid = np.zeros(self.grid_size)
        self.decayed_rewards = [0] * self.num_agents  # Initialize decayed rewards
        
        self.action_space = MultiDiscrete([8] * self.num_agents)  # 8 possible actions for each agent
        self.observation_space = Box(low=0, high=255, shape=(self.num_agents, 3, 15, 15), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.agents = [{'pos': (0, 0), 'direction': 0, 'reward': 0} for _ in range(self.num_agents)]
        self.hx = torch.zeros((self.num_agents, 128))
        self.cx = torch.zeros((self.num_agents, 128))
        return self._get_observation()

    def step(self, actions):
        rewards = []
        for i, agent in enumerate(self.agents):
            reward = 0
            action = actions[i]  # Get the action for the current agent
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

        # Check if episode is done
        terminated = False  # Example: Can be based on a condition such as all apples are collected
        truncated = False  # Example: Can be based on a condition such as time limit

        # Return the observation, reward, done, and additional info
        return self._get_observation(), sum(rewards), terminated, truncated, {}

    def _get_observation(self):
        # Create a combined observation for all agents
        observations = []
        for agent in self.agents:
            resized_grid = self.grid[:15, :15]  # Adjust this as needed
            observation = np.zeros((3, 15, 15), dtype=np.uint8)
            observation[0] = resized_grid
            observation[1] = resized_grid
            observation[2] = resized_grid
            observations.append(observation)
        return np.array(observations)

    def _update_grid(self):
        self.grid = np.zeros(self.grid_size)
        for agent in self.agents:
            self.grid[agent['pos'][0], agent['pos'][1]] = 1
        for apple in self.apples:
            self.grid[apple[0], apple[1]] = 2
        for waste in self.waste:
            self.grid[waste[0], waste[1]] = 3

    def _spawn_apples(self):
        cleanliness = 1 - (len(self.waste) / (self.grid_size[0] * self.grid_size[1]))
        spawn_rate = cleanliness
        if np.random.rand() < spawn_rate:
            while True:
                x, y = np.random.randint(8, self.grid_size[0]), np.random.randint(self.grid_size[1])
                if (x, y) not in self.waste:
                    self.apples.add((x, y))
                    self.grid[x, y] = 1
                    break
