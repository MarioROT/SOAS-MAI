import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CleanupEnv(gym.Env):
    def __init__(self):
        super(CleanupEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # left, right, up, down, rotate, clean
        self.observation_space = spaces.Box(low=0, high=255, shape=(15, 15, 3), dtype=np.uint8)
        
        # Define the grid size
        self.grid_size = (25, 18)
        
        # Initialize the grid, agents, apples, and waste
        self.grid = np.zeros(self.grid_size)
        self.agents = [{'pos': (0, 0), 'direction': 0} for _ in range(5)]  # Example initial positions
        self.apples = set()
        self.waste = set()
        
        self.reset()

    def reset(self):
        # Reset the grid
        self.grid = np.zeros(self.grid_size)
        
        # Randomly place agents
        for agent in self.agents:
            agent['pos'] = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
            agent['direction'] = np.random.randint(4)
        
        # Initialize apples and waste
        self.apples = set()
        self.waste = set()
        
        # Randomly place initial apples
        for _ in range(10):
            x, y = np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])
            self.apples.add((x, y))
            self.grid[x, y] = 1  # Represent apples with 1
        
        # Randomly place initial waste
        for _ in range(5):
            x, y = np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])
            self.waste.add((x, y))
            self.grid[x, y] = 2  # Represent waste with 2

        # Return initial observation
        return self._get_observation()

    def step(self, action):
        rewards = []
        for i, agent in enumerate(self.agents):
            reward = 0
            # Update agent position based on action
            if action[i] == 0:  # Move left
                agent['pos'] = (agent['pos'][0], max(agent['pos'][1] - 1, 0))
            elif action[i] == 1:  # Move right
                agent['pos'] = (agent['pos'][0], min(agent['pos'][1] + 1, self.grid_size[1] - 1))
            elif action[i] == 2:  # Move up
                agent['pos'] = (max(agent['pos'][0] - 1, 0), agent['pos'][1])
            elif action[i] == 3:  # Move down
                agent['pos'] = (min(agent['pos'][0] + 1, self.grid_size[0] - 1), agent['pos'][1])
            elif action[i] == 4:  # Rotate left
                agent['direction'] = (agent['direction'] - 1) % 4
            elif action[i] == 5:  # Clean
                if agent['pos'] in self.waste:
                    self.waste.remove(agent['pos'])
                    reward += 10
            
            # Collect apples
            if agent['pos'] in self.apples:
                self.apples.remove(agent['pos'])
                reward += 1

            rewards.append(reward)
        
        # Update the grid
        self._update_grid()

        # Check if episode is done
        done = False  # Example: Can be based on a condition such as all apples are collected

        # Return the observation, reward, done, and additional info
        return self._get_observation(), rewards, done, {}

    def _update_grid(self):
        self.grid = np.zeros(self.grid_size)
        for x, y in self.apples:
            self.grid[x, y] = 1
        for x, y in self.waste:
            self.grid[x, y] = 2

    def _get_observation(self):
        # Create observations for each agent
        observations = []
        for agent in self.agents:
            x, y = agent['pos']
            obs = self.grid[max(0, x - 7):min(self.grid_size[0], x + 8), max(0, y - 7):min(self.grid_size[1], y + 8)]
            if obs.shape != (15, 15):
                pad_x = 15 - obs.shape[0]
                pad_y = 15 - obs.shape[1]
                obs = np.pad(obs, ((0, pad_x), (0, pad_y)), 'constant', constant_values=0)
            observations.append(obs)
        return observations

    def render(self, mode='human'):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])
        
        for x, y in self.apples:
            ax.add_patch(patches.Rectangle((y, x), 1, 1, color='green'))
        
        for x, y in self.waste:
            ax.add_patch(patches.Rectangle((y, x), 1, 1, color='brown'))
        
        colors = ['pink', 'cyan', 'purple', 'yellow', 'blue']
        for idx, agent in enumerate(self.agents):
            x, y = agent['pos']
            ax.add_patch(patches.Rectangle((y, x), 1, 1, color=colors[idx]))
        
        plt.gca().invert_yaxis()
        plt.show()
