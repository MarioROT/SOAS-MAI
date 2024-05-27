
import gym
from gym import spaces
import numpy as np
import pygame

class CleanupEnv(gym.Env):
    def __init__(self, spawn_waste_rate = None):
        super(CleanupEnv, self).__init__()
        self.action_space = spaces.Discrete(8)  # left, right, up, down, rotate left, rotate right, clean, tag
        self.observation_space = spaces.Box(low=0, high=255, shape=(15, 15, 3), dtype=np.uint8)
        
        # Define the grid size
        self.grid_size = (18, 25)
        
        # Define the aquifer region (where waste affects apple spawning)
        self.aquifer_region = [(i, j) for i in range(5) for j in range(self.grid_size[1])]
        
        # Initialize the grid, agents, apples, and waste
        self.grid = np.zeros(self.grid_size)
        self.agents = [{'pos': (0, 0), 'direction': 0} for _ in range(5)]  # Example initial positions
        self.apples = set()
        self.waste = set()
        self.spawn_waste_rate = spawn_waste_rate
        
        self.reset()

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_size[1] * 20, self.grid_size[0] * 20))
        pygame.display.set_caption("Cleanup Environment")
        self.clock = pygame.time.Clock()

    def reset(self):
        # Reset the grid
        self.grid = np.zeros(self.grid_size)
        
        # Randomly place agents
        for agent in self.agents:
            agent['pos'] = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
            agent['direction'] = np.random.randint(4)
            agent['reward'] = 0
        
        # Initialize apples and waste
        self.apples = set()
        self.waste = set()
        
        # No initial apples (as per the description)
        
        # Randomly place initial waste in the aquifer region, 70% to 90% dirty
        # num_initial_waste = int(np.random.uniform(0.7, 0.9) * len(self.aquifer_region))
        num_initial_waste = len(self.aquifer_region)
        initial_waste_positions = np.random.choice(len(self.aquifer_region), num_initial_waste, replace=False)
        for idx in initial_waste_positions:
            x, y = self.aquifer_region[idx]
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
            elif action[i] == 5:  # Rotate right
                agent['direction'] = (agent['direction'] + 1) % 4
            elif action[i] == 6:  # Clean
                if agent['pos'] in self.waste:
                    self.waste.remove(agent['pos'])
                    # reward += 1
            elif action[i] == 7:  # Tag
                reward -= 1  # Cost for tagging
                for j, other_agent in enumerate(self.agents):
                    if i != j and agent['pos'] == other_agent['pos']:
                        other_agent['reward'] -= 50  # Penalty for being tagged

            agent['reward'] += reward
            rewards.append(reward)
        
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

    def _update_grid(self):
        self.grid = np.zeros(self.grid_size)
        for x, y in self.apples:
            self.grid[x, y] = 1
        for x, y in self.waste:
            self.grid[x, y] = 2

    def _spawn_apples(self):
        # Determine the cleanliness of the aquifer
        cleanliness = 1 - (len(self.waste) / len(self.aquifer_region))
        
        # Set the spawn rate proportional to cleanliness
        spawn_rate = cleanliness
        
        # Spawn apples based on the spawn rate
        if np.random.rand() < spawn_rate:
            while True:
                x, y = np.random.randint(5, self.grid_size[0]), np.random.randint(self.grid_size[1])
                if (x, y) not in self.aquifer_region:
                    self.apples.add((x, y))
                    self.grid[x, y] = 1
                    break

    def _spawn_waste(self):
        if not self.spawn_waste_rate:
            # Determine the cleanliness of the aquifer
            cleanliness = 1 - (len(self.waste) / len(self.aquifer_region))
            
            # Set the spawn rate proportional to cleanliness
            spawn_rate = cleanliness
        else:
            spawn_rate = self.spawn_waste_rate
        
        # Spawn apples based on the spawn rate
        if np.random.rand() < spawn_rate:
            while True:
                x, y = np.random.randint(0, 5), np.random.randint(self.grid_size[1])
                if (x, y) in self.aquifer_region:
                    self.waste.add((x, y))
                    self.grid[x, y] = 2
                    break


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
        self.screen.fill((0, 0, 0))  # Fill screen with black

        # Draw aquifer region
        for x, y in self.aquifer_region:
            pygame.draw.rect(self.screen, (0, 0, 139), pygame.Rect(y * 20, x * 20, 20, 20))
        
        # Draw apples
        for x, y in self.apples:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(y * 20, x * 20, 20, 20))
        
        # Draw waste
        for x, y in self.waste:
            pygame.draw.rect(self.screen, (139, 69, 19), pygame.Rect(y * 20, x * 20, 20, 20))
        
        # Draw agents
        colors = [(255, 182, 193), (0, 255, 255), (128, 0, 128), (255, 255, 0), (0, 0, 255)]
        for idx, agent in enumerate(self.agents):
            x, y = agent['pos']
            pygame.draw.rect(self.screen, colors[idx], pygame.Rect(y * 20, x * 20, 20, 20))
        
        pygame.display.flip()
        self.clock.tick(20)  # Limit the frame rate to 10 FPS

    def close(self):
        pygame.quit()

