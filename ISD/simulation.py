import numpy as np
from environment import CleanupEnv  # Assuming the environment is saved in CleanupEnv.py

# Initialize the environment
env = CleanupEnv()

# Number of episodes to simulate
num_episodes = 50

# Simulate multiple episodes
for episode in range(num_episodes):
    # Reset the environment at the beginning of each episode
    observations = env.reset()
    done = False
    total_rewards = [0] * len(env.agents)
    
    print(f"Episode {episode + 1}")
    
    # Run the simulation until the episode is done
    while not done:
        # Select actions for each agent randomly (as an example)
        actions = [env.action_space.sample() for _ in range(len(env.agents))]
        
        # Step the environment
        observations, rewards, done, info = env.step(actions)
        
        # Accumulate rewards
        total_rewards = [total_rewards[i] + rewards[i] for i in range(len(rewards))]
        
        # Optionally render the environment (this will print the grid to the console)
        env.render()
        
        # Pause for a short duration to observe the simulation (optional)
        # time.sleep(0.1)
    
    # Print the total rewards for each agent at the end of the episode
    print(f"Total rewards for Episode {episode + 1}: {total_rewards}\n")
