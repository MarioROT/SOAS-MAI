from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

class MultiAgentEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, num_agents):
        super().__init__(venv)
        self.num_agents = num_agents

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step(self.actions)
        return obs, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        return obs

# Wrap your multi-agent environment
# env = DummyVecEnv([lambda: CleanupEnv()])
# multi_agent_env = MultiAgentEnvWrapper(env, num_agents=2)
