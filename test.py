# import gym
from gym_examples.envs.grid_world import GridWorldEnv
import time
import torch

# env = gym.make('gym_examples/GridWorld-v0', size=10)
env = GridWorldEnv()
env.reset(torch.arange(env.num_envs, device='cuda'))

print(env)
print(env.observation_space)
print(env.action_space)

for _ in range(5):
    env.reset(torch.arange(env.num_envs, device='cuda'))
    env.render()
    time.sleep(1)