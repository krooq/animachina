from abc import abstractclassmethod

import gym
from gym.core import Env

class Sensor:
    @abstractclassmethod
    def obs(self, env):
        raise NotImplementedError
    
class Actuator:
    @abstractclassmethod
    def act(self) -> object:
        raise NotImplementedError

class Agent:    
    def __init__(self, observe, act, reward):
        self.observe = observe
        self.act = act
        self.reward = reward

def run_gym(env_name: str, agent: Agent, nb_eps: int, nb_timesteps: int):
    env = gym.make(env_name)
    best_reward = 0
    total_reward = 0
    # Start training regime
    for ep in range(nb_eps):
        observation = env.reset()
        episode_reward = 0
        # Start training episode
        for dt in range(nb_timesteps):
            env.render()
            agent.observe(observation)
            action = agent.act()
            observation, reward, done, info = env.step(action)
            agent.reward(reward)
            episode_reward += reward
            if done or dt == nb_timesteps:
                print("Episode finished [ep={}, ts={}, ep_r={}, avg_r={}]".format(ep, dt + 1, episode_reward, total_reward/(ep +1)))
                break
        best_reward = max(episode_reward, best_reward)
        total_reward += episode_reward
    print("Session complete, best reward {}".format(best_reward))
    env.close()