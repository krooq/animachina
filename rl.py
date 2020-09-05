from abc import abstractclassmethod

import gym

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

def run_gym(env: str, agent: Agent, nb_eps: int, nb_timesteps: int):
    env = gym.make(env)
    best_reward = 0
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
                print("Episode {} finished after {} timesteps with episode reward {}".format(ep, dt + 1, episode_reward))
                break
        best_reward = max(episode_reward, best_reward)
    print("Session complete, best reward {}".format(best_reward))
    env.close()