from typing import Tuple
import cv2
import numpy as np
import torch

def scale_aspect(img: np.ndarray, min_size: int, scale: int):
    w           = scale * img.shape[1]
    h           = scale * img.shape[0]
    aspect      = w/h
    if w < h:
        w = min_size
        h = round(w/aspect)
    elif h < w:
        h = min_size
        w = round(h*aspect)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

def show(tensor: torch.Tensor, reshape: Tuple[int,int] = None, resize: Tuple[int,int] = None, label: str = ''):
    img = tensor.numpy()
    if reshape:
        img = img.reshape(reshape)
    if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(label, img)

def run_gym(env: gym.Env, agent: Agent, nb_eps: int, nb_timesteps: int):
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