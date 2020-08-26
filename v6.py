# BindsNET is great in that it utilizes Pytorch and has a nice API
# but the clean API means it isn't very transparent
# the goal of this version is to improve v4 (DIY) with ideas from v5 (BindsNET)
# remember, we are not trying to create an API, just a powerful generic SNN
from typing import Callable, Dict
from bindsnet.learning.learning import LearningRule
import torch as torch
import cv2
import numpy as np
from torch import tensor
from util import *

epsilon         = 0.05
baseline        = 0.10
threshold       = 0.50
amplitude       = 0.10
stability       = 0.99
reuptake        = 0.000
plasticity      = 0.005
refractory      = epsilon

Id = int

class Node:
    def __init__(self, id: Id, label: str = None) -> None:
        self.id = id
        self.label = label

class Layer(Node):
    def __init__(self, id: Id, neurons: torch.Tensor) -> None:
        super().__init__(id, "layer_{}".format(id))
        self.neurons = neurons

class Connection(Node):
    def __init__(self, id: Id, source: Layer, target: Layer, weight: torch.Tensor, update) -> None:
        super().__init__(id, "connection_{} [{} to {}]".format(id, source.label, target.label))
        self.source = source
        self.target = target
        self.weight = weight
        self.update = update

class Net:
    def __init__(self):
        self.layers         : Dict[Id, Layer]       = {}
        self.connections    : Dict[Id, Connection]  = {}

    def add(self, n: int = 1) -> Layer:
        ''' Adds a new layer of neurons. '''
        layer = Layer(len(self.layers), torch.zeros(n))
        self.layers[layer.id] = layer
        return layer

    def connect(self, source: Layer, target: Layer, update) -> Connection:
        ''' Adds a new connection between layers. '''
        rows        = source.neurons.size(0)
        cols        = target.neurons.size(0)
        connection  = Connection(len(self.connections), source, target, torch.zeros((rows, cols)), update)
        self.connections[connection.id] = connection
        return connection

    def show(self, cxn: Connection, title: str = None, min_size:int = 256, scale: int = 20, duration: int = -1):
        '''
        Renders an cv2 image of the connection between 2 layers as a connectivity matrix.
        In this matrix, the rows are the source neurons and the columns are the targets.
        '''
        title = title or cxn.label
        img = scale_aspect(cxn.weight.numpy(), min_size, scale)
        cv2.imshow(title, img)
        cv2.waitKey(duration)

    def update(self):
        pass



def run_gym(env_name: str, nb_eps: int, nb_timesteps: int, net: Net):
    import gym
    env = gym.make(env_name)
    best_reward = None
    # Start training regime
    for i_ep in range(nb_eps):
        observation = env.reset()
        episode_reward = 0
        best_reward = max(episode_reward, best_reward or episode_reward)
        print(best_reward)
        # Start training episode
        for i_ts in range(nb_timesteps):
            env.render()
            net.input(observation)
            action = net.output()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(i_ts + 1))
                break
    print("Session complete, best reward {}".format(best_reward))
    env.close()


net = Net()
a   = net.add(20)
b   = net.add(30)
c   = net.add(5)
a_b = net.connect(c,a)

run_gym('CartPole-v0', 1, 1, net)
