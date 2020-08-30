# BindsNET is great in that it utilizes Pytorch and has a nice API
# but the clean API means it isn't very transparent
# the goal of this version is to improve v4 (DIY) with ideas from v5 (BindsNET)
# remember, we are not trying to create an API, just a powerful generic SNN
from abc import abstractclassmethod
import time
from typing import Callable, Dict
from bindsnet.learning.learning import LearningRule
import torch as torch
import cv2
import numpy as np
from torch import tensor
from util import *
import gym



threshold       = 0.80

dt              = 1.0
epsilon         = 0.05
baseline        = 0.10
amplitude       = 0.10
stability       = 0.99
reuptake        = 0.000
plasticity      = 0.005
refractory      = epsilon

Id = int

class Node:
    def __init__(self, id: Id, label: str = None):
        self.id = id
        self.label = label

class Layer(Node):
    def __init__(self, id: Id, neurons: torch.Tensor):
        super().__init__(id, "layer_{}".format(id))
        self.neurons = neurons

class Connection(Node):
    def __init__(self, id: Id, source: Layer, target: Layer, weight: torch.Tensor, update):
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
        for cxn in self.connections.values():
            cxn.update(cxn)

class Agent:
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractclassmethod
    def observe(self, observation):
        raise NotImplementedError

    @abstractclassmethod
    def act(self) -> torch.Tensor:
        raise NotImplementedError

class Sensor:
    @abstractclassmethod
    def observe(self, observation):
        raise NotImplementedError
    
class Actuator:
    @abstractclassmethod
    def act(self) -> object:
        raise NotImplementedError

class BreakoutAgent(Agent):    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, sensor: Sensor, actuator: Actuator):
        self.observation_space = observation_space
        self.action_space = action_space
        self.sensor = sensor
        self.acuator = actuator

    def observe(self, observation):
        self.sensor.observe(observation)

    def act(self) -> object:
        return self.acuator.act()

def run_gym(env: gym.Env, nb_eps: int, nb_timesteps: int, agent: Agent):
    best_reward = None
    # Start training regime
    for i_ep in range(nb_eps):
        observation = env.reset()
        episode_reward = 0
        best_reward = max(episode_reward, best_reward or episode_reward)
        # Start training episode
        for i_ts in range(nb_timesteps):
            env.render()
            agent.observe(observation)
            action = agent.act()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(i_ts + 1))
                break
    print("Session complete, best reward {}".format(best_reward))
    env.close()

class AtariVision(Sensor):
    def __init__(self, net: Net, layer: Layer):
        self.net = net
        self.layer = layer
        self.previous_img = 0
        self.sensitivity = threshold / 255
        self.frequency = 10
        self.leakiness = 0.2
        self.size = (210,160)
        
    def observe(self, observation):
        img = torch.tensor(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY).flatten(), dtype=torch.float32)
        for _ in range(self.frequency):
            # self.input_layer.neurons += img * self.sensitivity - self.input_layer.neurons.numpy() * self.leakiness
            # add the new normalized image data 
            self.layer.neurons += img / 255
            # cv2.imshow('', scale_aspect(self.input_layer.neurons.numpy().reshape((210,160)), 256, 1))
            net.update()

class BreakoutActuator(Actuator):
    def __init__(self, net: Net, layer: Layer):
        self.net = net
        self.layer = layer

    def act(self) -> object:
        # select softmax function
        return torch.multinomial(torch.softmax(self.layer.neurons, dim=0), num_samples=1).item()

def integrate_and_fire(cxn: Connection):
    activated = cxn.source.neurons > threshold
    nb_activated = activated.nonzero().size(0)
    cxn.target.neurons += nb_activated / cxn.source.neurons.size(0)
    cxn.source.neurons[activated] = 0
    if cxn.target.neurons.size(0) == 9:
        cv2.imshow('', scale_aspect(cxn.target.neurons.numpy().reshape((3,3)), 210, 50))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

net = Net()
a   = net.add(210*160)
b   = net.add(9)
c   = net.add(4)
a_b = net.connect(a, b, integrate_and_fire)
b_c = net.connect(b, c, integrate_and_fire)

env = gym.make('BreakoutDeterministic-v4')
agent = BreakoutAgent(env.observation_space, env.action_space, AtariVision(net, a), BreakoutActuator(net, c))

run_gym(env, nb_eps=10, nb_timesteps=100, agent=agent)
