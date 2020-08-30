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
baseline        = 0.10

dt              = 1.0
epsilon         = 0.05
amplitude       = 0.10
stability       = 0.99
reuptake        = 0.000
plasticity      = 0.005
refractory      = epsilon

Id = int

class Node:
    ''' A component of a neural network. '''
    def __init__(self, id: Id, label: str = None):
        self.id = id
        self.label = label

class Layer(Node):
    ''' A group of neurons. '''
    def __init__(self, id: Id, neurons: torch.Tensor):
        super().__init__(id, "layer_{}".format(id))
        self.neurons = neurons

class Connection(Node):
    ''' A connection between groups of neurons. '''
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
        id = len(self.layers)
        neurons = torch.zeros(n)
        layer = Layer(id, neurons)
        self.layers[layer.id] = layer
        return layer

    def connect(self, source: Layer, target: Layer, update) -> Connection:
        ''' Adds a new connection between layers. '''
        id = len(self.connections)
        rows = target.neurons.numel()
        cols = source.neurons.numel()
        weight = torch.ones((rows, cols))
        connection = Connection(id, source, target, weight, update)
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
    def reward(self, reward):
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
    def __init__(self, sensor: Sensor, actuator: Actuator, reward_sensor: Sensor):
        self.sensor = sensor
        self.acuator = actuator
        self.reward_sensor = reward_sensor

    def observe(self, observation):
        self.sensor.observe(observation)

    def reward(self, reward) -> object:
        return self.reward_sensor.observe(reward)

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
            agent.reward(reward)
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

class BreakoutReward(Sensor):
    def __init__(self, net: Net, layer: Layer):
        self.net = net
        self.layer = layer
        
    def observe(self, reward):
        # for cxn in self.net.connections.values():
        #     cxn.weight = torch.softmax((cxn.weight*cxn.weight).flatten(), dim=0).reshape(cxn.weight.shape)
        pass


# TODO: encoders instead of sensors
# def atari_vision(layer: Layer, img: np.ndarray):
#     img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten(), dtype=torch.float32)
#     layer.neurons += img / 255

def integrate_and_fire(cxn: Connection):
    source_active = cxn.source.neurons > threshold
    spikes = torch.where(source_active, torch.tensor(threshold)/source_active.numel(), torch.tensor(0.0))
    cxn.target.neurons += torch.matmul(cxn.weight, spikes)
    target_active = cxn.target.neurons > threshold
    weight_relevant = torch.einsum('i,j->ij', target_active.float(), source_active.float()).bool()
    # NOTE: rewards should be based on reward prediction error, i.e. RPE
    # simply put, this means weights should change relative to current weights
    # weights supporting the result should increase
    # weights against the result should decrease
    positive_weight_factor = torch.tensor(torch.sum(weight_relevant).float() / weight_relevant.numel())
    print(positive_weight_factor)
    negative_weight_factor = positive_weight_factor - 1
    cxn.weight += torch.where(weight_relevant, positive_weight_factor, negative_weight_factor)
    # cxn.weight = torch.softmax(cxn.weight.flatten(), dim=0).reshape(cxn.weight.shape)
    cxn.source.neurons[source_active] = baseline
    cv2.imshow(cxn.label, cv2.resize(cxn.weight.numpy(), (512, 128), interpolation=cv2.INTER_NEAREST))

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
agent = BreakoutAgent(AtariVision(net, a), BreakoutActuator(net, c), BreakoutReward(net, a))

run_gym(env, nb_eps=1, nb_timesteps=10, agent=agent)
