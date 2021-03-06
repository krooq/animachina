# BindsNET is great in that it utilizes Pytorch and has a nice API
# but the clean API means it isn't very transparent
# the goal of this version is to improve v4 (DIY) with ideas from v5 (BindsNET)
# remember, we are not trying to create an API, just a powerful generic SNN
from abc import abstractclassmethod
import time
from typing import Callable, Dict
from bindsnet.learning.learning import LearningRule
import cv2
import numpy as np
from torch.types import Number
from util import *
import gym
import torch as torch
import torch.nn.functional as f



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
        self.reward = 0
        self.activations = torch.zeros(weight.shape)

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
        # intialize weights so the sum over all cols is 1
        weight = torch.ones((rows, cols))/cols
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

class Sensor:
    @abstractclassmethod
    def obs(self, env):
        raise NotImplementedError
    
class Actuator:
    @abstractclassmethod
    def act(self) -> object:
        raise NotImplementedError

class Agent:    
    def __init__(self, sensor: Sensor, actuator: Actuator, reward_sensor: Sensor):
        self.sensor = sensor
        self.acuator = actuator
        self.reward_sensor = reward_sensor

    def observe(self, observation):
        self.sensor.obs(observation)

    def reward(self, reward) -> object:
        return self.reward_sensor.obs(reward)

    def act(self) -> object:
        return self.acuator.act()
    
class AtariVision(Sensor):
    def __init__(self, net: Net, layer: Layer):
        self.net = net
        self.layer = layer
        self.frequency = 1
        
    def observe(self, observation):
        # normalize the image data
        img = torch.tensor(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY).flatten(), dtype=torch.float32) / 256
        for _ in range(self.frequency):
            # scale the image by the thresold so we capture all the values (all values must be less than the threshold)
            # I dont really know why we divide by self.frequency
            self.layer.neurons += img / (self.frequency * threshold)
            show(self.layer.neurons, (210,160), label=self.layer.label)
            net.update()

class BreakoutActuator(Actuator):
    def __init__(self, net: Net, layer: Layer):
        self.net = net
        self.layer = layer

    def act(self) -> object:
        # take the softmax of each neuron in the layer to form a probability distribution
        distribution = torch.softmax(self.layer.neurons, dim=0)
        # select a single sample
        return torch.multinomial(distribution, num_samples=1).item()

class BreakoutReward(Sensor):
    def __init__(self, net: Net, layer: Layer):
        self.net = net
        self.layer = layer
        
    def observe(self, reward):
        # WIP
        for cxn in self.net.connections.values():
            # in breakout, reward is the number of broken blocks
            simple_reward(cxn, reward)

# TODO: encoders instead of sensors
# def atari_vision(layer: Layer, img: np.ndarray):
#     img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten(), dtype=torch.float32)
#     layer.neurons += img / 255


def integrate_and_fire(cxn: Connection):
    # find the spiking neurons, then update target neurons with weighted spikes
    source_active = cxn.source.neurons > threshold
    cxn.target.neurons += torch.matmul(cxn.weight, source_active.float())

    # find the target neurons that spike as a result of the update and record the activations of source and target
    target_active = cxn.target.neurons > threshold
    cxn.activations = torch.einsum('i,j->ij', target_active.float(), source_active.float()).bool()

    # reset the source neurons to baseline value
    cxn.source.neurons[source_active] = baseline

    # FIXME: leaky?
    # cxn.source.neurons = torch.clamp(cxn.source.neurons - 1e-2, baseline, 1.0)

    # debug
    # if cxn.id == 0:
    #     # print(cxn.weight)
    #     # if torch.max(cxn.weight) == 0: 
    #     #     exit(1)
    #     show(cxn.weight/torch.max(cxn.weight), shape=(210*9,160), label=cxn.label)

def simple_reward(cxn: Connection, reward: Number):
    reward_prediction_error = reward - cxn.reward
    cxn.weight += 0.5 * reward_prediction_error * torch.mean(cxn.weight) * cxn.activations
    # renormalize the weights
    cxn.weight = f.normalize(cxn.weight, p=1, dim=1)
    if cxn.id == 0:
        if reward_prediction_error != 0:
            # print("max weight: {}".format(torch.max(cxn.weight)))
            # print("rpe: {}".format(reward_prediction_error))
            cv2.imshow('weights', (cxn.weight/torch.mean(cxn.weight)/10).reshape((210*4,160)).numpy())
    cxn.reward = reward


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

net = Net()
a   = net.add(210*160)
b   = net.add(4)
c   = net.add(4)
a_b = net.connect(a, b, integrate_and_fire)
b_c = net.connect(b, c, integrate_and_fire)


# source = torch.tensor([1, 0, 1, 0])
# target = torch.tensor([0, 0, 0])
# weight = torch.ones((3, 4))

# print(weight)
# print(f.normalize(weight, p=1, dim=1))

env = gym.make('BreakoutDeterministic-v4')
agent = Agent(AtariVision(net, a), BreakoutActuator(net, c), BreakoutReward(net, a))

run_gym(env, agent, nb_eps=1000, nb_timesteps=1000)
