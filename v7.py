# v7 builds on v6, still trying to make our own simple spiking neural net model like BindsNET
# the goal of v7 is to adopt a temporal coding mechanism over the rate coding in v6

from rl import Actuator, Agent, Id, Node, Sensor
from typing import Dict
import cv2
from util import *
import gym
import torch as torch
import torch.nn.functional as f

###############################################################################
# Globals
###############################################################################
threshold       = 0.80
baseline        = 0.10

###############################################################################
# Types
###############################################################################
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

    def update(self):
        for cxn in self.connections.values():
            cxn.update(cxn)

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
# Functions
###############################################################################
def integrate_and_fire(cxn: Connection):
    # find the spiking neurons, then update target neurons with weighted spikes
    source_active = cxn.source.neurons > threshold
    cxn.target.neurons += torch.matmul(cxn.weight, source_active.float())
    # find the target neurons that spike as a result of the update and record the activations of source and target
    target_active = cxn.target.neurons > threshold
    cxn.activations = torch.einsum('i,j->ij', target_active.float(), source_active.float()).bool()
    # reset the source neurons to baseline value
    cxn.source.neurons[source_active] = baseline


###############################################################################
# Run
###############################################################################
if __name__ == '__main__':
    net = Net()
    a   = net.add(210*160)
    b   = net.add(4)
    c   = net.add(4)
    a_b = net.connect(a, b, integrate_and_fire)
    b_c = net.connect(b, c, integrate_and_fire)

    # debugging pytorch stuff
    # source = torch.tensor([1, 0, 1, 0])
    # target = torch.tensor([0, 0, 0])
    # weight = torch.ones((3, 4))
    # print(weight)
    # print(f.normalize(weight, p=1, dim=1))

    env = gym.make('BreakoutDeterministic-v4')
    agent = Agent(AtariVision(net, a), BreakoutActuator(net, c), BreakoutReward(net, a))

    run_gym(env, agent, nb_eps=1000, nb_timesteps=1000)
