###############################################################################
# v7
# This version builds on v6.
# Still trying to make our own simple spiking neural net model like BindsNET
# The goal of v7 is to adopt a temporal coding mechanism over the rate coding
###############################################################################
from snn import Connection, Layer, Network
from rl import Agent, run_gym
import cv2 
from util import *
import gym
from torch.types import Number
import torch as torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###############################################################################
# Globals
###############################################################################
threshold       = 0.80
baseline        = 0.10
recovery_rate   = 0.005

###############################################################################
# Types
###############################################################################
class Net(Network):
    def add(self, n: int) -> Layer:
        return super().add(torch.zeros(n))

    def connect(self, source: Layer, target: Layer) -> Connection:
        rows = target.neurons.numel()
        cols = source.neurons.numel()
        weight = torch.rand((rows, cols))
        return super().connect(source, target, weight)

###############################################################################
# Functions
###############################################################################
def integrate_and_fire(cxn: Connection):

    # find the spiking neurons, then update target neurons with weighted spikes
    source_active = cxn.source.neurons > threshold
    cxn.target.neurons += torch.matmul(cxn.weight, source_active.float())
    # increase the weights of the activations and renormalize
    # TODO: fix this thats breaking everything
    cxn.weight = torch.einsum('i,j->ij', cxn.target.neurons, cxn.source.neurons + 1)
    if net.t % 1000 == 0 and cxn.id == 1:
        print(cxn.weight)
    cxn.weight = f.normalize(cxn.weight, p=1, dim=1)
    # reset the source neurons to baseline value
    cxn.source.neurons[source_active] = baseline
    # apply homeostatis, returning neuron potentials back to baseline
    cxn.source.neurons += recovery_rate * (baseline - cxn.source.neurons)
    cxn.target.neurons += recovery_rate * (baseline - cxn.target.neurons)

def select_softmax(tensor: torch.Tensor) -> Number:
    # take the softmax of each neuron in the layer to form a probability distribution
    distribution = torch.softmax(tensor, dim=0)
    # select a single sample
    return torch.multinomial(distribution, num_samples=1).item()

def greyscale(image) -> torch.Tensor:
    # normalize the image data wrt. the image space
    return torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten(), dtype=torch.float32) / 256

def breakout_reward(net:Net, reward):
    # WIP
    for cxn in net.connections.values():
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
        
def tts(cxn: Connection):
    ''' Time-to-spike encoding, whatever that means. '''
    # TODO: time to spike
    pass

def isi(cxn: Connection):
    ''' Inter-spike-interval encoding, whatever that means. '''
    # TODO: inter-spike interval
    pass

###############################################################################
# Run
###############################################################################
if __name__ == '__main__':
    net = Net()
    a   = net.add(210*160)
    b   = net.add(6)
    c   = net.add(4)
    a_b = net.connect(a, b)
    b_c = net.connect(b, c)

    def observe(observation):
        obs = greyscale(observation)
        # TODO: Temporal coding
        period = 10
        # Scale the image by the thresold so we capture all the values (all values must be less than the threshold)
        # Also scale by period value, this well depend on the update rate
        a.neurons +=  obs / threshold / period
        show(a.neurons, (210,160), label=a.label)
        for _ in range(period):
            update()

    def update():
        integrate_and_fire(a_b)
        cv2.imshow('weights', (a_b.weight/torch.mean(a_b.weight)/2).reshape((210*6,160)).numpy())
        # show(a_b.weight, (210*4, 160), label=a_b.label)
        # if net.t % 1000 == 0:
        #     print(a_b.target.neurons.numpy())
        integrate_and_fire(b_c)
        net.t += 1

    def act() -> object:
        net.last_action = select_softmax(c.neurons)
        return net.last_action

    def reward(rewards):
        b_c.weight[net.last_action,:] *= (1 + rewards)
        if rewards:
            print("reward: {}".format(rewards))
            print("action: {}".format(net.last_action))
            print(b_c.weight)
        # b_c.weight = f.normalize(b_c.weight, p=1, dim=0)
        # a_b.weight[:, None] *= (1 + rewards)
        # a_b.weight = f.normalize(a_b.weight, p=1, dim=1)
        
    # debugging pytorch stuff
    # source = torch.tensor([1, 0, 1, 0])
    # target = torch.tensor([0, 0, 0])
    # weight = torch.ones((3, 4))
    # print(weight)
    # print(f.normalize(weight, p=1, dim=1))

    agent = Agent(observe, act, reward)
    run_gym('BreakoutDeterministic-v4', agent, nb_eps=1000, nb_timesteps=1000)
