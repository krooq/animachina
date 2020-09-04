###############################################################################
# v7
# This version builds on v6.
# Still trying to make our own simple spiking neural net model like BindsNET
# The goal of v7 is to adopt a temporal coding mechanism over the rate coding
###############################################################################
from snn import Connection, Encoder, Layer, Network
from rl import Agent, run_gym
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
class Net(Network):
    def add(self, n: int = 1) -> Layer:
        return super().add(torch.zeros(n))

    def connect(self, source: Layer, target: Layer) -> Connection:
        rows = target.neurons.numel()
        cols = source.neurons.numel()
        weight = torch.ones((rows, cols)) / cols
        return super().connect(source, target, weight)

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

def select_softmax(tensor: torch.Tensor) -> torch.Number:
    # take the softmax of each neuron in the layer to form a probability distribution
    distribution = torch.softmax(tensor, dim=0)
    # select a single sample
    return torch.multinomial(distribution, num_samples=1).item()

def greyscale(image) -> torch.Tensor:
    # normalize the image data wrt. the image space
    img = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten(), dtype=torch.float32) / 256
    # TODO: Temporal coding
    # Scale the image by the thresold so we capture all the values (all values must be less than the threshold)
    return img / threshold
    # show(self.layer.neurons, (210,160), label=self.layer.label)
    # net.update()

def breakout_reward(self, reward):
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
    b   = net.add(4)
    c   = net.add(4)
    a_b = net.connect(a, b)
    b_c = net.connect(b, c)

    def observe(observation):
        a.neurons += greyscale(observation)
        show(a.neurons, (210,160), label=a.label)
        net.update()

    def update():
        integrate_and_fire(a_b)
        integrate_and_fire(b_c)

    def act() -> object:
        return select_softmax(c)

    def reward(rewards):
        breakout_reward(rewards)

    # debugging pytorch stuff
    # source = torch.tensor([1, 0, 1, 0])
    # target = torch.tensor([0, 0, 0])
    # weight = torch.ones((3, 4))
    # print(weight)
    # print(f.normalize(weight, p=1, dim=1))

    env = gym.make('BreakoutDeterministic-v4')
    agent = Agent(observe, act, reward)

    run_gym(env, agent, nb_eps=1000, nb_timesteps=1000)
