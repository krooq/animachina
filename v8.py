###############################################################################
# v8
# This version builds on v7 and is actually a regressed simplified version.
# The goal is to beat cartpole with a scratch built SNN but really this version
# will focus on developing the reward and update functions.
# By using cartpole we can avoid the large tensors required for atari games.
###############################################################################
from snn import Connection, Layer, Network
from rl import Agent, run_gym
import cv2 
from util import *
from torch.types import Number
import torch as torch
import torch.nn.functional as f

###############################################################################
# Globals
###############################################################################
threshold       = 0.80
baseline        = 0.10
recovery_rate   = 0.005
learning_rate   = 1.0

###############################################################################
# Types
###############################################################################
class Net(Network):
    def add(self, n: int) -> Layer:
        return super().add(torch.rand(n))

    def connect(self, source: Layer, target: Layer) -> Connection:
        rows = target.neurons.numel()
        cols = source.neurons.numel()
        weight = torch.ones((rows, cols))/cols
        return super().connect(source, target, weight)

###############################################################################
# Functions
###############################################################################
def lif(cxn: Connection):
    s = cxn.source.neurons
    s_a = s > threshold
    t0 = cxn.target.neurons
    t1 = t0 + torch.mv(cxn.weight, s_a.float())
    t1 = f.normalize(t1, p=1, dim=0)
    w0 = cxn.weight
    w1 = w0 + torch.ger(t1 - t0, s) * learning_rate
    w1 = f.normalize(w1, p=1, dim=1)

    # if net.t > 0 and net.t % 100 == 0:
    #     exit(0)
    #     print("weights:\n {}".format(w1.numpy()))
    # if net.t % 1000 == 0 and cxn.id==0:
    #     cv2.imshow('weights', (cxn.weight/torch.mean(cxn.weight)/2).reshape((210*4,160)).numpy())

    cxn.source.neurons[s_a] = baseline
    cxn.target.neurons = t1
    cxn.weight = w1
    cxn.source.neurons += recovery_rate * (baseline - s)
    cxn.target.neurons += recovery_rate * (baseline - t1)

def select_softmax(tensor: torch.Tensor) -> Number:
    # take the softmax of each neuron in the layer to form a probability distribution
    distribution = torch.softmax(tensor, dim=0)
    # select a single sample
    return torch.multinomial(distribution, num_samples=1).item()

def greyscale(image) -> torch.Tensor:
    # normalize the image data wrt. the image space
    return torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten(), dtype=torch.float32) / 256

###############################################################################
# Run
###############################################################################
if __name__ == '__main__':
    # debug()
    net = Net()
    a   = net.add(4)
    b   = net.add(3)
    c   = net.add(2)
    a_b = net.connect(a, b)
    b_c = net.connect(b, c)

    def observe(observation):
        obs = f.normalize(torch.tensor(observation), p=1, dim=0)
        a.neurons += obs
        update()

    def update():
        lif(a_b)
        lif(b_c)
        net.t += 1

    def act() -> object:
        net.last_action = select_softmax(c.neurons)
        c.neurons[c.neurons > threshold] = baseline
        return net.last_action

    def reward(rewards):
        b_c.weight[net.last_action,:] *= (1 + rewards)
        b_c.source.neurons = b_c.source.neurons + torch.matmul(b_c.target.neurons, b_c.weight)
        # if rewards:
        #     print("reward: {}".format(rewards))
        #     print("action: {}".format(net.last_action))


    agent = Agent(observe, act, reward)
    # average episode reward for random carpole agent is ~21
    run_gym('CartPole-v1', agent, nb_eps=1000, nb_timesteps=200)

###############################################################################
# Debug
###############################################################################
def debug():
    pass
    # debugging pytorch stuff
    # source = torch.tensor([1, 0, 1, 0])
    # target = torch.tensor([0, 0, 0])
    # weight = torch.ones((3, 4))
    # print(weight)
    # print(f.normalize(weight, p=1, dim=1))
    # exit()