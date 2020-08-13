import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

# simple neuron model
# position  : (2,) [0,j],[0,k]      # position in a numpy array
# baseline  : ()   [0,t]            # value of this neuron at rest                         
# potential : ()   [0,t]            # current value (state)                                
# threshold : ()   [n,1]            # value at which the neuron fires                      
# input_weights: (j,) [0,1]            # ease at which this neuron responds to input signals       
# amplitude : ()   [0,1]            # how much this neuron affects downstream neurons
# elasticity: ()   [0,1]            # rate at which plastified afinities decay without stimulation
# plasticity: ()   [0,1]            # rate at which input_weights change when stimulated (akin to learning rate)         
# stability : ()   [0,1]            # resistance to potential decay i.e. affects potential (expected to be a function of plasticity)
#
# class Defaults:
#     baseline=0
#     potential=0
#     threshold=0.5
#     input_weights=1.0
#     # meta parameters, not strictly required
#     amplitude=1.0
#     elasticity=0.5
#     plasticity=0.5
#     stability=0.5
#
# neuron update rule sketch
# note: all of this should be performed as immutable update
# 
# for neurons `n` with inputs `i` and outputs `o`
# 
# if potential[n] > threshold[n]:
#   potential[n] = baseline[n]                                                      # normally neurons have a refractory period, perhaps a refractory period can regulate repeated stimulation
#   potential[o] += amplitude[n] * input_weights[o][n]                                 # this should scale with number of input/output neurons
#   input_weights[n][i] *= (1 - plasticity[i]) * (1 + potential[o] - threshold[o])     # careful, will have to clamp to minimum or dropout and clamp max as well
# 
# input_weights[n][i] *= (1 - elasticity[n])                                           # this one will require a lot of tuning
# potential[n] *= (1 - stability[n])                                                # this one will require some tuning

rng = np.random.default_rng()
epsilon = 0.005

def reprnd(arr, name="arr"):
    return "-------------------------------\n{}: {}\n{}\n".format(name, arr.shape, arr.__repr__())

def repr3d(arr, name="arr"):
    arrfmt = "\n\n".join(["z=" + str(z) + "\n" + arr[...,z].__repr__() for z in range(arr.shape[2])])
    return "-------------------------------\n{}: {}\n{}\n".format(name, arr.shape, arrfmt)

def ndbg(ndarray, label=""):
    print(reprnd(ndarray, label))
        
def banner(text):
    print("{}\n{:^64}\n{}\n".format("=" * 64, text,"=" * 64))

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Model:
    ''' A spiking neural net model. 
    '''
    def __init__(self, shape, metric='chebyshev'):
        self.shape          = shape
        self.size           = np.zeros(shape).size
        self.refractory     = epsilon
        self.baseline       = 0.10
        self.threshold      = 0.50
        self.amplitude      = 0.10
        self.stability      = 0.99
        self.reuptake       = 0.10
        self.plasticity     = 0.005
        self.potential      = self.baseline * np.ones(self.shape).flatten()
        self.position       = np.array([i for i in np.ndindex(self.shape)])
        self.neuron         = np.arange(self.position.shape[0])
        # this guy is a problem since it scales with the square of the input size
        self.distance       = squareform(pdist(self.position, metric))
        # A list of adjacent neurons for each neuron.
        # e.g. adjacent[0] is a list of neuron indexes that are exactly 1 unit away from position[0]
        adjacents           = np.array([self.neuron[self.distance[n] <= 1] for n in self.neuron
                                    # This line ensures that positions on the edges of the model 
                                    # are not output positions (although they may still receive outputs).
                                    # This is required so that the output arrays are all the same size for numpy.
                                        if np.mod(self.position[n], np.array(self.shape) - 1).all()])
        # TODO: replace=False?, there is some repeated units here as edges are not incuded in adjacents
        self.outputs        = rng.choice(adjacents, self.neuron.size)
        # input_weights fill a similar role to Glutamate/GABA
        self.input_weights  = np.ones(self.outputs.shape)
        self.dopamine       = np.zeros(self.outputs.shape)


    def __repr__(self):
        return "{}\n{}".format(
            reprnd(self.potential, "potential"), 
            reprnd(self.input_weights, "input_weights")
        )

    def update(self):
        ''' Spiking neuron update algorithm:
            ### Psuedo code:
                for each activated neuron
                    # set activated neuron potential back to the baseline
                    new.activated.potential = baseline
                    # increase output neuron potential
                    new.output.potential += activated.amplitude * output.affinity
                    # update activated neuron input_weights
                    # (why do this only for activated neurons? good question)
                    new.activated.input_weights += (1 + new.output.potential - output.threshold)
                # decrease all neuron potentials
                new.potential *= stability
        '''
        new_input_weights   = self.input_weights.copy()
        new_potential       = self.potential.copy()
        activated           = self.potential > self.threshold
        # Set activated potentials back to refactory value
        show(new_potential, new_input_weights)
        new_potential[activated] = self.refractory
        # Increase output neuron potentials
        show(new_potential, new_input_weights)
        new_potential[self.outputs[activated]] += self.amplitude * self.input_weights[activated]
        show(new_potential, new_input_weights)
        # ndbg(self.outputs[activated,:],"out ac")
        # ndbg(new_input_weights[self.outputs[activated,:]],"aff out")
        # ndbg((1 - self.plasticity) * (1 + new_potential[self.outputs[activated,:]] - self.threshold),"out pot upd")
        for o in range(self.outputs[activated].shape[1]):
            # ndbg(new_input_weights[self.outputs[activated, o]] ,"aff out ac")
            # Without this loop the broadcast wouldn't work, this is what we are trying to do...
            # [act-idx, out-idx, aff] += [act-idx, out-pot]
            # The size of output potentials is the same as the input_weights since they are directly correlated.
            # Neuron n has:
            # [o0, o1, o2, ..., oN] output neurons,
            # [a0, a1, a2, ..., aN] associated input_weights
            # So each output neuron also has:
            # [a0, a1, a2, ..., aN] associated input_weights
            # since they are neurons, duh.
            # The size of N depends on the nb of outputs for each neuron which currently is based on the metric used to select outputs.
            new_input_weights[self.outputs[activated, o]] += self.plasticity * (self.dopamine[self.outputs[activated, o]] + self.potential[self.outputs[activated,:]] - self.threshold)
        show(new_potential, new_input_weights)
        # Apply homeostatic adjustments
        new_potential *= self.stability
        self.dopamine *= (1 - self.reuptake)
        show(new_potential, new_input_weights)
        # Push the update
        self.potential = new_potential.clip(epsilon, 1 - epsilon)
        self.input_weights = new_input_weights.clip(epsilon, 1 - epsilon)
        return activated

    def train(self, neurons, value):
        # TODO: This is a naieve approach...
        # We can instead increase dopamine for only the inputs that cause
        # a spike in the given neuron, we will address this later
        self.dopamine[neurons] += value

    def sense(self, neurons, observation):
        self.potential[neurons] += observation

    def probe(self, coords):
        return self.potential.reshape(self.shape)[coords]


def show(potential, input_weights, duration=1, debug=False):
    cv2.imshow("potenital",cv2.resize(np.hstack((potential.reshape(shape), input_weights[:,4].reshape(shape))), (1024, 512), interpolation=cv2.INTER_NEAREST))
    cv2.moveWindow("potenital", 50, 700)
    if debug:
        ndbg(potential.reshape(shape), "potentials")
        ndbg(input_weights, "input_weights")
    cv2.waitKey(duration)


def run(model, iterations):
    np.set_printoptions(linewidth=200)
    banner("RUN BEGINS")
    for _ in range(iterations):
        if _ == 1:
            model.potential[rng.integers(0, model.size,10)] = 0.8
            cv2.waitKey(0)
        model.update()
    cv2.waitKey(0)


def cart_pole_sensor(observation):
    return sigmoid(observation)

def cart_pole_actuator(active):
    # While normally actions would be coded to more than 1 neuron
    # It doesn't matter in this case as the actions are opposite
    # It might matter in a more complex example
    return 1 if np.any(active) > 0.5 else 0

# Run the model
shape = (6, 6)
iterations = 10000
model = Model(shape)
inputs = np.array([[0,63],[63,63]])

# run(model, iterations)

def run_gym(env):
    neurons                         = model.neuron.reshape(model.shape)
    sensor_neurons                  = neurons[[1,1,1,1],[1,2,3,4]]
    actuator_neurons                = neurons[[4,4,4,4],[1,2,3,4]]
    activations                     = model.update()

    previous_episode_reward         = 0
    episode_reward                  = 0
    best_reward                     = 0
    for i_episode in range(20):
        observation                 = env.reset()
        best_reward                 = max(episode_reward, best_reward)
        previous_episode_reward     = episode_reward
        episode_reward              = 0
        for t in range(100):
            env.render()
            # Sense
            model.sense(sensor_neurons, cart_pole_sensor(observation))
            # Think
            activations = model.update()
            # Act
            action = cart_pole_actuator(activations[actuator_neurons])
            print(np.any(activations[actuator_neurons]))
            # Observe
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        # Skip the first episode for training
        if i_episode > 0:
            delta_reward = sigmoid(episode_reward - previous_episode_reward)
            model.train(actuator_neurons, delta_reward)
    print("Session complete, best reward {}".format(best_reward))
    cv2.waitKey(0)
    env.close()

import gym
env = gym.make('CartPole-v0')
run_gym(env)
