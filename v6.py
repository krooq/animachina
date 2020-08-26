# BindsNET is great in that it utilizes Pytorch and has a nice API
# but the clean API means it isn't very transparent
# the goal of this version is to improve v4 (DIY) with ideas from v5 (BindsNET)
# remember, we are not trying to create an API, just a powerful generic SNN
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple, Union
from collections import namedtuple

import torch as torch
import cv2
import numpy as np

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
        self.id     = id
        self.label  = label

class Layer(Node):
    def __init__(self, id: Id, neurons: torch.Tensor) -> None:
        super().__init__(id, "layer_{}".format(id))
        self.neurons    = neurons

class Connection(Node):
    def __init__(self, id: Id, source: Layer, target: Layer, weight: torch.Tensor) -> None:
        super().__init__(id, "connection_{} [{} to {}]".format(id, source.label, target.label))
        self.source    = source
        self.target    = target
        self.weight    = weight

class Net:
    def __init__(self):
        self.layers         : Dict[Id, Layer]       = {}
        self.connections    : Dict[Id, Connection]  = {}

    def add(self, n: int = 1) -> Layer:
        ''' Adds a new layer of neurons. '''
        layer = Layer(len(self.layers), torch.zeros(n))
        self.layers[layer.id] = layer
        return layer

    def connect(self, source: Layer, target: Layer) -> Connection:
        ''' Adds a new connection between layers. '''
        width   = source.neurons.size(0)
        height  = target.neurons.size(0)
        connection = Connection(len(self.connections), source, target, torch.rand((width, height)))
        self.connections[connection.id] = connection
        return connection

    def show(self, cxn: Connection, title: str = None, px: int = 20, duration: int = -1):
        ''' Shows the connection between 2 layers using a connectivity matrix. '''
        title       = title if title is not None else cxn.label
        img         = cxn.weight.numpy()
        img_size    = (px * img[0].size, px * img[1].size)
        img         = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(title, img)
        cv2.waitKey(duration)

    def show_layers(self, title: str = "net", px: int = 20, duration: int = -1):
        ''' Shows the net as a stack of layers. '''
        width = max(t.numpy().size for t in self.layers)
        img = [np.zeros(width) for _ in self.layers]
        img = np.vstack(img)
        for i,t in enumerate(self.layers):
            img[i,:t.numpy().size] = t.numpy()
        img_size = (px * width, px * len(self.layers))
        cv2.imshow(title, cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(duration)


net = Net()
a   = net.add(20)
b   = net.add(30)
c   = net.add(5)
a_b = net.connect(a,b)
net.show(a_b)