
# Id for a network component
from typing import List, TypeVar


Id = int
# The tensor type
T = TypeVar('T')

class Node:
    ''' A component of a network. '''
    def __init__(self, id: Id, label: str = None):
        self.id = id
        self.label = label

class Layer(Node):
    ''' A group of neurons. '''
    def __init__(self, id: Id, neurons: T):
        super().__init__(id, "layer_{}".format(id))
        self.neurons = neurons

class Connection(Node):
    ''' A connection between groups of neurons. '''
    def __init__(self, id: Id, source: Layer, target: Layer, weight: T):
        super().__init__(id, "connection_{} [{} to {}]".format(id, source.label, target.label))
        self.source = source
        self.target = target
        self.weight = weight

class Network:
    def __init__(self):
        self.t: int = 0
        self.layers: List[Layer] = {}
        self.connections: List[Connection] = {}

    def add(self, neurons: T) -> Layer:
        ''' Adds a new layer of neurons. '''
        id = len(self.layers)
        self.layers[id] = Layer(id, neurons)
        return self.layers[id]

    def connect(self, source: Layer, target: Layer, weight: T) -> Connection:
        ''' Adds a new connection between layers. '''
        id = len(self.connections)
        self.connections[id] = Connection(id, source, target, weight)
        return self.connections[id]