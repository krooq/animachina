import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, Nodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from collections import namedtuple
from typing import Dict, Optional, Type, Iterable, Union, Sequence

# Model
layer_0 = Input(n=100)
layer_1 = LIFNodes(n=1000)
cxn_0_1 = Connection(layer_0, layer_1, w=0.05 + 0.1 * torch.randn(layer_0.n, layer_1.n))
cxn_1_1 = Connection(layer_1, layer_1, w=0.025 * (torch.eye(layer_1.n) - 1))

# Monitoring
time = 500
mon_0 = Monitor(layer_0, ("s",), time)
mon_1 = Monitor(layer_1, ("s","v"), time)

net = Network()
net.add_layer(layer_0, 'layer_0')
net.add_layer(layer_1, 'layer_1')
net.add_connection(cxn_0_1, 'layer_0', 'layer_1')
net.add_connection(cxn_1_1, 'layer_1', 'layer_1')
net.add_monitor(mon_0, 'mon_0')
net.add_monitor(mon_1, 'mon_1')

# Create and assign input signals for each timestep
input_data = torch.bernoulli(0.1 * torch.ones(time, layer_0.n)).byte()
inputs = {'layer_0': input_data}

# Simulate network on input data.
net.run(inputs, 100)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {'layer_0': mon_0.get("s"), 'layer_1': mon_1.get("s")}
voltages = {'layer_1': mon_1.get("v")}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()