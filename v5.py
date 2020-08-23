import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, Nodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

# Very high level API for bindsnet
class Net(Network):
    def add_layer(self, nodes: Nodes, name: str) -> str:
        super().add_layer(nodes, name)
        setattr(nodes, 'name', name)
        return nodes

    def add_connection(self, connection: Connection) -> Connection:
        super().add_connection(connection, source=connection.source.name, target=connection.target.name)
        return connection

    def add_monitor(self, monitor: Monitor, name: str) -> Monitor:
        super().add_monitor(monitor, name)
        return monitor

# Model
time = 500
net = Net()
layer_0 = net.add_layer(Input(n=100), 'layer_0')
layer_1 = net.add_layer(LIFNodes(n=1000), 'layer_1')
cxn_0_1 = net.add_connection(Connection(layer_0, layer_1, w=0.05 + 0.1 * torch.randn(layer_0.n, layer_1.n)))
cxn_1_1 = net.add_connection(Connection(layer_1, layer_1, w=0.025 * (torch.eye(layer_1.n) - 1)))
mon_0   = net.add_monitor(Monitor(layer_0, ("s",), time), 'mon_0')
mon_1   = net.add_monitor(Monitor(layer_1, ("s","v"), time), 'mon_1')

# Create input signals for each timestep and assign to input layers
inputs = {'layer_0': torch.bernoulli(0.1 * torch.ones(time, layer_0.n)).byte()}
# Simulate network on input data
net.run(inputs, 100)

# Retrieve and plot simulation spike, voltage data from monitors.
spikes = {'layer_0': mon_0.get("s"), 'layer_1': mon_1.get("s")}
voltages = {'layer_1': mon_1.get("v")}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()