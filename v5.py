import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, Nodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import MSTDP
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.pipeline.action import select_softmax

# Very high level API for bindsnet
class Net(Network):
    def add_layer(self, nodes: Nodes, name: str) -> str:
        super().add_layer(nodes, name)
        nodes.name = name
        return nodes

    def add_connection(self, connection: Connection) -> Connection:
        super().add_connection(connection, source=connection.source.name, target=connection.target.name)
        return connection

    def add_monitor(self, monitor: Monitor, name: str) -> Monitor:
        super().add_monitor(monitor, name)
        return monitor

# Model
net = Net()
layer_0 = net.add_layer(Input(n=80 * 80, shape=[1,1,1,80, 80]), 'layer_0')
layer_1 = net.add_layer(LIFNodes(n=10, traces=True), 'layer_1')
layer_2 = net.add_layer(LIFNodes(n=4, traces=True, refrac=0,), 'layer_2')
cxn_0_1 = net.add_connection(Connection(layer_0, layer_1, wmin=0, wmax=1))
cxn_1_2 = net.add_connection(Connection(layer_1, layer_2, wmin=0, wmax=1, update_rule=MSTDP, nu=1e-1, norm=0.5 * layer_1.n))
mon_0_1 = net.add_monitor(Monitor(layer_1, ("v",)), 'mon_0_1')

# Load the gym environment.
env = GymEnvironment('BreakoutDeterministic-v4')
env.reset()

# Build pipeline from specified components.
pipeline = EnvironmentPipeline(
    net,
    env,
    action_function=select_softmax,
    encoding=bernoulli,
    # num_episodes=100,
    output=layer_2.name,
    render_interval=1,
    reward_delay=None,
    time=100,
)
import cv2

for i in range(100):
    total_reward = 0
    pipeline.reset_state_variables()
    is_done = False
    while not is_done:
        result = pipeline.env_step()
        pipeline.step(result)
        print(mon_0_1.get("v").numpy())
        # cv2.imshow("input_weights", cv2.resize(mon_0_1.get("v").numpy().reshape(10,100), (256, 256), interpolation=cv2.INTER_NEAREST))
        mon_0_1.reset_state_variables()
        reward = result[1]
        total_reward += reward

        is_done = result[2]
    print(f"Episode {i} total reward:{total_reward}")

env.close()