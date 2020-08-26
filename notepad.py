
def demo():
    time = 500
    # Create input signals for each timestep and assign to input layers
    inputs = {'layer_0': torch.bernoulli(0.1 * torch.ones(time, layer_0.n)).byte()}
    # Simulate network on input data
    net.run(inputs, 500)
    mon_0   = net.add_monitor(Monitor(layer_0, ("s",), time), 'mon_0')
    mon_1   = net.add_monitor(Monitor(layer_1, ("s","v"), time), 'mon_1')
    # Retrieve and plot simulation spike, voltage data from monitors.
    spikes = {'layer_0': mon_0.get("s"), 'layer_1': mon_1.get("s")}
    voltages = {'layer_1': mon_1.get("v")}
    plt.ioff()
    plot_spikes(spikes)
    plot_voltages(voltages, plot_type="line")
    plt.show()
