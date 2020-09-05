
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


def show_layers(self, title: str = "net", px: int = 50, duration: int = -1):
    ''' Shows the net as a stack of layers. '''
    width = max(t.numpy().size for t in self.layers)
    img = [np.zeros(width) for _ in self.layers]
    img = np.vstack(img)
    for i,t in enumerate(self.layers):
        img[i,:t.numpy().size] = t.numpy()
    img_size = (px * width, px * len(self.layers))
    cv2.imshow(title, cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(duration)



def show(self, cxn: Connection, title: str = None, min_size:int = 256, scale: int = 20, duration: int = -1):
    '''
    Renders an cv2 image of the connection between 2 layers as a connectivity matrix.
    In this matrix, the rows are the source neurons and the columns are the targets.
    '''
    title = title or cxn.label
    img = scale_aspect(cxn.weight.numpy(), min_size, scale)
    cv2.imshow(title, img)
    cv2.waitKey(duration)

    # update weights with the reward prediction error i.e. signed difference between actual and predicted
    # target_active = cxn.target.neurons > threshold
    # weight_mask = torch.einsum('i,j->ij', target_active.float(), source_active.float()).bool()
    # reward_prediction_error = torch.max(cxn.weight)
    # cxn.weight += weight_mask * reward_prediction_error

# create plot
plt.ion()
t = []
v = []
plt.show(block=False)
# add data
t.append(net.t)
v.append(a.neurons.numpy()[210*70 + 80])
plt.gca().cla()
plt.plot(t, v)
plt.draw()
plt.pause(0.01)



class Monitor(Node):
    def __init__(self, id: Id, duration: int):
        super().__init__(id, "monitor_{}".format(id))
        self.v = np.zeros(duration)
        self.t = np.arange(duration)
    
    def record(self, t: int, v: np.float):
        self.v[t] = v

    
    def monitor(self, duration) -> Monitor:
        id = len(self.monitors)
        self.monitors[id] = Monitor(id, duration)
        return self.monitors[id]

        self.monitors: List[Monitor] = {}