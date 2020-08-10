import cv2
import numpy as np
import pyautogui
from screeninfo import get_monitors
from scipy.spatial.distance import pdist, squareform
import time

# # display screen resolution, get it from your OS settings
# SCREEN_SIZE = (3840, 2160)
# # define the codec
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# # create the video write object
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, (SCREEN_SIZE))

# while True:
#     # make a screenshot
#     # img = pyautogui.screenshot()
#     img = pyautogui.screenshot(region=(0, 0, 300, 400))
#     # convert these pixels to a proper numpy array to work with OpenCV
#     frame = np.array(img)
#     # convert colors from BGR to RGB
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # write the frame
#     out.write(frame)
#     # show the frame
#     cv2.imshow("screenshot", frame)
#     # if the user clicks q, it exits
#     if cv2.waitKey(1) == ord("q"):
#         break

# # make sure everything is closed when exited
# cv2.destroyAllWindows()
# out.release()


# Python program to take 
# screenshots   



def screen_capture(filename, region=None):
    pm = get_monitors()[0]
    (x,y,w,h) = region if region != None else (pm.x, pm.y, pm.width, pm.height)
    print(str((x,y,w,h)))
    # take screenshot using pyautogui 
    image = pyautogui.screenshot(region=(x,y,w,h)) 

    # since the pyautogui takes as a  
    # PIL(pillow) and in RGB we need to  
    # convert it to numpy array and BGR  
    # so we can write it to the disk 
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
    # writing it to the disk using opencv 
    cv2.imwrite(filename, image)

# screen_capture("img.png", region=(0,0,32,23))

# Q: Can artificial neurons self organize to capture spacial data from linearizd data?
# i.e. if we input linearized data will the network automatically capture the 2D representation in their structure?
# is it worth exploring or should we just do this manually? Probs not worth it....
#
# ALGORITHM: sensing an image
# NOTE: this is only for sensing an image, 
#       there will need to be an actuator algorithm to respond to the image, 
#       perhaps this can be "look around" and we can validate by seeing the
#       image change in memory (we will need some sort of probe)
# - capture image as some typical image format
# - map image to internal data format decoding the 2D information that encoded in the size and indices of the image
# - convert to spike signals direct into sensor neurons
# - push sensor signals into short term memory neurons that are highly elastic
# - push short term memory into long term memory (in file?)
# NOTE: there may be many levels of short-to-long term memory, but at some point it needs to move to file

def image_sensor(pos=(0,0),size=(64,64)) -> np.ndarray:
    (x,y) = pos
    (w,h) = size
    image = pyautogui.screenshot(region=(x,y,w,h))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def update_stm(signal: np.ndarray, stm, ltm):
    # update the stm based on new signal and exisiting ltm and stm
    pass

def update_ltm(stm, ltm):
    # update the ltm based on exisiting ltm and stm
    pass

def image_actuator(stm, ltm):
    # apply some activation function to stm + ltm
    # if activates:
    #    move pos of image_sensor based on result
    pass



# simple neuron model
# position  : (2,) [0,j],[0,k]      # position in a numpy array
# baseline  : ()   [0,t]            # value of this neuron at rest                         
# potential : ()   [0,t]            # current value (state)                                
# threshold : ()   [n,1]            # value at which the neuron fires                      
# affinities: (j,) [0,1]            # ease at which this neuron responds to input signals       
# amplitude : ()   [0,1]            # how much this neuron affects downstream neurons
# elasticity: ()   [0,1]            # rate at which plastified afinities decay without stimulation
# plasticity: ()   [0,1]            # rate at which affinities change when stimulated (akin to learning rate)         
# stability : ()   [0,1]            # resistance to potential decay i.e. affects potential (expected to be a function of plasticity)


class Defaults:
    baseline=0
    potential=0
    threshold=0.5
    affinities=1.0
    # meta parameters, not strictly required
    amplitude=1.0
    elasticity=0.5
    plasticity=0.5
    stability=0.5

# neuron update rule sketch
# note: all of this should be performed as immutable update
# 
# for neurons `n` with inputs `i` and outputs `o`
# 
# if potential[n] > threshold[n]:
#   potential[n] = baseline[n]                                                      # normally neurons have a refractory period, perhaps a refractory period can regulate repeated stimulation
#   potential[o] += amplitude[n] * affinities[o][n]                                 # this should scale with number of input/output neurons
#   affinities[n][i] *= (1 - plasticity[i]) * (1 + potential[o] - threshold[o])     # careful, will have to clamp to minimum or dropout and clamp max as well
# 
# affinities[n][i] *= (1 - elasticity[n])                                           # this one will require a lot of tuning
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

class Model:
    ''' A spiking neural net model. 
    '''
    def __init__(self, shape, metric='cityblock'):
        self.shape          = shape
        self.size           = np.zeros(shape).size
        self.refractory     = epsilon
        self.baseline       = 0.10
        self.threshold      = 0.50
        self.amplitude      = 0.20
        self.stability      = 0.99
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
        self.outputs        = rng.choice(adjacents, self.neuron.size)
        self.affinities     = np.ones(self.outputs.shape)

    def __repr__(self):
        return "{}\n{}".format(
            reprnd(self.potential, "potential"), 
            reprnd(self.affinities, "affinities")
        )

# if potential[n] > threshold[n]:
#   potential[n] = baseline[n]                                                      # normally neurons have a refractory period, perhaps a refractory period can regulate repeated stimulation
#   potential[o] += amplitude[n] * affinities[o][n]                                 # this should scale inversely with number of input/output neurons
#   affinities[n][i] *= (1 - plasticity[i]) * (1 + potential[o] - threshold[o])     # careful, will have to clamp to minimum or dropout and clamp max as well
# 
# affinities[n][i] *= (1 - elasticity[n])                                           # this one will require a lot of tuning
# potential[n] *= (1 - stability[n])                                                # this one will require some tuning

    def update(self):
        ''' Spiking neuron update algorithm:
            ### Psuedo code:
                for each activated neuron
                    # set activated neuron potential back to the baseline
                    new.activated.potential = baseline
                    # increase output neuron potential
                    new.output.potential += activated.amplitude * output.affinity
                    # update activated neuron affinities
                    # (why do this only for activated neurons? good question)
                    new.activated.affinities += (1 + new.output.potential - output.threshold)
                # decrease all neuron potentials
                new.potential *= stability
        '''
        new_affinities = self.affinities.copy()
        new_potential = self.potential.copy()
        activated = self.potential > self.threshold
        # Set activated potentials back to refactory value
        show(new_potential, new_affinities)
        new_potential[activated] = self.refractory
        # Increase output neuron potentials
        show(new_potential, new_affinities)
        new_potential[self.outputs[activated]] += self.amplitude * self.affinities[activated]
        show(new_potential, new_affinities)
        # ndbg(self.outputs[activated,:],"out ac")
        # ndbg(new_affinities[self.outputs[activated,:]],"aff out")
        # ndbg((1 - self.plasticity) * (1 + new_potential[self.outputs[activated,:]] - self.threshold),"out pot upd")
        for o in range(self.outputs[activated].shape[1]):
            # ndbg(new_affinities[self.outputs[activated, o]] ,"aff out ac")
            # Without this loop the broadcast wouldn't work, this is what we are trying to do...
            # [act-idx, out-idx, aff] *= [act-idx, out-pot]
            # The size of output potentials is the same as the affinities since they are directly correlated.
            # Neuron n has:
            # [o0, o1, o2, ..., oN] output neurons,
            # [a0, a1, a2, ..., aN] associated affinities
            # So each output neuron also has (since they are neurons, duh):
            # [a0, a1, a2, ..., aN] associated affinities
            # The size of N depends on the nb of outputs for each neuron which currently is based on the metric used to select outputs.
            new_affinities[self.outputs[activated, o]] += self.plasticity * (new_potential[self.outputs[activated,:]] - self.threshold)

        # ndbg(self.outputs[activated],"activated outputs")
        # ndbg(new_affinities[activated],"new activated affinities")
        show(new_potential, new_affinities)
        new_potential *= self.stability
        show(new_potential, new_affinities)
        # Push the update
        self.potential = new_potential.clip(epsilon, 1-epsilon)
        self.affinities = new_affinities.clip(epsilon, 1-epsilon)

def show(potential, affinities, duration=0):
    cv2.imshow("potenital",cv2.resize(potential.reshape(shape), (512,512), interpolation=cv2.INTER_NEAREST))
    cv2.imshow("affinities",cv2.resize(affinities[:,4].reshape(shape), (512,512), interpolation=cv2.INTER_NEAREST))
    cv2.moveWindow("potenital", 50, 200)
    cv2.moveWindow("affinities", 600, 200)
    ndbg(potential.reshape(shape), "potentials")
    ndbg(affinities, "affinities")
    cv2.waitKey(duration)

def run(model, iterations):
    np.set_printoptions(linewidth=200)
    banner("RUN BEGINS")
    # FIXME: hack to get some activated neurons
    # model.potential[12] = 0.8
    # model.potential[13] = 0.6
    # model.potential[9:11] = 1
    # ndbg(model.potential.copy().reshape(model.shape), "inital pot")
    # ndbg(model.position[model.outputs[12,4]], "outputs[12]")
    # ndbg(model.position[model.outputs[13,4]], "outputs[13]")
    for _ in range(iterations):
        model.potential[rng.integers(0, model.size)] = 0.8
        model.potential[rng.integers(0, model.size)] = 0.8
        model.update()
        # ndbg(model.potential.copy().reshape(model.shape), "new pot")
        # ndbg(model.affinities, "new aff")
    cv2.waitKey(0)


# Run the model
shape = (4, 4)
iterations = 10000
model=Model(shape)

run(model, iterations)

