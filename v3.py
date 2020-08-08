import cv2
import numpy as np
import pyautogui
from screeninfo import get_monitors
from scipy.spatial.distance import pdist, squareform

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
# plasticity: ()   [0,1]            # rate at which affinities change when stimulated                       
# stability : ()   [0,1]            # resistance to potential decay i.e. affects potential


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

def reprnd(arr, name="arr"):
    return "---------------------\n{}: {}\n{}\n".format(name, arr.shape, arr.__repr__())

def repr3d(arr, name="arr"):
    arrfmt = "\n\n".join(["z=" + str(z) + "\n" + arr[...,z].__repr__() for z in range(arr.shape[2])])
    return "---------------------\n{}: {}\n{}\n".format(name, arr.shape, arrfmt)

class Model:
    ''' A spiking neural net model. 
    '''
    def __init__(self, shape, nb_inputs=1, nb_outputs=1, metric='chebyshev'):
        self.shape          = shape
        self.baseline       = 0.00 * np.ones(self.shape)
        self.potential      = 0.00 * np.ones(self.shape)
        self.threshold      = 0.50 * np.ones(self.shape)
        self.affinities     = 0.50 * np.ones(self.shape)
        self.stability      = 0.99 * np.ones(self.shape)
        indices = np.array([i for i in np.ndindex(shape)])
        distances = squareform(pdist(indices, metric))
        # A list of index neighbourhoods of distance exactly 1 from the corresponding index.
        # e.g. index_neighbourhoods[0] is a list of indices that are exactly 1 unit away from indices[0] 
        index_neighbourhoods = [
            indices[(distances[idx] - 1) == 0] 
                for idx in range(indices.shape[0])
                # This line ensures that edge indices do not have any inputs.
                # This is required so that the input arrays can be the same size for numpy.
                    if np.mod(indices[idx], np.array(shape) - 1).all()
        ]
        self.inputs = rng.choice(index_neighbourhoods, (indices.shape[0], nb_inputs))

    def __repr__(self):
        return "{}\n{}".format(
            reprnd(self.potential, "potential"), 
            reprnd(self.affinities, "affinities")
        )

    def asarray(self):
        return [self.baseline, self.potential, self.threshold, self.affinities, self.stability]

# if potential[n] > threshold[n]:
#   potential[n] = baseline[n]                                                      # normally neurons have a refractory period, perhaps a refractory period can regulate repeated stimulation
#   potential[o] += amplitude[n] * affinities[o][n]                                 # this should scale with number of input/output neurons
#   affinities[n][i] *= (1 - plasticity[i]) * (1 + potential[o] - threshold[o])     # careful, will have to clamp to minimum or dropout and clamp max as well
# 
# affinities[n][i] *= (1 - elasticity[n])                                           # this one will require a lot of tuning
# potential[n] *= (1 - stability[n])                                                # this one will require some tuning

    def update(self):
        ''' Spiking neuron update algorithm:
            ** performed immutably **
                for each activated neuron
                    # set activated neuron potential back to the baseline
                    new.activated.potential = baseline
                    # increase output neuron potential
                    new.output.potential += activated.amplitude * output.affinity
                    # update activated neuron affinities (why do this only for activated neurons? good question)
                    new.activated.affinities += (1 + new.output.potential - output.threshold)
                # decrease all neuron potentials
                new.potential -= (1 - stability)
        '''
        new_affinities = self.affinities.copy()
        
        # Set activated potentials back to baseline value
        new_potential = np.where(self.potential > self.threshold, self.baseline, self.potential)

        if self.potential > self.threshold:                                            
            new_potential[o] += affinities[o][n]                            
            new_affinities[n][i] *= (1 + potential[o] - threshold[o])

        new_potential *= (1 - self.stability)

        # Do the update
        self.potential = new_potential
        self.affinities = new_affinities

# distance metrics:
#   chebyshev
#   cityblock / manhattan

def run():

    # model = Model((4,4), 1)
    pass
    # [baseline_0, potential_0, threshold_0, affinities_0, stability_0] = 
    # model = Model(16, 2)
    # print(model)
    # baseline     = baseline_0.copy()
    # potential    = potential_0.copy()
    # threshold    = threshold_0.copy()
    # affinities   = affinities_0.copy()
    # stability    = stability_0.copy()
    # for i in range(10):
    #     [new_potential, new_affinities] = update(layer)
    #     potential = new_potential
    #     affinities = new_affinities

run()

# def dbg(fstring):
#     print("---------------\n{}\n".format(fstring.replace("=","\n",1)))

def dbg(thing):
    print("---------------\n{}\n".format(fstring.replace("=","\n",1)))

def ndbg(ndarray, label=""):
    print(reprnd(ndarray, label))

# cv2.imshow("img", image_sensor())
# cv2.waitKey(0)