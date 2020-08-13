import cv2
import numpy as np
import pyautogui
from screeninfo import get_monitors
from scipy.spatial.distance import pdist, squareform
import time

import torch
import torchvision
import torchvision.transforms as transforms

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

