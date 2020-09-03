from abc import abstractclassmethod


class Sensor:
    @abstractclassmethod
    def obs(self, env):
        raise NotImplementedError
    
class Actuator:
    @abstractclassmethod
    def act(self) -> object:
        raise NotImplementedError

class Agent:    
    def __init__(self, sensor: Sensor, actuator: Actuator, reward_sensor: Sensor):
        self.sensor = sensor
        self.acuator = actuator
        self.reward_sensor = reward_sensor

    def observe(self, observation):
        self.sensor.obs(observation)

    def reward(self, reward) -> object:
        return self.reward_sensor.obs(reward)

    def act(self) -> object:
        return self.acuator.act()


Id = int

class Node:
    ''' A component of a neural network. '''
    def __init__(self, id: Id, label: str = None):
        self.id = id
        self.label = label
