import kivy

kivy.require("1.11.1")  # replace with your current kivy version !

from kivy.app import App, Widget
from kivy.graphics import Rectangle, Color, Fbo, Canvas
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

from array import array

import numpy as np
import time
import sys

rng = np.random.default_rng()
f32 = np.float32
u8 = np.uint8

def fbounds(dtype):
    ''' Bounds for a floating point data type. '''
    return (np.finfo(f32).min, np.finfo(f32).max)

def rgba(width=128, height=128):
    ''' RGBA array of zeros. '''
    return np.zeros([width, height, 4], dtype=u8)

def dist(a):
    ''' Probability distribution array from parts. '''
    return np.array(a)/np.sum(a)

class World:
    def __init__(self, height, width):
        self.size = (height, width)
        self.environment = np.zeros(self.size, dtype=f32)

    def update(self):
        # select a batch of random cell's (maybe all?)
            # increase the selected cell's energy by some small amount
            add = np.random.choice(np.array([-1,0,1]), self.environment.shape, p=dist([1,4,1]))
            add = add * rng.random(self.environment.shape, dtype=f32)
            self.environment = np.array(np.add(add, self.environment).clip(0, np.finfo(f32).max), dtype=f32)

        # select a batch of random cell's (maybe all?)
            # decrease the selected cell's energy by some small amount
            # rem = np.random.choice(np.array([0,1], dtype=np.uint16), self.environment.shape, p=[0.75, 0.25])
            # self.environment = np.array(np.subtract(self.environment, rem).clip(0,255), dtype=np.uint8)

        # select a batch of random cell's (maybe all?)
            # transfer a random % of that cell's energy to an adjacent cell
            # tfr = np.random.choice([0,1], self.environment.shape, p=dist([10, 1]))
            # tfr = tfr * rng.random(self.environment.shape, dtype=np.float32)
            # tfr = tfr * self.environment
            # print(tfr)
            # self.environment = np.array(np.subtract(self.environment, tfr).clip(0,255), dtype=np.uint8)
            # self.environment -= tfr
            # mask = np.ma.masked_values(tfr, 0)
            # tfr = tfr[mask]
            # for (x, y), value in np.ndenumerate(tfr):
            #     dx = np.random.choice([-1,1], (1,1))
            #     s = rng.standard_normal(size=(1,1))
            #     if 0 <= x <= self.environment.shape[0]:


class TextureBuffer(Widget):
    def __init__(self, pixel_buffer, texture_size, **kwargs):
        super(TextureBuffer, self).__init__(**kwargs)
        self.texture = None
        self.rect = None
        self.update(pixel_buffer, texture_size)

    def update(self, pixel_buffer, texture_size, canvas_size=None):
        canvas_size = texture_size if canvas_size is None else canvas_size
        self.texture = Texture.create(size=texture_size)
        self.texture.blit_buffer(pixel_buffer.tobytes(), colorfmt="rgba")
        self.canvas.clear()
        with self.canvas:
            self.rect = Rectangle(
                texture=self.texture, pos=self.pos, size=canvas_size
            )
        self.canvas.ask_update()


class Animachina(App):

    def __init__(self, **kwargs):
        super(Animachina, self).__init__(**kwargs)
        self.frame_monitor = FrameMonitor()
        self.world = World(256,256)
        self.pbuffer = rgba(*self.world.size)
        self.view = None
        # self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')

    def update(self, dt):
        with self.frame_monitor:
            self.world.update()
            speed = 10
            self.pbuffer = speed * np.full((*self.world.size, 4), [1,1,1,1], dtype=f32)
            self.pbuffer[:,:] *= self.world.environment[:,:,None]
            self.pbuffer = self.pbuffer.clip(0,255).astype(u8)
            # print(self.pbuffer)
            # self.pbuffer = self.world.environment[:,:]
            # print(self.pbuffer)
            self.view.update(self.pbuffer, self.pbuffer.shape[:2], Window.size)


    def build(self):
        self.view = TextureBuffer(self.pbuffer, self.pbuffer.shape[:2])
        Clock.schedule_interval(self.update, 0)
        return self.view

    # def _keyboard_closed(self):
    #     self._keyboard.unbind(on_key_down=self._on_keyboard_down)
    #     self._keyboard = None

    # def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
    #     print('The key', keycode, 'have been pressed')
    #     print(' - text is %r' % text)
    #     print(' - modifiers are %r' % modifiers)

    #     # Keycode is composed of an integer + a string
    #     # If we hit escape, release the keyboard
    #     if keycode[1] == 'escape':
    #         keyboard.release()

    #     # Return True to accept the key. Otherwise, it will be used by the system.
    #     return True

class FrameMonitor:
    def __init__(self):
        self.frame_time = 0
        self.frame_number = 0

    def __enter__(self):
        self.frame_number += 1
        self.frame_number %= 30
        self.frame_start = time.time_ns()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.frame_number == 0:
            self.frame_time = (self.frame_time + (time.time_ns() - self.frame_start) / 1000000)/2
            sys.stdout.write("frame_time = {:3.9f} ms\r".format(self.frame_time))
            sys.stdout.flush()

if __name__ == "__main__":
    Animachina().run()
