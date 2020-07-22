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

import cv2

rng = np.random.default_rng()
f32 = np.float32
i32 = np.int32
u32 = np.uint32
u8 = np.uint8

class FrameMonitor:
    ''' Logs frame time. '''
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

frame_monitor = FrameMonitor()

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
    def __init__(self, data: np.ndarray):
        self.shape = data.shape
        self.environment = data

    def update(self):
        # select a batch of random cell's (maybe all?)
            # increase the selected cell's energy by some small amount
        # select a batch of random cell's (maybe all?)
            # decrease the selected cell's energy by some small amount
            add = np.random.choice(np.array([-1,0,1]), self.environment.shape, p=dist([1,10,1]))
            add = add * rng.random(self.environment.shape, dtype=f32)
            self.environment = np.array(np.add(add, self.environment).clip(0, np.finfo(f32).max), dtype=f32)

        # select a batch of random cell's (maybe all?)
            # transfer a random % of that cell's energy to an adjacent cell
            tfr = np.random.choice([0,1], self.environment.shape, p=dist([10, 1]))
            tfr = tfr * rng.random(self.environment.shape, dtype=f32) * self.environment

            # print(tfr)
            self.environment = np.array(np.subtract(self.environment, tfr).clip(0, np.finfo(f32).max), dtype=f32)

            # I think this is working and its faster but can be way more efficient
            indices = tfr.nonzero() #.astype(np.uint32)
            print(np.indices(self.environment.shape).shape)
            values = tfr[indices]
            # this especially can be more effcient
            offsets = rng.standard_normal(size=tfr.shape).clip(-1,1).round().flatten().take(indices).astype(i32)
            offsets = (offsets[0], offsets[1])
            # This clipping assumes a square environment
            neighbors = np.add(offsets, indices).clip(0,self.environment.shape[0] - 1)
            neighbors = (neighbors[0], neighbors[1])
            self.environment[neighbors] += values


            # I think this was working but it was slow
            # for (x, y), value in np.ndenumerate(tfr):
            #     (dx,dy) = np.add(rng.standard_normal(size=(2)).round(), (x,y)).clip(0,self.environment.shape[0]-1).astype(np.uint32)
            #     self.environment[dx,dy] += value



class TextureBuffer(Widget):
    def __init__(self, pixel_buffer, texture_size, **kwargs):
        super(TextureBuffer, self).__init__(**kwargs)
        self.texture = None
        self.rect = None
        self.update(pixel_buffer, texture_size)

    def update(self, pixel_buffer, texture_size, canvas_size=None):
        canvas_size = texture_size if canvas_size is None else canvas_size
        self.texture = Texture.create(size=texture_size)
        self.texture.mag_filter = 'nearest'
        self.texture.min_filter = 'nearest'
        self.texture.blit_buffer(pixel_buffer.tobytes(), colorfmt="rgba")
        self.canvas.clear()
        with self.canvas:
            self.rect = Rectangle(
                texture=self.texture, pos=self.pos, size=canvas_size
            )
        self.canvas.ask_update()


class Animachina(App):

    def __init__(self, persistent_data, **kwargs):
        super(Animachina, self).__init__(**kwargs)
        self.persistent_data = persistent_data
        self.world = World(persistent_data.data)
        self.pbuffer = rgba(*self.world.shape)
        self.view = None
        # self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')

    def update(self, dt):
        with frame_monitor:
            self.world.update()
            self.persistent_data.data = self.world.environment
            scale = 256/16
            self.pbuffer = scale * np.full((*self.world.shape, 4), [1,1,1,1], dtype=f32)
            self.pbuffer[:,:] *= self.world.environment[:,:,None]
            self.pbuffer = self.pbuffer.clip(0,255).astype(u8)
            self.view.update(self.pbuffer, self.pbuffer.shape[:2], Window.size)


    def build(self):
        Window.size = (512,512)
        Window.left = 100
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


class PersistentData:
    ''' Loads and saves data in image formats. '''
    def __init__(self, load_path=None, save_path=None):
        self.load_path = load_path
        self.data = cv2.imread(load_path, cv2.IMREAD_UNCHANGED) if load_path is not None else None
        if self.data is not None:
            self.data = cv2.flip(self.data,0)
        self.save_path = save_path if save_path is not None else load_path

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.data = cv2.flip(self.data, 0)
        cv2.imwrite(self.save_path, self.data)

if __name__ == "__main__":
    # NOTE: Saved image will be flipped vertically to correct for OpenGL NDC data coordinates, this may make calculations confusing.
    persistent_data = PersistentData(load_path="data/world.png")
    if persistent_data.data is None:
        persistent_data.data = np.zeros((256,256), dtype=f32)

    with persistent_data:
        Animachina(persistent_data).run()
