import kivy

kivy.require("1.11.1")  # replace with your current kivy version !

from kivy.app import App, Widget
from kivy.graphics import Rectangle, Color, Fbo, Canvas
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

from array import array

import numpy as np

rng = np.random.default_rng()

def rgba(width=128, height=128):
    return np.zeros([width, height, 4], dtype=np.uint8)


class World:
    def __init__(self, height, width):
        self.environment = np.zeros((height, width), dtype=np.uint8)

    def update(self):
        # select a batch of random cell's (maybe all?)
            # increase the selected cell's energy by some small amount
            add = np.random.choice([0,1], *self.environment.shape, p=[0.75, 0.25])
            self.environment += add
        # select a batch of random cell's (maybe all?)
            # decrease the selected cell's energy by some small amount
            rem = np.random.choice([0,1], *self.environment.shape, p=[0.75, 0.25])
            self.environment -= rem
        # select a batch of random cell's (maybe all?)
            # transfer a random % of that cell's energy to an adjacent cell
            # tfr = np.random.choice([0,1], *self.environment.shape, p=[0.75, 0.25])
            # tfr *= np.random.rand(*self.environment.shape)
            # tfr *= self.environment
            # self.environment -= tfr
            # mask = np.ma.masked_values(tfr, 0)
            # tfr = tfr[mask]
            # for (x, y), value in np.ndenumerate(tfr):
            #     dx = np.random.choice([-1,1], (1,1))
            #     s = rng.standard_normal(size=(1,1))
                # if 0 <= x <= self.environment.shape[0]:


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
        self.world = World(128,128)
        (h, w) = Window.size
        self.pbuffer = rgba(*self.world.environment.shape)
        self.view = None

    def update(self, dt):
        self.world.update()
        (h, w) = Window.size
        self.pbuffer[:, :] = [255, 255, 255, 255]
        self.view.update(self.pbuffer, self.pbuffer.shape[:2], (h, w))

    def build(self):
        self.view = TextureBuffer(self.pbuffer, self.pbuffer.shape[:2])
        Clock.schedule_interval(self.update, 1.0 / 60.0)
        return self.view


if __name__ == "__main__":
    Animachina().run()
