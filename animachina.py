import kivy

kivy.require("1.11.1")  # replace with your current kivy version !

from kivy.app import App, Widget
from kivy.graphics import Rectangle, Color, Fbo, Canvas
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

from array import array

import numpy as np


class FboTest(Widget):
    def __init__(self, **kwargs):
        super(FboTest, self).__init__(**kwargs)

        # first step is to create the fbo and use the fbo texture on other
        # rectangle

        with self.canvas:
            # create the fbo
            self.fbo = Fbo(size=(256, 256))

            # show our fbo on the widget in different size
            Color(1, 1, 1)
            Rectangle(size=(32, 32), texture=self.fbo.texture)
            Rectangle(pos=(32, 0), size=(64, 64), texture=self.fbo.texture)
            Rectangle(pos=(96, 0), size=(128, 128), texture=self.fbo.texture)

        # in the second step, you can draw whatever you want on the fbo
        with self.fbo:
            Color(1, 0, 0, 0.8)
            Rectangle(size=(256, 64))
            Color(0, 1, 0, 0.8)
            Rectangle(size=(64, 256))


class TextureBuffer(Widget):
    texture = None

    def __init__(self, pixel_buffer, texture_size, **kwargs):
        super(TextureBuffer, self).__init__(**kwargs)
        self.texture = Texture.create(size=texture_size)
        self.paint(pixel_buffer)
        with self.canvas:
            Rectangle(texture=self.texture, pos=self.pos, size=texture_size)

    def paint(self, pixel_buffer):
        self.texture.blit_buffer(pixel_buffer.flatten(), colorfmt="rgba")
        self.canvas.ask_update()


class Animachina(App):
    view = None

    blue = 0.0
    pixel_buffer = None

    def update(self, dt):
        # (height, width) = Window.size
        self.blue += 10 * dt / 60
        self.pixel_buffer[:, :] = [255, 128, int(255 * self.blue % 255), 255]
        self.view.paint(self.pixel_buffer)

    def build(self):
        (height, width) = Window.size
        self.pixel_buffer = np.ones([height, width, 4], dtype=np.uint8)
        self.pixel_buffer[:, :] = [255, 128, 0, 255]
        self.view = TextureBuffer(self.pixel_buffer, (height, width))

        Clock.schedule_interval(self.update, 1.0 / 60.0)
        return self.view


if __name__ == "__main__":
    Animachina().run()
