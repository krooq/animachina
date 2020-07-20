import kivy

kivy.require("1.11.1")  # replace with your current kivy version !

from kivy.app import App, Widget
from kivy.graphics import Rectangle, Color, Fbo, Canvas
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

from array import array

import numpy as np


class TextureBuffer(Widget):
    texture_size = None
    texture = None
    rect = None

    def __init__(self, pixel_buffer, texture_size, **kwargs):
        super(TextureBuffer, self).__init__(**kwargs)
        self.update(pixel_buffer, texture_size)
        with self.canvas:
            self.rect = Rectangle(
                texture=self.texture, pos=self.pos, size=self.texture_size
            )

    def update(self, pixel_buffer, texture_size):
        self.texture = Texture.create(size=texture_size)
        self.texture.blit_buffer(pixel_buffer.flatten(), colorfmt="rgba")
        if self.rect is not None:
            self.rect.texture = self.texture
            self.rect.size = self.texture.size
        self.canvas.ask_update()


class Animachina(App):
    view = None

    blue = 0.0
    pixel_buffer = None

    def update(self, dt):
        self.blue += 10 * dt / 60
        self.pixel_buffer[:, :] = [255, 128, int(255 * self.blue % 255), 255]
        self.view.update(self.pixel_buffer, Window.size)

    def build(self):
        (height, width) = Window.size
        self.pixel_buffer = np.zeros([height, width, 4], dtype=np.uint8)
        self.view = TextureBuffer(self.pixel_buffer, (height, width))

        Clock.schedule_interval(self.update, 1.0 / 60.0)
        return self.view


if __name__ == "__main__":
    Animachina().run()
