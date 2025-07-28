from manim import *
from manim.opengl import *


class PhenolVsFuran(Scene):
    def construct(self):
        hello_world = Text("Phenol vs Furan", font_size=72)
        self.play(Write(hello_world))
        # self.play(
        #     self.camera.animate.set_euler_angles(
        #         theta=-10*DEGREES,
        #         phi=30*DEGREES,
        #     )
        # )
        self.play(FadeOut(hello_world))