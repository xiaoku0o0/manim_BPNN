from manim import *
import numpy as np

color = "#f9f8ff"


class MA9(Scene):
    def construct(self):
        frame = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, -5, 1],
            y_length=8
        )
        line = []
        # line1
        axes1 = Axes(x_range=[-5, 5, 1], y_range=[0, 1, 1], x_length=4, y_length=1, tips=False).move_to(
            frame.coords_to_point(3, -1))
        line.append(
            VGroup(
                Text("阶跃函数", font_size=30, color=color).move_to(frame.coords_to_point(-5, -1)),
                Tex(r"$f(x)=\left \{  \begin{matrix} 1, & x\ge 0,\\ 0, & x<0.\end{matrix}\right .$", font_size=30, color=color).move_to(frame.coords_to_point(-2, -1)),
                axes1,
                axes1.plot(lambda x: 0, x_range=[-5, 0], color=ORANGE),
                axes1.plot(lambda x: 1, x_range=[0, 5], color=ORANGE)
            )
        )
        # line2
        axes2 = Axes(x_range=[-5, 5, 1], y_range=[0, 1, 1], x_length=4, y_length=1, tips=False).move_to(
            frame.coords_to_point(3, -2))
        line.append(
            VGroup(
                Text("sigmoid", font_size=30, color=color).move_to(frame.coords_to_point(-5, -2)),
                Tex(r"$f(x)=\frac{1}{1+e^{-x}}$", font_size=30,
                    color=color).move_to(frame.coords_to_point(-2, -2)),
                axes2,
                axes2.plot(lambda x: 1/(1+np.exp(-x)), x_range=[-5, 5], color=ORANGE)
            )
        )
        # line3
        axes3 = Axes(x_range=[-5, 5, 1], y_range=[-1, 1, 1], x_length=4, y_length=1, tips=False).move_to(
            frame.coords_to_point(3, -3))
        line.append(
            VGroup(
                Text("tanh", font_size=30, color=color).move_to(frame.coords_to_point(-5, -3)),
                Tex(r"$f(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$", font_size=30,
                    color=color).move_to(frame.coords_to_point(-2, -3)),
                axes3,
                axes3.plot(lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)), x_range=[-5, 5], color=ORANGE)
            )
        )
        # line4
        axes4 = Axes(x_range=[-5, 5, 1], y_range=[0, 5, 1], x_length=4, y_length=1, tips=False).move_to(
            frame.coords_to_point(3, -4))
        line.append(
            VGroup(
                Text("ReLu", font_size=30, color=color).move_to(frame.coords_to_point(-5, -4)),
                Tex(r"$f(x)=\max{\{0,x\}}$", font_size=30,
                    color=color).move_to(frame.coords_to_point(-2, -4)),
                axes4,
                axes4.plot(lambda x: 0, x_range=[-5, 0], color=ORANGE),
                axes4.plot(lambda x: x, x_range=[0, 5], color=ORANGE),
            )
        )
        for obj in line:
            self.play(Write(obj))
        self.wait()
