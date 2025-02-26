from manim import *
import numpy as np


class MA3(Scene):
    """分镜头MA3"""
    def construct(self):
        # 展示阶跃函数细节
        axes_temp = Axes(
            x_range=[-2, 2, 1],
            y_range=[-0.5, 1.5, 1],
            x_length=6,
            y_length=3,
            y_axis_config={
                "numbers_to_include": np.array([0, 1])
            },
            tips=False
        ).center()
        jump_func_obj0 = axes_temp.plot(lambda x: 0, x_range=[axes_temp.x_range[0], 0], color=BLUE_D)
        jump_func_obj1 = axes_temp.plot(lambda x: 1, x_range=[0, axes_temp.x_range[1]], color=YELLOW_D)
        self.play(Write(axes_temp))
        self.play(Write(jump_func_obj0))
        self.play(Write(jump_func_obj1))
        self.wait()

        # 点动
        x_position = ValueTracker(-1.7)
        def update_dot():
            dot = Dot(
                point=axes_temp.coords_to_point(x_position.get_value(),
                                                0 if x_position.get_value() < 0 else 1),
                color=BLUE_D if x_position.get_value() < 0 else YELLOW_D
            )
            return dot
        dot = update_dot().add_updater(lambda x: x.become(update_dot()))
        self.add(dot)
        self.play(x_position.animate(run_time=2).set_value(1.7))
        self.wait()

