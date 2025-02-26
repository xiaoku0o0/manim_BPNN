from manim import *


class MB3(ThreeDScene):
    def construct(self):
        temp1 = Tex(
            r"""$
            \begin{bmatrix}
             x_{1} \\
             x_{2}
            \end{bmatrix}
            $""",
            font_size=35, fill_color=GREEN_D
        )
        temp2 = Tex(
            r"""$
            \begin{bmatrix}
             w_{11} & w_{12}\\
             w_{21} & w_{22}
            \end{bmatrix}
            $""",
            font_size=35, fill_color=ORANGE
        )
        temp3 = Tex(
            r"""$=
            \begin{bmatrix}
             w_{11} x_{1} + w_{12} x_{2} \\
             w_{21} x_{1} + w_{22} x_{2}
            \end{bmatrix}
            $""",
            font_size=35, fill_color=GREEN_D
        )
        dim_label = Tex(
            r"$\mathbb{R} ^2$",r"$\to \mathbb{R} ^2$", font_size=35
        )
        dim_label.next_to(VGroup(temp2, temp1, temp3).arrange(RIGHT, buff=0.1).center(), direction=DOWN, buff=0.4)
        self.play(Write(temp1))
        self.wait()
        self.play(Write(temp2))
        self.play(Write(temp3), Write(dim_label))
        self.wait()

        # 输入改成三维
        temp1_3 = Tex(
            r"""$
            \begin{bmatrix}
             x_{1} \\
             x_{2} \\
             x_{3}
            \end{bmatrix}
            $""",
            font_size=35, fill_color=GREEN_D
        ).move_to(temp1.get_center())
        temp2_3 = Tex(
            r"""$
            \begin{bmatrix}
             w_{11} & w_{12} & w_{13}\\
             w_{21} & w_{22} & w_{23}
            \end{bmatrix}
            $""",
            font_size=35, fill_color=ORANGE
        ).next_to(temp1_3, direction=LEFT, buff=0.1)
        temp3_3 = Tex(
            r"""$=
            \begin{bmatrix}
             w_{11} x_{1} + w_{12} x_{2} + w_{13} x_{3}\\
             w_{21} x_{1} + w_{22} x_{2} + w_{23} x_{3}
            \end{bmatrix}
            $""",
            font_size=35, fill_color=GREEN_D
        ).next_to(temp1_3, direction=RIGHT, buff=0.1)
        dim_label_32 = Tex(
            r"$\mathbb{R} ^3$",r"$\to \mathbb{R} ^2$", font_size=35
        ).move_to(dim_label.get_center())
        self.play(
            TransformMatchingTex(temp1, temp1_3),
            TransformMatchingTex(temp2, temp2_3),
            TransformMatchingTex(temp3, temp3_3),
            TransformMatchingTex(dim_label, dim_label_32)
        )
        self.wait()
