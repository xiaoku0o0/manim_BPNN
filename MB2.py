from manim import *


class MB2(ThreeDScene):
    def construct(self):
        # 下侧文字
        temp1 = Tex(
            r"""$
            \begin{bmatrix}
             1 \\
             3
            \end{bmatrix}
            $""",
            font_size=35, fill_color=GREEN_D
        )
        temp2 = Tex(
            r"""$
            \begin{bmatrix}
             1 & 1\\
             0 & 1
            \end{bmatrix}
            $""",
            font_size=35, fill_color=ORANGE
        )
        temp3 = Tex(
            r"""$=
            \begin{bmatrix}
             1 \times 1 + 1 \times 3 \\
             0 \times 1 + 1 \times 3
            \end{bmatrix}
            =
            \begin{bmatrix}
             4 \\
             3
            \end{bmatrix}
            $""",
            font_size=35, fill_color=GREEN_D
        )
        temp_group = VGroup(temp2, temp1, temp3).arrange(direction=RIGHT, buff=0.1).move_to((-2, 0, 0))
        frame_obj = SurroundingRectangle(temp_group, buff=0.5, fill_opacity=0.8, fill_color=BLACK, stroke_opacity=False)
        self.play(FadeIn(frame_obj), Write(temp1))
        self.wait()
        self.play(Write(temp2))
        self.wait()
        self.play(Write(temp3))
        self.wait()

        # 变回去
        self.play(
            FadeOut(temp2), FadeOut(temp3)
        )

        # 映射到三维空间上
        temp2_aft = Tex(
            r"""$
            \begin{bmatrix}
             1 & 1\\
             0 & 1\\
             -1 & 1
            \end{bmatrix}
            $""",
            font_size=35, fill_color=ORANGE
        ).next_to(temp1, direction=LEFT, buff=0.1)
        temp3_aft = Tex(
            r"""$=
            \begin{bmatrix}
             1 \times 1 + 1 \times 3 \\
             0 \times 1 + 1 \times 3 \\ 
             -1 \times 1 + 1 \times 3
            \end{bmatrix}
            =
            \begin{bmatrix}
             4 \\
             3 \\
             2
            \end{bmatrix}
            $""",
            font_size=35, fill_color=GREEN_D
        ).next_to(temp1, direction=RIGHT, buff=0.1)
        frame_obj_aft = SurroundingRectangle(VGroup(temp2_aft, temp1, temp3_aft), buff=0.5, fill_opacity=0.8, fill_color=BLACK, stroke_opacity=False)
        self.play(Write(temp2_aft), Write(temp3_aft), ReplacementTransform(frame_obj, frame_obj_aft))
        self.wait()
