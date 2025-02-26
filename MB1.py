from manim import *


class MB1(ThreeDScene):
    def construct(self):
        axes = NumberPlane().add_coordinates()
        axes_bef = NumberPlane().add_coordinates()
        def update_base_vector():
            """更新基向量"""
            return VGroup(
                Vector(axes.coords_to_point(1, 0), color=MAROON_D),
                Vector(axes.coords_to_point(0, 1), color=TEAL_D),
            )
        def update_base_vector_label():
            """更新基向量坐标显示"""
            i_hat = axes_bef.point_to_coords(base_vector[0].get_end())
            j_hat = axes_bef.point_to_coords(base_vector[1].get_end())
            return VGroup(
                Tex(
                    r"$\begin{bmatrix}"+rf"{i_hat[0]:.2f}\\{i_hat[1]:.2f}"+r"\end{bmatrix}$",
                    font_size=30, color=MAROON_D
                ).next_to(base_vector[0].get_end(), direction=RIGHT, buff=0.1),
                Tex(
                    r"$\begin{bmatrix}"+rf"{j_hat[0]:.2f}\\{j_hat[1]:.2f}"+r"\end{bmatrix}$",
                    font_size=30, color=TEAL_D
                ).next_to(base_vector[1].get_end(), direction=RIGHT, buff=0.1)
            )
        def update_label():
            """更新变换矩阵"""
            i_hat = axes_bef.point_to_coords(base_vector[0].get_end())
            j_hat = axes_bef.point_to_coords(base_vector[1].get_end())
            return Tex(
                r"$\text{变换矩阵}\begin{bmatrix}"+rf"{i_hat[0]:.2f}&{j_hat[0]:.2f}\\{i_hat[1]:.2f}&{j_hat[1]:.2f}"+r"\end{bmatrix}$",
                tex_template=TexTemplateLibrary.ctex,
                font_size=30, color=ORANGE, fill_opacity=0.8, background_stroke_color=BLACK
            ).to_edge(UL, buff=0.5)
        def update_dot():
            return Circle(
                radius=0.1, color=BLACK, stroke_width=2, fill_opacity=True, fill_color=GREEN_D
            ).move_to(axes.coords_to_point(1, 3))
        base_vector = update_base_vector().add_updater(lambda x: x.become(update_base_vector()))
        base_vector_label = update_base_vector_label().add_updater(lambda x: x.become(update_base_vector_label()))
        label = update_label().add_updater(lambda x: x.become(update_label()))
        dot = update_dot().add_updater(lambda x: x.become(update_dot()))
        self.play(Write(axes))
        self.play(Write(base_vector), Write(base_vector_label), Write(label), Write(dot))
        self.wait()
        self.play(axes.animate.apply_matrix([[1, 1], [0, 1]], about_point=axes.coords_to_point(0, 0)))
        self.wait()

        # 变回去
        self.play(
            axes.animate.apply_matrix([[1, -1], [0, 1]], about_point=axes.coords_to_point(0, 0)),
        )
        self.wait()

        # 映射到三维空间上
        axes3d = ThreeDAxes(
            x_range=[-7, 7, 1],
            y_range=[-4, 4, 1],
            z_range=[-7, 7, 1],
            x_length=14,
            y_length=8,
            z_length=14
        )
        def update_base_vector_label_aft():
            """更新基向量坐标显示"""
            i_hat = axes3d.point_to_coords(base_vector[0].get_end())
            j_hat = axes3d.point_to_coords(base_vector[1].get_end())
            return VGroup(
                Tex(
                    r"$\begin{bmatrix}"+rf"{i_hat[0]:.2f}\\{i_hat[1]:.2f}\\{i_hat[2]:.2f}"+r"\end{bmatrix}$",
                    font_size=30, color=MAROON_D
                ).next_to(base_vector[0].get_end(), direction=RIGHT, buff=0.1),
                Tex(
                    r"$\begin{bmatrix}"+rf"{j_hat[0]:.2f}\\{j_hat[1]:.2f}\\{j_hat[2]:.2f}"+r"\end{bmatrix}$",
                    font_size=30, color=TEAL_D
                ).next_to(base_vector[1].get_end(), direction=RIGHT, buff=0.1)
            )
        def update_label_aft():
            """更新变换矩阵"""
            i_hat = axes3d.point_to_coords(base_vector[0].get_end())
            j_hat = axes3d.point_to_coords(base_vector[1].get_end())
            return Tex(
                r"$\text{变换矩阵}\begin{bmatrix}"+rf"{i_hat[0]:.2f}&{j_hat[0]:.2f}\\{i_hat[1]:.2f}&{j_hat[1]:.2f}\\{i_hat[2]:.2f}&{j_hat[2]:.2f}"+r"\end{bmatrix}$",
                tex_template=TexTemplateLibrary.ctex,
                font_size=30, color=ORANGE, fill_opacity=0.8, background_stroke_color=BLACK
            ).to_edge(UL, buff=0.5)
        def update_dot_aft():
            return Dot3D(
                radius=0.1, color=GREEN_D
            ).move_to(axes.coords_to_point(1, 3))
        base_vector_label_aft = update_base_vector_label_aft().add_updater(lambda x: x.become(update_base_vector_label_aft()))
        label_aft = update_label_aft().add_updater(lambda x: x.become(update_label_aft()))
        dot_aft = update_dot_aft().add_updater(lambda x: x.become(update_dot_aft()))
        self.add(axes3d)
        self.remove(base_vector_label, label, dot)
        self.add(dot_aft)
        self.add(base_vector_label_aft)
        self.add_fixed_in_frame_mobjects(label_aft)
        self.wait()
        self.move_camera(phi=60*DEGREES, theta=-60*DEGREES, added_anims=[
            axes.animate.apply_matrix([[1, 1], [0, 1], [-1, 1]], about_point=axes.coords_to_point(0, 0))
        ])
        self.begin_3dillusion_camera_rotation(rate=0.1)
        self.wait(10)
