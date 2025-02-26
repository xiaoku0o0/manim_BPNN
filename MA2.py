from manim import *
import numpy as np


font_size = 30
font_type = "Times New Roman"

class MA2(Scene):
    """分镜头MA2"""
    def construct(self):
        def load_svg(svg_type: int, xy: list, fill: bool, color_direct: ManimColor = None) -> SVGMobject:
            """加载svg图形"""
            """
            svg_type: 0-圆形; 1-三角形
            xy: 坐标轴上的xy坐标
            fill: 是否填充
            """
            if fill:
                path = [r".\svg_files\圆形_填充.svg", r".\svg_files\三角形_填充.svg"]
            else:
                path = [r".\svg_files\圆形.svg", r".\svg_files\三角形.svg"]
            color = [BLUE_B, YELLOW_B]
            if color_direct is None:
                return SVGMobject(path[svg_type],
                                  fill_color=color[svg_type],
                                  height=0.4).move_to(axes_frame.coords_to_point(*xy))
            return SVGMobject(path[svg_type],
                              fill_color=color_direct,
                              height=0.4).move_to(axes_frame.coords_to_point(*xy))
        axes_frame = Axes(
            x_range=[-1,5,1],
            y_range=[-1,5,1],
            tips=False
        )
        axes_label = axes_frame.get_axis_labels(x_label="x_1", y_label="x_2")
        for obj in axes_label:
            obj.font_size = font_size   # 调整坐标轴标签字号
        # 放置图形
        data_in = [[1, 1], [1, 3], [3, 1], [3, 3]]
        data_out = [[0], [0], [1], [1]]
        svg_queue = [load_svg(data_out[i][0], data_in[i], fill=False) for i in range(len(data_in))]
        self.add(*svg_queue)
        # 绘制直线
        """w1*x1 + w2*x2 - b = 0"""
        decision_w1 = ValueTracker(2)
        decision_w2 = ValueTracker(1)
        decision_b = ValueTracker(6)

        def cut_line_y(x):
            """分隔直线方程，输入x(x_1)，输出y(x_2)"""
            return (decision_b.get_value() - decision_w1.get_value() * x) / decision_w2.get_value()

        def cut_line_x(y):
            """分隔直线方程，输入y(x_2)，输出x(x_1)"""
            return (decision_b.get_value() - decision_w2.get_value() * y) / decision_w1.get_value()

        def update_cut_line():
            """分割线更新函数"""
            x_range = axes_frame.x_range[:-1]
            y_range = axes_frame.y_range[:-1]
            # 以下代码防止直线超过坐标轴范围
            if y_range[0] <= cut_line_y(x_range[0]) <= y_range[1]:
                start_point = [x_range[0], cut_line_y(x_range[0])]  # 若左端点在y区间内，则以x为依据给出左端点
            elif cut_line_y(x_range[0]) > y_range[1]:
                start_point = [cut_line_x(y_range[1]), y_range[1]]  # 若左端点在y区间上侧，则以y上边界给出左端点
            else:
                start_point = [cut_line_x(y_range[0]), y_range[0]]  # 若左端点在y区间下侧，则以y下边界给出左端点
            if y_range[0] <= cut_line_y(x_range[1]) <= y_range[1]:
                end_point = [x_range[1], cut_line_y(x_range[1])]
            elif cut_line_y(x_range[1]) > y_range[1]:
                end_point = [cut_line_x(y_range[1]), y_range[1]]
            else:
                end_point = [cut_line_x(y_range[0]), y_range[0]]
            cut_line = Line(
                start=axes_frame.coords_to_point(*start_point),
                end=axes_frame.coords_to_point(*end_point)
            )
            cut_line.stroke_width = 1
            cut_line.stroke_color = ORANGE
            return cut_line

        decision_line_obj = update_cut_line().add_updater(lambda x: x.become(update_cut_line()))
        self.add(decision_line_obj)
        self.wait()

        # 载入分隔区域
        def update_area():
            """分割区域更新函数"""
            x_range = axes_frame.x_range[:-1]
            y_range = axes_frame.y_range[:-1]
            sum_area = Polygon(axes_frame.coords_to_point(x_range[1], y_range[1]),
                               axes_frame.coords_to_point(x_range[0], y_range[1]),
                               axes_frame.coords_to_point(x_range[0], y_range[0]),
                               axes_frame.coords_to_point(x_range[1], y_range[0])
                               )  # 坐标轴所有区域（全集）
            def get_area0(x, y):
                # x, y = point[:2]
                return decision_w1.get_value() * x + decision_w2.get_value() * y - decision_b.get_value()

            cut_func = axes_frame.plot(cut_line_y, x_range=x_range)
            area0 = axes_frame.get_area(cut_func, x_range=x_range,
                                        bounded_graph=axes_frame.plot(lambda x: y_range[0], x_range=x_range))
            area0 = Intersection(sum_area, area0, fill_color=BLUE_B, fill_opacity=0.5, stroke_opacity=False)
            area1 = Difference(sum_area, area0, fill_color=YELLOW_B, fill_opacity=0.5, stroke_opacity=False)
            # 判断两区域填色是否正确
            origin_type = 1 if get_area0(0, 0) > 0 else 0
            if (origin_type == 1 and cut_line_y(0) > 0) or (origin_type == 0 and cut_line_y(0) < 0):
                area0.fill_color = YELLOW_B
                area1.fill_color = BLUE_B
            return [area0, area1]

        area0_obj, area1_obj = update_area()
        area0_obj.add_updater(lambda x: x.become(update_area()[0]))
        area1_obj.add_updater(lambda x: x.become(update_area()[1]))

        axes = axes_frame.copy()
        self.play(Write(axes))
        self.play(Write(axes_label))

        # 展示各数据坐标
        data_coordinate_queue = [
            Tex(
                "$($", f"${data_in[i][0]}$", "$,$", f"${data_in[i][1]}$", "$)$", font_size=font_size
            ).next_to(svg_queue[i], direction=RIGHT, buff=0.1)
            for i in range(len(data_in))]
        self.play(*[Write(obj) for obj in data_coordinate_queue])
        self.wait()

        # 展示直线方程
        line_func = Tex(
            "$w_{1}$", "$x_{1}$", "$+w_{2}$", "$x_{2}$", "$-b$", "$=0$", font_size=font_size, color=ORANGE
        ).next_to(decision_line_obj.get_start(), direction=RIGHT, buff=0.1)
        self.play(Write(line_func))
        self.wait()

        # 展示右上角图例
        mark_obj = VGroup(
            Tex(r"$0 \longrightarrow \bigcirc $", font_size=font_size, color=BLUE_B),
            Tex(r"$1 \longrightarrow \bigtriangleup $", font_size=font_size, color=YELLOW_B)
        ).arrange(direction=DOWN, aligned_edge=LEFT).to_edge(UR, buff=0.5)
        mark_obj += SurroundingRectangle(
            mark_obj, corner_radius=0.1, buff=0.2, color=MAROON_B, stroke_width=1
        )    # 添加边框
        self.play(Write(mark_obj))
        self.wait()

        # 添加判断标准（映射关系）
        judge_obj = Tex(
            "$z=f(x_{1},x_{2})$", font_size=font_size, color=MAROON_B
        ).next_to(mark_obj, direction=DOWN, buff=0.3)
        self.play(Write(judge_obj))

        # 点的坐标代入直线方程
        temp1 = Tex(
            "$w_{1}$", r"$\cdot$", f"${data_in[1][0]}$", "$+w_{2}$", r"$\cdot$", f"${data_in[1][1]}$", "$-b$",
            font_size=font_size, color=BLUE_B
        ).next_to(svg_queue[1], direction=DOWN, buff=0.3).shift(0.5*LEFT)
        self.play(TransformMatchingTex(
            VGroup(line_func.copy(), data_coordinate_queue[1].copy()), temp1
        ))
        self.wait()

        # 判断比0大还是比0小
        temp2 = Tex(
            r"""$
            \left \{
            \begin{matrix}
             <0 \longrightarrow 0\\
             >0 \longrightarrow 1
            \end{matrix}
            \right .
            $""", font_size=font_size, color=BLUE_B
        ).next_to(temp1, direction=RIGHT, buff=0.2)
        self.play(Write(temp2))
        self.wait()

        # 将函数和直线方程复合
        temp3 = Tex(
            "$z=f(x_{1},x_{2})$", "$=g($", "$w_{1}$", "$x_{1}$", "$+w_{2}$", "$x_{2}$", "$-b$", "$)$",
            font_size=font_size, color=MAROON_B,
        ).next_to(temp1, direction=DOWN, buff=0.5).shift(1*RIGHT)
        self.play(TransformMatchingTex(
            VGroup(line_func.copy(), judge_obj.copy()), temp3
        ))
        self.wait()

        # 展示阶跃函数细节
        axes_temp = Axes(
            x_range=[-2,2,1],
            y_range=[-0.5,1.5,1],
            x_length=2,
            y_length=1,
            y_axis_config={
                "numbers_to_include": np.array([0, 1])
            },
            tips=False
        ).move_to(axes_frame.coords_to_point(4.5, 2))
        jump_func_obj0 = axes_temp.plot(lambda x: 0, x_range=[axes_temp.x_range[0], 0], color=BLUE_D)
        jump_func_obj1 = axes_temp.plot(lambda x: 1, x_range=[0, axes_temp.x_range[1]], color=YELLOW_D)
        self.play(Write(axes_temp))
        self.play(Write(jump_func_obj1))
        self.wait()
        self.play(Write(jump_func_obj0))
        self.wait()

        # 转场
        self.play(FadeOut(axes_temp),
                  FadeOut(jump_func_obj0),
                  FadeOut(jump_func_obj1))
        self.wait()

        # ============================================= #
        # 接EA1镜头
        # ============================================= #

        self.play(FadeOut(temp1),
                  FadeOut(temp2),
                  FadeOut(temp3))
        # 随机产生w1 w2 b
        line_arg_mark_obj = VGroup(
            Tex("$w_{1}=$", font_size=font_size, color=ORANGE),
            Tex("$w_{2}=$", font_size=font_size, color=ORANGE),
            Tex("$b=$", font_size=font_size, color=ORANGE)
        ).arrange(DOWN, aligned_edge=RIGHT).to_edge(UL, buff=0.5)
        def update_line_arg_obj():
            """直线参数标签更新函数"""
            temp = VGroup(
                Text(f"{decision_w1.get_value():.3f}", font=font_type, font_size=font_size-4, color=ORANGE),
                Text(f"{decision_w2.get_value():.3f}", font=font_type, font_size=font_size-4, color=ORANGE),
                Text(f"{decision_b.get_value():.3f}", font=font_type, font_size=font_size-4, color=ORANGE),
            )
            # 对齐
            for idx in range(len(temp)):
                temp[idx].next_to(line_arg_mark_obj[idx], direction=RIGHT, buff=0.1)
            return temp
        line_arg_obj = update_line_arg_obj().add_updater(lambda x: x.become(update_line_arg_obj()))
        line_arg_frame = SurroundingRectangle(
            VGroup(line_arg_mark_obj, line_arg_obj), corner_radius=0, buff=0.2, color=ORANGE, stroke_width=1
        )
        self.play(
            decision_w1.animate.set_value(3.892),
            decision_w2.animate.set_value(2.371),
            decision_b.animate.set_value(8.702),
            Write(line_arg_mark_obj),
            Write(line_arg_obj),
            Write(line_arg_frame)
        )
        self.wait()

        # 添加误差
        def cal_err() -> list[bool]:
            """计算误差，返回每项是否正确分类"""
            def jump_func(x: float) -> int:
                """阶跃函数"""
                return 1 if x > 0 else 0
            res = []
            for idx in range(len(data_in)):
                res.append(
                    jump_func(
                        decision_w1.get_value()*data_in[idx][0] + decision_w2.get_value()*data_in[idx][1] - decision_b.get_value()
                    ) == data_out[idx][0]
                )
            return res
        def update_svg_fill(idx) -> SVGMobject:
            """svg填色根据误差更新"""
            err_ls = cal_err()
            return load_svg(data_out[idx][0], data_in[idx], True, GREEN_D if err_ls[idx] else RED_D)
        svg_fill_queue = [update_svg_fill(i).add_updater(lambda x, idx=i: x.set_color(GREEN_D if cal_err()[idx] else RED_D)) for i in range(len(data_in))]
        def update_err_label() -> VGroup:
            """更新误差标签"""
            temp = VGroup(
                Text(
                    "误差:", color=RED_D, font_size=font_size-4
                ).next_to(line_arg_frame, direction=DOWN, buff=0.3)
            )
            temp += Text(
                    f"{len(cal_err())-sum(cal_err())}", font=font_type, color=RED_D, font_size=font_size-4
                ).next_to(temp[0], direction=RIGHT, buff=0.2)
            return temp
        err_label_obj = update_err_label().add_updater(lambda x: x.become(update_err_label()))
        self.play(
            FadeIn(area0_obj),
            FadeIn(area1_obj),
            *[FadeIn(obj) for obj in svg_fill_queue],
            Write(err_label_obj)
        )
        self.wait()

        # 调整参数
        self.play(
            decision_w1.animate.set_value(2),
            decision_w2.animate.set_value(1),
            decision_b.animate.set_value(6),
            run_time=2
        )
        self.wait()
