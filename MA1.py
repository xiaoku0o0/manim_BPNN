from manim import *
import numpy as np


class MA1(Scene):
    """分镜头MA1"""
    def construct(self):
        def load_svg(svg_type: int, xy: list, fill: bool):
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
            return SVGMobject(path[svg_type],
                              fill_color=color[svg_type],
                              height=0.4).move_to(axes.coords_to_point(*xy))
        axes = Axes(
            x_range=[-1,5,1],
            y_range=[-1,5,1],
            tips=False
        )
        axes_label = axes.get_axis_labels(x_label="x_1", y_label="x_2")
        # 放置图形
        data_in = [[1, 1], [1, 3], [3, 1], [3, 3]]
        data_out = [[0], [0], [1], [1]]
        svg_queue = [load_svg(data_out[i][0], data_in[i], fill=False) for i in range(len(data_in))]
        self.add(*svg_queue)
        # 绘制直线
        """w1*x1 + w2*x2 - b = 0"""
        decision_w1 = ValueTracker(1)
        decision_w2 = ValueTracker(1e-4)
        decision_b = ValueTracker(2)

        def cut_line_y(x):
            """分隔直线方程，输入x(x_1)，输出y(x_2)"""
            return (decision_b.get_value() - decision_w1.get_value() * x) / decision_w2.get_value()

        def cut_line_x(y):
            """分隔直线方程，输入y(x_2)，输出x(x_1)"""
            return (decision_b.get_value() - decision_w2.get_value() * y) / decision_w1.get_value()

        def update_cut_line():
            """分割线更新函数"""
            x_range = axes.x_range[:-1]
            y_range = axes.y_range[:-1]
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
                start=axes.coords_to_point(*start_point),
                end=axes.coords_to_point(*end_point)
            )
            cut_line.stroke_width = 1
            cut_line.stroke_color = ORANGE
            return cut_line

        decision_line_obj = update_cut_line().add_updater(lambda x: x.become(update_cut_line()))
        self.play(Write(decision_line_obj))
        self.wait()


        # 载入分隔区域
        def update_area():
            x_range = axes.x_range[:-1]
            y_range = axes.y_range[:-1]
            sum_area = Polygon(axes.coords_to_point(x_range[1], y_range[1]),
                               axes.coords_to_point(x_range[0], y_range[1]),
                               axes.coords_to_point(x_range[0], y_range[0]),
                               axes.coords_to_point(x_range[1], y_range[0])
                               )  # 坐标轴所有区域（全集）
            def get_area0(x, y):
                # x, y = point[:2]
                return decision_w1.get_value() * x + decision_w2.get_value() * y - decision_b.get_value()

            cut_func = axes.plot(cut_line_y, x_range=x_range)
            area0 = axes.get_area(cut_func, x_range=x_range,
                                  bounded_graph=axes.plot(lambda x: y_range[0], x_range=x_range))
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
        svg_fill_queue = [[load_svg(0, data_in[i], fill=True) for i in range(len(data_in)) if data_out[i][0] == 0],
                          [load_svg(1, data_in[i], fill=True) for i in range(len(data_in)) if data_out[i][0] == 1]]
        self.play(Create(area0_obj), *[FadeIn(obj) for obj in svg_fill_queue[0]])
        self.wait()
        self.play(Create(area1_obj), *[FadeIn(obj) for obj in svg_fill_queue[1]])
        self.wait()
        self.play(decision_w1.animate.set_value(2),
                  decision_w2.animate.set_value(-0.6),
                  decision_b.animate.set_value(2),
                  run_time=1)
        self.play(decision_w1.animate.set_value(2),
                  decision_w2.animate.set_value(1),
                  decision_b.animate.set_value(6),
                  run_time=1)
        self.wait()
