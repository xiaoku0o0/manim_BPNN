from manim import *
import numpy as np


font_size = 30
font_type = "Times New Roman"

class MA4(Scene):
    """分镜头MA4"""
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
            x_length=6,
            y_length=6,
            tips=False
        ).move_to((-3, 0, 0))
        axes_label = axes_frame.get_axis_labels(x_label="x_1", y_label="x_2")
        self.add(axes_frame, axes_label)
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
            area0 = Intersection(sum_area, area0, fill_color=BLUE_B, fill_opacity=0.1, stroke_opacity=False)
            area1 = Difference(sum_area, area0, fill_color=YELLOW_B, fill_opacity=0.1, stroke_opacity=False)
            # 判断两区域填色是否正确
            origin_type = 1 if get_area0(0, 0) > 0 else 0
            if (origin_type == 1 and cut_line_y(0) > 0) or (origin_type == 0 and cut_line_y(0) < 0):
                area0.fill_color = YELLOW_B
                area1.fill_color = BLUE_B
            return [area0, area1]

        area0_obj, area1_obj = update_area()
        area0_obj.add_updater(lambda x: x.become(update_area()[0]))
        area1_obj.add_updater(lambda x: x.become(update_area()[1]))
        self.add(area0_obj, area1_obj)

        # 展示阶跃函数 在阶跃函数图像中表示各点
        axes_temp = Axes(
            x_range=[-10, 10, 1],
            y_range=[-0.5, 1.5, 1],
            x_length=5,
            y_length=3,
            y_axis_config={
                "numbers_to_include": np.array([0, 1])
            },
            tips=False
        ).move_to((3, 0, 0))
        act_func_bef_obj = VGroup(
            axes_temp.plot(lambda x: 0, x_range=[axes_temp.x_range[0], 0], color=BLUE_D),
            axes_temp.plot(lambda x: 1, x_range=[0, axes_temp.x_range[1]], color=YELLOW_D)
        )
        def update_dot_bef(idx):
            """更新右侧图像上的点(阶跃函数)"""
            x = data_in[idx][0]*decision_w1.get_value() + data_in[idx][1]*decision_w2.get_value() - decision_b.get_value()
            dot = Square(
                side_length=0.1, color=BLUE_B if data_out[idx][0] == 0 else YELLOW_B,
                fill_opacity=True, fill_color=BLACK,
                stroke_width=2
            ).move_to(
                axes_temp.coords_to_point(x, 0 if x < 0 else 1)
            ).rotate(40 * DEGREES)
            return dot
        act_dot_bef_queue = [
            update_dot_bef(idx).add_updater(lambda x, i=idx: x.become(update_dot_bef(i))) for idx in range(len(data_in))
        ]
        def cal_err_bef():
            """计算误差(阶跃函数)"""
            err = 0
            for idx in range(len(data_in)):
                x = data_in[idx][0]*decision_w1.get_value() + data_in[idx][1]*decision_w2.get_value() - decision_b.get_value()
                err += abs((0 if x < 0 else 1) - data_out[idx][0])
            return err
        def update_err_obj_bef():
            """更新误差显示"""
            temp = VGroup(
                Text(
                    "误差:", color=RED_D, font_size=font_size - 4
                ).move_to(axes_temp.coords_to_point(0, -0.7)).shift(0.6*LEFT)
            )
            temp += Text(
                f"{cal_err_bef()}", font=font_type, color=RED_D, font_size=font_size - 4
            ).next_to(temp[0], direction=RIGHT, buff=0.2)
            return temp
        err_obj_bef = update_err_obj_bef().add_updater(lambda x: x.become(update_err_obj_bef()))
        self.add(axes_temp, act_func_bef_obj, *act_dot_bef_queue, err_obj_bef)

        # 展示垂线
        def cal_foot(idx: int) -> list:
            """计算垂足"""
            x0, y0 = data_in[idx]   # 点的坐标
            a, b, c = decision_w1.get_value(), decision_w2.get_value(), -decision_b.get_value() # 直线参数
            return [
                (b*b*x0 - a*b*y0 - a*c) / (a*a + b*b),
                (-a*b*x0 + a*a*y0 - b*c) / (a*a + b*b)
            ]
        def update_vertical_line(idx):
            """垂线更新"""
            return DashedLine(
                start=axes_frame.coords_to_point(*data_in[idx]),
                end=axes_frame.coords_to_point(*cal_foot(idx)),
                color=BLUE_D if data_out[idx][0] == 0 else YELLOW_D,
                stroke_width=2,
                dashed_ratio=0.5
            )
        vertical_line_queue = [
            update_vertical_line(idx).add_updater(
                lambda x, i=idx: x.become(update_vertical_line(i))
            ) for idx in range(len(data_in))]
        self.play(*[Write(obj) for obj in vertical_line_queue])
        self.wait()
        self.play(
            decision_w1.animate.set_value(3),
            decision_w2.animate.set_value(2),
            decision_b.animate.set_value(7)
        )
        self.wait()

        # 激活函数改变
        act_func_aft_obj = VGroup(
            axes_temp.plot(lambda x: 1/(1+np.exp(-x)), x_range=[axes_temp.x_range[0], 0], color=BLUE_D),
            axes_temp.plot(lambda x: 1/(1+np.exp(-x)), x_range=[0, axes_temp.x_range[1]], color=YELLOW_D)
        )
        fill_flag = [False for _ in range(len(data_in))]
        def update_dot_aft(idx):
            x = data_in[idx][0]*decision_w1.get_value() + data_in[idx][1]*decision_w2.get_value() - decision_b.get_value()
            dot = Square(
                side_length=0.1, color=BLUE_B if data_out[idx][0] == 0 else YELLOW_B,
                fill_opacity=True, fill_color=BLACK,
                stroke_width=2
            ).move_to(
                axes_temp.coords_to_point(x, 1/(1+np.exp(-x)))
            ).rotate(40*DEGREES)
            return dot
        act_dot_aft_queue = [
            update_dot_aft(idx).add_updater(lambda x, i=idx: x.become(update_dot_aft(i))) for idx in range(len(data_in))
        ]
        def cal_err_aft():
            """计算误差(sigmod函数)"""
            err = 0
            for idx in range(len(data_in)):
                x = data_in[idx][0]*decision_w1.get_value() + data_in[idx][1]*decision_w2.get_value() - decision_b.get_value()
                err += abs(1/(1+np.exp(-x)) - data_out[idx][0])
            return err
        def update_err_obj_aft():
            """更新误差显示"""
            temp = VGroup(
                Text(
                    "误差:", color=RED_D, font_size=font_size - 4
                ).move_to(axes_temp.coords_to_point(0, -0.7)).shift(0.6*LEFT)
            )
            temp += Text(
                f"{cal_err_aft():.4f}", font=font_type, color=RED_D, font_size=font_size - 4
            ).next_to(temp[0], direction=RIGHT, buff=0.2)
            return temp
        err_obj_aft = update_err_obj_aft().add_updater(lambda x: x.become(update_err_obj_aft()))
        self.play(
            Transform(act_func_bef_obj, act_func_aft_obj),
            *[ReplacementTransform(act_dot_bef_queue[idx], act_dot_aft_queue[idx]) for idx in range(len(data_in))],
            Unwrite(err_obj_bef[1]),
            Write(err_obj_aft[1]),
            run_time=2
        )
        self.remove(err_obj_bef)
        self.add(err_obj_aft)
        self.wait()

        # 调整直线
        self.play(
            decision_w1.animate.set_value(-2),
            decision_w2.animate.set_value(3),
            decision_b.animate.set_value(2)
        )
        self.wait()

        # 直线调整成竖线
        self.play(
            decision_w1.animate.set_value(1),
            decision_w2.animate.set_value(-1e-4),
            decision_b.animate.set_value(2)
        )
        self.wait()

        # 追加样本点 展示缓冲区边界
        data_in_extra = [[-1, 1.5], [0.5, 5], [1.5, 3.5],
                         [4, 1.5], [4, 4], [4.5, 2]]
        data_out_extra = [[0], [0], [0],
                          [1], [1], [1]]
        fill_flag.extend([False for _ in range(len(data_in_extra))])
        data_in.extend(data_in_extra)
        data_out.extend(data_out_extra)
        svg_queue_extra = [load_svg(data_out_extra[i][0], data_in_extra[i], fill=False)
                           for i in range(len(data_in_extra))]
        vertical_line_queue_extra = [
            update_vertical_line(idx).add_updater(
                lambda x, i=idx: x.become(update_vertical_line(i))
            ) for idx in range(len(data_in)-len(data_in_extra), len(data_in))]
        act_dot_aft_queue_extra = [
            update_dot_aft(idx).add_updater(lambda x, i=idx: x.become(update_dot_aft(i)))
            for idx in range(len(data_in)-len(data_in_extra), len(data_in))
        ]
        self.play(
            *[FadeIn(obj) for obj in svg_queue_extra],
            *[Write(obj) for obj in vertical_line_queue_extra],
            *[Write(obj) for obj in act_dot_aft_queue_extra],
            run_time=1
        )
        svg_queue.extend(svg_queue_extra)
        vertical_line_queue.extend(vertical_line_queue_extra)
        act_dot_aft_queue.extend(act_dot_aft_queue_extra)
        def cal_dist() -> list[float]:
            """计算各点距离"""
            dist = []
            for idx in range(len(data_in)):
                x = data_in[idx][0]*decision_w1.get_value() + data_in[idx][1]*decision_w2.get_value() - decision_b.get_value()
                dist.append(x)
            return dist
        def update_buff_line() -> VGroup:
            """更新缓冲区边界"""
            res = VGroup()
            dist = np.array(cal_dist())
            dot_idx = [np.where(dist == np.max(dist[dist < 0]))[0][0],     # 圆形支持向量索引
                       np.where(dist == np.min(dist[dist > 0]))[0][0]]     # 三角形支持向量索引
            k = -decision_w1.get_value()/decision_w2.get_value()
            x_range = axes_frame.x_range
            y_range = axes_frame.y_range
            for typ in [0, 1]:
                dot = data_in[dot_idx[typ]]
                def line_y(x):
                    """边界的方程"""
                    return k*(x-dot[0]) + dot[1]
                def line_x(y):
                    """边界的方程"""
                    return (y-dot[1])/k + dot[0]
                # 以下代码防止直线超过坐标轴范围
                if y_range[0] <= line_y(x_range[0]) <= y_range[1]:
                    start_point = [x_range[0], line_y(x_range[0])]  # 若左端点在y区间内，则以x为依据给出左端点
                elif line_y(x_range[0]) > y_range[1]:
                    start_point = [line_x(y_range[1]), y_range[1]]  # 若左端点在y区间上侧，则以y上边界给出左端点
                else:
                    start_point = [line_x(y_range[0]), y_range[0]]  # 若左端点在y区间下侧，则以y下边界给出左端点
                if y_range[0] <= cut_line_y(x_range[1]) <= y_range[1]:
                    end_point = [x_range[1], line_y(x_range[1])]
                elif line_y(x_range[1]) > y_range[1]:
                    end_point = [line_x(y_range[1]), y_range[1]]
                else:
                    end_point = [line_x(y_range[0]), y_range[0]]
                res += Line(
                    start=axes_frame.coords_to_point(*start_point),
                    end=axes_frame.coords_to_point(*end_point),
                    color=BLUE_D if typ == 0 else YELLOW_D,
                    stroke_width=3
                )
            return res
        buff_line_group = update_buff_line().add_updater(lambda x: x.become(update_buff_line()))
        self.play(
            Write(buff_line_group),
            decision_w1.animate.set_value(2),
            decision_w2.animate.set_value(-1),
            decision_b.animate.set_value(2)
        )
        self.wait()

        # 给支持向量加标记
        dist = np.array(cal_dist())
        dot_idx = [np.where(dist == np.max(dist[dist < 0]))[0][0],  # 圆形支持向量索引
                   np.where(dist == np.min(dist[dist > 0]))[0][0]]  # 三角形支持向量索引
        svg_mark = [load_svg(data_out[idx][0], data_in[idx], True) for idx in dot_idx]
        def update_dot_mark(idx):
            x = data_in[idx][0]*decision_w1.get_value() + data_in[idx][1]*decision_w2.get_value() - decision_b.get_value()
            dot = Square(
                side_length=0.1, color=BLUE_B if data_out[idx][0] == 0 else YELLOW_B,
                fill_opacity=True, fill_color=BLUE_B if data_out[idx][0] == 0 else YELLOW_B,    # 带填色
                stroke_width=2
            ).move_to(
                axes_temp.coords_to_point(x, 1/(1+np.exp(-x)))
            ).rotate(40*DEGREES)
            return dot
        act_dot_mark_queue = [
            update_dot_mark(idx).add_updater(lambda x, i=idx: x.become(update_dot_mark(i))) for idx in dot_idx
        ]
        self.play(
            *[Write(obj) for obj in svg_mark],
            *[Write(obj) for obj in act_dot_mark_queue]
        )
        self.wait()
