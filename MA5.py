from manim import *
import numpy as np
from rely_BPNN import *     # 自定义库，实现SVM


class MA5(Scene):
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


        # 创建坐标轴
        axes = Axes(x_range=[-1, 5, 1],
                    y_range=[-1, 5, 1],
                    x_length=5,
                    y_length=5,
                    tips=False,
                    ).move_to(np.array([3,0,0]))
        axes_label = axes.get_axis_labels(x_label="x_1",y_label="x_2")
        for obj in axes_label:
            obj.font_size = 30  # 调整坐标轴标签的字号
        self.add(axes, axes_label)

        # 载入图形
        data_in = [[1, 1], [1, 3], [3, 1], [3, 3]]
        data_out = [[0], [0], [1], [1]]
        svg_queue = [load_svg(svg_type=data_out[i][0], xy=data_in[i], fill=False) for i in range(len(data_in))]
        self.add(*svg_queue)

        # 展示各数据坐标
        data_coordinate_queue = [
            Tex(
                "$($", f"${data_in[i][0]}$", "$,$", f"${data_in[i][1]}$", "$)$", font_size=font_size
            ).next_to(svg_queue[i], direction=RIGHT, buff=0.1)
            for i in range(len(data_in))]
        self.add(*data_coordinate_queue)

        # 绘制直线
        """w1*x1 + w2*x2 - b = 0"""
        decision_w1 = ValueTracker(2)
        decision_w2 = ValueTracker(-1)
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
        self.add(decision_line_obj)

        # 展示直线方程
        line_func = Tex(
            "$w_{1}$", "$x_{1}$", "$+w_{2}$", "$x_{2}$", "$-b$", "$=0$", font_size=font_size, color=ORANGE
        ).next_to(decision_line_obj.get_start(), direction=RIGHT, buff=0.1)
        self.add(line_func)
        self.wait()

        # 输入代入直线方程
        temp1 = Tex(
            "$w_{1}$", r"$\cdot$", "$x_{1}$", "$+w_{2}$", r"$\cdot$", "$x_{2}$", "$-b$",
            font_size=30, color=ORANGE
        ).move_to((-2.5, 3, 0))
        self.play(TransformMatchingTex(
            VGroup(line_func.copy(), data_coordinate_queue[1].copy()), temp1,
            key_map={
                f"${data_in[1][0]}$": "$x_{1}$",
                f"${data_in[1][1]}$": "$x_{2}$"
            }
        ))
        self.wait()

        # 作用激活函数
        temp2 = VGroup(
            Tex(r"$z=\text{sigmoid}($", font_size=30, color=MAROON_B).next_to(temp1, direction=LEFT, buff=0.1),
            Tex(r"$)$", font_size=30, color=MAROON_B).next_to(temp1, direction=RIGHT, buff=0.1)
        )
        self.play(Write(temp2))
        self.wait()

        # 看成两个向量点乘
        temp3 = Tex(r"$($", "$w_1$", "$,$", "$w_2$", "$)$", r"$\cdot$", "$($", "$x_1$", "$,$", "$x_2$", "$)$", "$-b$",
                    font_size=30, color=ORANGE).move_to(temp1.get_center())
        self.play(
            ReplacementTransform(temp1, temp3),
            temp2[0].animate.next_to(temp3, direction=LEFT, buff=0.1),
            temp2[1].animate.next_to(temp3, direction=RIGHT, buff=0.1)
        )
        self.wait()

        # 写成矩阵形式
        temp4 = Tex(
            r"""$\begin{bmatrix}
              w_1&w_2
            \end{bmatrix}
            \begin{bmatrix}
             x_1\\
             x_2
            \end{bmatrix}$""", "$-b$",
            font_size=30, color=ORANGE
        ).move_to(temp3.get_center())
        self.play(
            TransformMatchingTex(temp3, temp4),
            temp2[0].animate.next_to(temp4, direction=LEFT, buff=0.1),
            temp2[1].animate.next_to(temp4, direction=RIGHT, buff=0.1)
        )
        self.wait()

        # 回顾基本形式
        temp1 = Tex(
            "$w_{1}$", r"$\cdot$", "$x_{1}$", "$+w_{2}$", r"$\cdot$", "$x_{2}$", "$-b$",
            font_size=30, color=ORANGE
        ).move_to(temp3.get_center())
        self.play(
            FadeIn(temp1),
            temp1.animate.shift(0.8*DOWN)
        )
        self.wait()

        # 载入NN架构
        bpnn = BPNN([2,1])
        # 手动设置权重和阈值
        bpnn.node_ls[-1][0].w_ls = [decision_w1.get_value(), decision_w2.get_value()]
        bpnn.node_ls[-1][0].b = [decision_b.get_value()]
        bpnn.node_ls[-1][0].manim_update()  # 载入进manim动画
        manim_obj_bpnn_axes = Axes(
            x_range=[0, bpnn.layer_numbers-1, 1],  # 一层一个x
            y_range=[-(max(bpnn.node_numbers_ls)-1)/2, (max(bpnn.node_numbers_ls)-1)/2, 1],  # 每个节点的纵向距离最小为1
            x_length=4,
            y_length=3,
            tips=False
        ).move_to(np.array([-3, 0, 0]))  # 这个bpnn_axes并不展示出来，只是建立一个参考系
        # self.add(manim_obj_bpnn_axes)   # TODO 测试用，记得删
        # 放置节点
        # ===============这里抢救不过来了，后期在Pr里加个交叉淡化
        for layer_index in range(bpnn.layer_numbers):
            for node_index in range(bpnn.node_numbers_ls[layer_index]):
                node = bpnn.node_ls[layer_index][node_index]
                dy = 1  # 节点纵向间距
                y_position = dy * (bpnn.node_numbers_ls[layer_index] - 1) / 2 - node_index * dy
                node.manim_obj_node.move_to(manim_obj_bpnn_axes.coords_to_point(layer_index, y_position))
                node.manim_obj_b.move_to(manim_obj_bpnn_axes.coords_to_point(layer_index, y_position)).shift(
                    shift_b * RIGHT)
                node.manim_obj_value.move_to(node.manim_obj_node.get_center()).shift(shift_value * DOWN)
        # 连接节点
        line_array = [[[None for node_now_idx in range(bpnn.node_numbers_ls[layer_idx])] for node_bef_idx in
                       ([] if layer_idx == 0 else range(bpnn.node_numbers_ls[layer_idx - 1]))] for layer_idx in
                      range(bpnn.layer_numbers)]
        """
            line_array[当前层索引0,1,2...][前一层节点索引0,1,2...][当前层节点索引0,1,2...]
            line_array[layer_idx][node_bef_idx][node_now_idx]
            line_array保存连接了前一层节点与当前层节点的Line对象
        """
        for layer_idx in range(1, bpnn.layer_numbers):
            for node_now_idx in range(bpnn.node_numbers_ls[layer_idx]):
                for node_bef_idx in range(bpnn.node_numbers_ls[layer_idx - 1]):
                    line_array[layer_idx][node_bef_idx][node_now_idx] = Line(
                        start=bpnn.node_ls[layer_idx - 1][node_bef_idx].manim_obj_node.get_center(),
                        end=bpnn.node_ls[layer_idx][node_now_idx].manim_obj_node.get_center()
                    )
                    line = line_array[layer_idx][node_bef_idx][node_now_idx]
                    self.add(line)  # 载入节点间连线
                    # 调整权重位置
                    node = bpnn.node_ls[layer_idx][node_now_idx]
                    node.manim_obj_w[node_bef_idx].move_to(
                        line.get_end() + np.array([-shift_w, -shift_w * np.tan(line.get_angle()), 0])).shift(
                        0.2 * UP).rotate(line.get_angle())
                    self.add(node.manim_obj_w[node_bef_idx])  # 载入权重
        # 载入节点
        # 节点载入在线条之后，保证不被线条遮挡
        for layer_index in range(bpnn.layer_numbers):
            for node_index in range(bpnn.node_numbers_ls[layer_index]):
                node = bpnn.node_ls[layer_index][node_index]
                self.add(node.manim_obj_node, node.manim_obj_value)  # 载入节点、数值
                if layer_index != 0:  # 输入层无阈值
                    self.add(node.manim_obj_b)  # 载入b
        # 输入层添加节点标签
        for node_idx in range(bpnn.node_numbers_ls[0]):
            bpnn.node_ls[0][node_idx].manim_obj_label = MathTex("x_{" + f"{node_idx + 1}" + "}").move_to(
                bpnn.node_ls[0][node_idx].manim_obj_node.get_center())
            bpnn.node_ls[0][node_idx].manim_obj_label.font_size = 20
            bpnn.node_ls[0][node_idx].manim_obj_label.color = BLUE
            self.add(bpnn.node_ls[0][node_idx].manim_obj_label)
        # 输出层加个z
        bpnn.node_ls[1][0].manim_obj_label = MathTex("z").move_to(bpnn.node_ls[1][0].manim_obj_node.get_center())
        bpnn.node_ls[1][0].manim_obj_label.font_size = 20
        bpnn.node_ls[1][0].manim_obj_label.color = GREEN
        self.add(bpnn.node_ls[1][0].manim_obj_label)

        def w_b_update():
            """更新权重和阈值"""
            for layer_idx in range(bpnn.layer_numbers):
                for node_now_idx in range(bpnn.node_numbers_ls[layer_idx]):
                    node = bpnn.node_ls[layer_idx][node_now_idx]
                    node.manim_update()  # 数值更新
                    # 位置重载
                    node.manim_obj_value.move_to(node.manim_obj_node.get_center()).shift(shift_value * DOWN)
                    if layer_idx != 0:  # 输入层不更新权重和阈值，仅更新value
                        node.manim_obj_b.move_to(node.manim_obj_node.get_center()).shift(shift_b * RIGHT)
                        for node_bef_idx in range(bpnn.node_numbers_ls[layer_idx - 1]):
                            line = line_array[layer_idx][node_bef_idx][node_now_idx]
                            node.manim_obj_w[node_bef_idx].move_to(
                                line.get_end() + np.array([-shift_w, -shift_w * np.tan(line.get_angle()), 0])).shift(
                                0.2 * UP).rotate(line.get_angle())
        self.wait()
        for idx in range(len(data_in)):
            svg_fill = load_svg(data_out[idx][0], data_in[idx], True)
            self.add(svg_fill)
            bpnn.update(data_in[idx])
            w_b_update()
            self.wait(0.5)
            self.remove(svg_fill)
        self.wait()
