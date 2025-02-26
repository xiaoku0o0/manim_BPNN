from manim import *


class MC1(Scene):
    def construct(self):
        def get_net(
                node_number_ls: list[int],      # 各层节点数量
                active_func_ls: list[None | str]       # 各层激活函数名称
        ) -> VGroup:
            """产生NN的manim VGroup对象"""
            layer_number = len(node_number_ls)
            if layer_number != len(active_func_ls):
                raise ValueError(f"节点层数与激活函数名称数不匹配（输入无激活函数使用None）")
            axes = Axes(
                x_range=[0, 10, 1],
                y_range=[-1, 1, 1]
            )
            axes_x_range = axes.x_range
            axes_y_range = axes.y_range
            dx = (axes_x_range[1]-axes_x_range[0])/(layer_number-1)     # 横向间距
            def get_node(layer_idx) -> VGroup:
                """产生节点"""
                node_ls = []
                for node_idx in range(node_number_ls[layer_idx]):
                    dy = 0 if node_number_ls[layer_idx]==1 else (axes_y_range[1]-axes_y_range[0])/(node_number_ls[layer_idx]-1)
                    dy = min(dy, 0.6)   # 设置最大纵向间距
                    y_position = dy * (node_number_ls[layer_idx] - 1) / 2 - node_idx * dy
                    node_ls.append(
                        Circle(radius=0.2, color=BLUE_D, fill_opacity=True, fill_color=BLACK, stroke_width=1).move_to(
                            axes.coords_to_point(axes_x_range[0]+layer_idx*dx, y_position)
                        )
                    )
                node_temp = VGroup(*node_ls)
                return node_temp
            node_ls = [get_node(layer_idx) for layer_idx in range(layer_number)]
            label_ls = [
                Text(
                    active_func_ls[layer_idx], font_size=20, font="Times New Roman", color=BLUE_D
                ).move_to(axes.coords_to_point(axes_x_range[0]+layer_idx*dx, axes_y_range[0])).shift(DOWN*0.3)
            for layer_idx in range(layer_number) if active_func_ls[layer_idx] is not None]
            # 产生线条
            line_ls = []
            for next_layer_idx in range(1, layer_number):
                line_temp = VGroup()
                for bef_node_idx in range(node_number_ls[next_layer_idx-1]):
                    for aft_node_idx in range(node_number_ls[next_layer_idx]):
                        line_temp += Line(start=node_ls[next_layer_idx-1][bef_node_idx].point_at_angle(0),
                                          end=node_ls[next_layer_idx][aft_node_idx].point_at_angle(180*DEGREES),
                                          color=BLUE_B,
                                          stroke_width=0.7)
                line_ls.append(line_temp)
            # 线条与节点交错
            res = VGroup(node_ls[0])
            for layer_idx in range(1, layer_number):
                res += line_ls[layer_idx-1]
                res += node_ls[layer_idx]
            res += VGroup(*label_ls)
            return res
        nn_1 = get_net([2, 3, 1], [None, "sigmod", "sigmoid"])
        self.add(nn_1)
        nn_2 = get_net([2, 3, 1], [None, "tanh", "sigmoid"])
        self.play(ReplacementTransform(nn_1, nn_2))
        self.wait()
        nn_3 = get_net([2, 3, 8, 1], [None, "tanh", "sigmod", "sigmoid"])
        self.play(ReplacementTransform(nn_2, nn_3))
        nn_4 = get_net([2, 3, 8, 7, 1], [None, "tanh", "sigmoid", "ReLu", "sigmoid"])
        self.play(ReplacementTransform(nn_3, nn_4))
        nn_5 = get_net([2, 3, 8, 7, 9, 1], [None, "tanh", "sigmoid", "ReLu", "tanh", "sigmoid"])
        self.play(ReplacementTransform(nn_4, nn_5))
        nn_6 = get_net([2, 3, 8, 7, 9, 6, 1], [None, "tanh", "sigmoid", "ReLu", "tanh", "sigmoid", "sigmoid"])
        self.play(ReplacementTransform(nn_5, nn_6))
        self.wait()
