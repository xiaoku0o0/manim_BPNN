from manim import *
import numpy as np


def activation_function(x):
    """BPNN激活函数"""
    return 1/(1+np.exp(-x))


# manim参数
font_type = "Times New Roman"
font_size = 20
shift_b = 0.5       # 权重b标签向右偏移量
shift_w = 0.6       # 阈值w标签横向偏移量
shift_value = 0.4   # 数值value标签向下偏移量


class Node():
    """NN节点类"""
    def __init__(self, w_ls: list, b: int):
        """
        w_ls 前方权重列表
        b 后方阈值
        """
        self.w_ls = w_ls
        self.b = [b]    # 把b放进列表里，便于引用
        self.value = None   # 节点数据，减去阈值，作用激活函数后
        # manim动画方面
        self.manim_obj_node = Circle(radius=0.2, color=BLUE_D, fill_opacity=True, fill_color=BLACK, stroke_width=1)  # 节点的manim对象
        self.manim_obj_w = [Text(f"{w:.2f}", color=PURPLE_B, font=font_type, font_size=font_size) for w in self.w_ls]
        self.manim_obj_b = Text(f"{self.b[0]:.2f}", color=ORANGE, font=font_type, font_size=font_size)
        self.manim_obj_label = None     # 节点上的标签
        self.manim_obj_value = Text("")
        def update_w(idx):
            """权重的text更新函数"""
            temp = Text(f"{self.w_ls[idx]:.2f}", color=PURPLE_B, font=font_type, font_size=font_size)
            return temp
        def update_b():
            """阈值的text更新函数"""
            temp = Text(f"{self.b[0]:.2f}", color=ORANGE, font=font_type, font_size=font_size)
            return temp
        # 建立更新，调整字号
        for w_idx in range(len(self.w_ls)):
            self.manim_obj_w[w_idx].add_updater(lambda x: x.become(update_w(w_idx)))
        self.manim_obj_b.add_updater(lambda x: x.become(update_b()))
    def update(self, value):
        """此处value需要在外部减去阈值，作用激活函数"""
        self.value = value
    def manim_update(self):
        """manim动画更新权重和阈值"""
        def update_w(idx):
            """权重的text更新函数"""
            temp = Text(f"{self.w_ls[idx]:.2f}", color=PURPLE_B, font=font_type, font_size=font_size)
            return temp
        def update_b():
            """阈值的text更新函数"""
            temp = Text(f"{self.b[0]:.2f}", color=ORANGE, font=font_type, font_size=font_size)
            return temp
        def update_value():
            """数值的text更新函数"""
            if self.value is None:
                temp = Text("")
            else:
                temp = Text(f"{self.value:.3f}", color=GREEN_B, font=font_type, font_size=font_size)
            return temp
        # 建立更新，调整字号
        for w_idx in range(len(self.w_ls)):
            self.manim_obj_w[w_idx].become(update_w(w_idx))
        self.manim_obj_b.become(update_b())
        self.manim_obj_value.become(update_value())


class BPNN():
    def __init__(self,node_numbers_ls: list):
        """node_number_ls为节点个数列表，第0项为输入层节点个数，第-1项为输出层节点个数"""
        self.node_numbers_ls = node_numbers_ls
        self.layer_numbers = len(self.node_numbers_ls)    # 层数（含输入层与输出层）
        if self.layer_numbers < 2:
            raise ValueError(f"BPNN应至少具备两层（输入层与输出层）")
        if node_numbers_ls[0] < 1 or node_numbers_ls[-1] < 1:
            raise ValueError(f"输入层与输出层应至少具备一个节点")
        # 初始化节点
        node_in_ls = [Node([0],0) for _ in range(self.node_numbers_ls[0])]  # 输入层前面没有结构，权重无意义；输入层无阈值
        self.node_ls = [node_in_ls] # 输入层比较特殊，单独处理
        for layer_idx in range(1, self.layer_numbers):
            self.node_ls.append(
                [Node([1 / self.node_numbers_ls[layer_idx - 1]] * self.node_numbers_ls[layer_idx - 1], 0) for _ in range(self.node_numbers_ls[layer_idx])]
            )   # 初始化权重为等权，初始化阈值为0


    def update(self,input_ls: list) -> list:
        """数据正向传递"""
        if len(input_ls) != self.node_numbers_ls[0]:
            raise ValueError(f"数据输入维度{len(input_ls)}与输入层节点个数{self.node_numbers_ls[0]}不匹配")
        # 输入层更新
        for i in range(len(input_ls)):
            self.node_ls[0][i].update(input_ls[i])
        # 后续数据传递
        for layer_idx in range(1, self.layer_numbers):
            for node_idx in range(self.node_numbers_ls[layer_idx]):
                sum_temp = 0
                for bef_node_idx in range(self.node_numbers_ls[layer_idx - 1]):
                    sum_temp += self.node_ls[layer_idx-1][bef_node_idx].value * \
                                self.node_ls[layer_idx][node_idx].w_ls[bef_node_idx]
                self.node_ls[layer_idx][node_idx].update(activation_function(sum_temp-self.node_ls[layer_idx][node_idx].b[0]))
        # 返回输出
        return [self.node_ls[-1][i].value for i in range(self.node_numbers_ls[-1])]
    def update_by_data(self,input_data: list) -> list:
        """训练集数据正向传递"""
        out_data = []
        for input_ls in input_data:
            out_data.append(self.update(input_ls))
        return out_data
    def back(self, input_data: list, except_out_data: list, alpha: float = 0.5, e: float = 1e-2) -> float:
        """误差反向传播"""
        """
        input_data 输入数据集列表
        except_out_data 预期输出列表
        alpha 学习率
        e 容差
        """
        if len(input_data[0]) != self.node_numbers_ls[0]:
            raise ValueError(f"数据输入维度{len(input_data[0])}与输入层节点个数{self.node_numbers_ls[0]}不匹配")
        if len(except_out_data[0]) != self.node_numbers_ls[-1]:
            raise ValueError(f"预期输出维度{len(except_out_data)}与输出层节点个数{self.node_numbers_ls[-1]}不匹配")
        def get_args_ls() -> list:
            """获取参数列表"""
            args_ls = []
            for layer_idx in range(1, self.layer_numbers):
                for node_idx in range(self.node_numbers_ls[layer_idx]):
                    args_ls.extend(self.node_ls[layer_idx][node_idx].w_ls)
                    args_ls.append(self.node_ls[layer_idx][node_idx].b[0])
            return args_ls
        def update_args(args_ls: list) -> None:
            """依据参数列表，更新节点参数"""
            layer_idx = 1
            node_idx = 0
            wight_idx = 0
            for args_idx in range(len(args_ls)):
                if wight_idx < len(self.node_ls[layer_idx][node_idx].w_ls):
                    self.node_ls[layer_idx][node_idx].w_ls[wight_idx] = args_ls[args_idx]
                    wight_idx += 1
                else:   # 权重已完成赋值，更新阈值后转向下一个节点
                    self.node_ls[layer_idx][node_idx].b = [args_ls[args_idx]]
                    wight_idx = 0
                    node_idx += 1
                    if node_idx >= self.node_numbers_ls[layer_idx]: # 该层节点已完成赋值，转向下一层
                        layer_idx += 1
                        node_idx = 0
        def cal_error(now_output_data: list) -> float:
            """计算误差（损失函数）"""
            err = 0
            for data_idx in range(len(except_out_data)):
                err += np.linalg.norm(np.array(now_output_data[data_idx])-np.array(except_out_data[data_idx]))
            return err/len(input_data)
        args_ls = get_args_ls()
        # 梯度下降
        def cal_grad(args_ls,h = 1e-6) -> list:
            """计算数值梯度"""
            """
            args_ls 计算位置
            h 数值步长
            """
            grad = []
            # 计算初始误差
            update_args(args_ls)
            error_init = cal_error(self.update_by_data(input_data))
            for args_idx in range(len(args_ls)):
                args_temp = args_ls[:]
                args_temp[args_idx] += h
                update_args(args_temp)
                grad.append((cal_error(self.update_by_data(input_data)) - error_init)/h)
            return grad
        while np.linalg.norm(np.array(self.update_by_data(input_data))-np.array(except_out_data))/len(input_data) > e:
            grad_now = cal_grad(args_ls)
            args_ls = list(np.array(args_ls) - alpha * np.array(grad_now))  # 向负梯度方向前进
            update_args(args_ls)    # 更新以便于在下轮循环开启前计算输出
            # print(args_ls,bpnn.update(input_ls))
            # print("\t",grad_now,np.linalg.norm(grad_now))
        update_args(args_ls)    # 迭代完成，更新参数
        return cal_error(self.update_by_data(input_data))     # 返回误差

    def train(self, input_data: list, except_out_data: list, turn: int, alpha: float = 0.5, e_one: float = 1e-2,
              e_all: float = 0.3):
        """训练神经网络"""
        """
        input_data 二维列表，每个列表项是输入数据列表
        except_out_data 二维列表，每个列表项是预期输出列表
        turn 最大训练轮次
        alpha 学习率
        e_one 单次容差
        e_all 总训练集容差
        """
        data_nums = len(input_data)
        if data_nums != len(except_out_data):
            raise ValueError(f"输入数据量{data_nums}与预期输出数据量{len(except_out_data)}不匹配")
        if turn < 1:
            turn = 1
        err = 0
        for times in range(turn):
            err = self.back(input_data, except_out_data, alpha, e_one)
            print(f"第{times + 1}轮，误差{err}")
        return err


class MA6(Scene):
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
                              height=0.2).move_to(axes.coords_to_point(*xy))


        # 创建坐标轴
        axes = Axes(x_range=[-1, 5, 1],
                    y_range=[-1, 5, 1],
                    x_length=4,
                    y_length=4,
                    tips=False,
                    ).move_to(np.array([-3, 1, 0]))
        axes_label = axes.get_axis_labels(x_label="x_1",y_label="x_2")
        for obj in axes_label:
            obj.font_size = 30  # 调整坐标轴标签的字号
        self.add(axes, axes_label)


        # 载入图形
        data_in = [[1, 1], [1, 3], [3, 1], [3, 3], [0, 1.5], [0.5, 4], [1.5, 3.5], [4, 1.5], [4, 4], [4.5, 2]]
        data_out = [[0], [0], [1], [1], [0], [0], [0], [1], [1], [1]]
        svg_queue = [load_svg(svg_type=data_out[i][0], xy=data_in[i], fill=False) for i in range(len(data_in))]
        self.add(*svg_queue)


        # 载入NN架构
        bpnn = BPNN([2,1])
        manim_obj_bpnn_axes = Axes(
            x_range=[0, bpnn.layer_numbers-1, 1],  # 一层一个x
            y_range=[-(max(bpnn.node_numbers_ls)-1)/2, (max(bpnn.node_numbers_ls)-1)/2, 1],  # 每个节点的纵向距离最小为1
            x_length=4,
            y_length=3,
            tips=False
        ).move_to(np.array([3, 2, 0]))  # 这个bpnn_axes并不展示出来，只是建立一个参考系
        # self.add(manim_obj_bpnn_axes)   # TODO 测试用，记得删
        # 放置节点
        for layer_index in range(bpnn.layer_numbers):
            for node_index in range(bpnn.node_numbers_ls[layer_index]):
                node = bpnn.node_ls[layer_index][node_index]
                dy = 1   # 节点纵向间距
                y_position = dy * (bpnn.node_numbers_ls[layer_index]-1)/2-node_index*dy
                node.manim_obj_node.move_to(manim_obj_bpnn_axes.coords_to_point(layer_index,y_position))
                node.manim_obj_b.move_to(manim_obj_bpnn_axes.coords_to_point(layer_index,y_position)).shift(shift_b*RIGHT)
                node.manim_obj_value.move_to(node.manim_obj_node.get_center()).shift(shift_value*DOWN)
        # 连接节点
        line_array = [[[None for node_now_idx in range(bpnn.node_numbers_ls[layer_idx])] for node_bef_idx in ([] if layer_idx==0 else range(bpnn.node_numbers_ls[layer_idx-1]))] for layer_idx in range(bpnn.layer_numbers)]
        """
            line_array[当前层索引0,1,2...][前一层节点索引0,1,2...][当前层节点索引0,1,2...]
            line_array[layer_idx][node_bef_idx][node_now_idx]
            line_array保存连接了前一层节点与当前层节点的Line对象
        """
        for layer_idx in range(1,bpnn.layer_numbers):
            for node_now_idx in range(bpnn.node_numbers_ls[layer_idx]):
                for node_bef_idx in range(bpnn.node_numbers_ls[layer_idx-1]):
                    line_array[layer_idx][node_bef_idx][node_now_idx] = Line(
                        start=bpnn.node_ls[layer_idx-1][node_bef_idx].manim_obj_node.get_center(),
                        end=bpnn.node_ls[layer_idx][node_now_idx].manim_obj_node.get_center()
                    )
                    line = line_array[layer_idx][node_bef_idx][node_now_idx]
                    self.add(line)  # 载入节点间连线
                    # 调整权重位置
                    node = bpnn.node_ls[layer_idx][node_now_idx]
                    node.manim_obj_w[node_bef_idx].move_to(line.get_end()+np.array([-shift_w,-shift_w*np.tan(line.get_angle()),0])).shift(0.2*UP).rotate(line.get_angle())
                    self.add(node.manim_obj_w[node_bef_idx])    # 载入权重
        # 载入节点
        # 节点载入在线条之后，保证不被线条遮挡
        for layer_index in range(bpnn.layer_numbers):
            for node_index in range(bpnn.node_numbers_ls[layer_index]):
                node = bpnn.node_ls[layer_index][node_index]
                self.add(node.manim_obj_node,node.manim_obj_value)  # 载入节点、数值
                if layer_index != 0:  # 输入层无阈值
                    self.add(node.manim_obj_b)  # 载入b
        # 输入层添加节点标签
        for node_idx in range(bpnn.node_numbers_ls[0]):
            bpnn.node_ls[0][node_idx].manim_obj_label = MathTex("x_{" + f"{node_idx + 1}" + "}").move_to(bpnn.node_ls[0][node_idx].manim_obj_node.get_center())
            bpnn.node_ls[0][node_idx].manim_obj_label.font_size = 20
            bpnn.node_ls[0][node_idx].manim_obj_label.color = BLUE
            self.add(bpnn.node_ls[0][node_idx].manim_obj_label)
        def w_b_update():
            """更新权重和阈值"""
            for layer_idx in range(bpnn.layer_numbers):
                for node_now_idx in range(bpnn.node_numbers_ls[layer_idx]):
                    node = bpnn.node_ls[layer_idx][node_now_idx]
                    node.manim_update()    # 数值更新
                    # 位置重载
                    node.manim_obj_value.move_to(node.manim_obj_node.get_center()).shift(shift_value * DOWN)
                    if layer_idx != 0:  # 输入层不更新权重和阈值，仅更新value
                        node.manim_obj_b.move_to(node.manim_obj_node.get_center()).shift(shift_b * RIGHT)
                        for node_bef_idx in range(bpnn.node_numbers_ls[layer_idx-1]):
                            line = line_array[layer_idx][node_bef_idx][node_now_idx]
                            node.manim_obj_w[node_bef_idx].move_to(line.get_end()+np.array([-shift_w,-shift_w*np.tan(line.get_angle()),0])).shift(0.2*UP).rotate(line.get_angle())

        # 在左侧图形区载入分隔直线
        def update_cut_line():
            """分割线更新函数"""
            def cut_line_y(x):
                """分隔直线方程，输入x(x_1)，输出y(x_2)"""
                b = bpnn.node_ls[-1][0].b[0]
                w = bpnn.node_ls[-1][0].w_ls
                return (b-w[0]*x)/w[1]
            def cut_line_x(y):
                """分隔直线方程，输入y(x_2)，输出x(x_1)"""
                b = bpnn.node_ls[-1][0].b[0]
                w = bpnn.node_ls[-1][0].w_ls
                return (b-w[1]*y)/w[0]
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
            cut_line.stroke_color = YELLOW_A
            return cut_line
        cut_line = update_cut_line()
        self.add(cut_line)
        # 载入分隔区域
        def update_area():
            def cut_line_y(x):
                """分隔直线方程，输入x(x_1)，输出y(x_2)"""
                b = bpnn.node_ls[-1][0].b[0]
                w = bpnn.node_ls[-1][0].w_ls
                return (b-w[0]*x)/w[1]
            def cut_line_x(y):
                """分隔直线方程，输入y(x_2)，输出x(x_1)"""
                b = bpnn.node_ls[-1][0].b[0]
                w = bpnn.node_ls[-1][0].w_ls
                return (b-w[1]*y)/w[0]
            x_range = axes.x_range[:-1]
            y_range = axes.y_range[:-1]
            sum_area = Polygon(axes.coords_to_point(x_range[1], y_range[1]),
                               axes.coords_to_point(x_range[0], y_range[1]),
                               axes.coords_to_point(x_range[0], y_range[0]),
                               axes.coords_to_point(x_range[1], y_range[0])
                               )    # 坐标轴所有区域（全集）
            def get_area0(x,y):
                # x, y = point[:2]
                b = bpnn.node_ls[-1][0].b[0]
                w = bpnn.node_ls[-1][0].w_ls
                return w[0]*x + w[1]*y - b
            cut_func = axes.plot(cut_line_y,x_range=x_range)
            area0 = axes.get_area(cut_func, x_range=x_range,
                                  bounded_graph=axes.plot(lambda x: y_range[0], x_range=x_range))
            area0 = Intersection(sum_area, area0, fill_color=BLUE_B, fill_opacity=0.5, stroke_opacity=False)
            area1 = Difference(sum_area, area0, fill_color=YELLOW_B, fill_opacity=0.5, stroke_opacity=False)
            # 判断两区域填色是否正确
            origin_type = 1 if get_area0(0,0)>0 else 0
            if (origin_type==1 and cut_line_y(0)>0) or (origin_type==0 and cut_line_y(0)<0):
                area0.fill_color = YELLOW_B
                area1.fill_color = BLUE_B
            return [area0, area1]
        area1, area2 = update_area()
        self.add(area1, area2)

        # 添加激活函数上各点
        axes_act = Axes(
            x_range=[-10, 10, 1],
            y_range=[-0.1, 1.1, 1],
            x_length=5,
            y_length=3,
            y_axis_config={
                "numbers_to_include": np.array([0, 1])
            },
            tips=False
        ).move_to((3, -2, 0))
        act_func_obj = VGroup(
            axes_act.plot(lambda x: 1 / (1 + np.exp(-x)), x_range=[axes_act.x_range[0], 0], color=BLUE_D),
            axes_act.plot(lambda x: 1 / (1 + np.exp(-x)), x_range=[0, axes_act.x_range[1]], color=YELLOW_D)
        )
        def update_dot(idx):
            x = data_in[idx][0]*bpnn.node_ls[1][0].w_ls[0] + data_in[idx][1]*bpnn.node_ls[1][0].w_ls[1] - bpnn.node_ls[1][0].b[0]
            dot = Square(
                side_length=0.1, color=BLUE_B if data_out[idx][0] == 0 else YELLOW_B,
                fill_opacity=True, fill_color=BLACK,
                stroke_width=2
            ).move_to(
                axes_act.coords_to_point(x, 1/(1+np.exp(-x)))
            ).rotate(40*DEGREES)
            return dot
        act_dot_queue = [
            update_dot(idx).add_updater(lambda x, i=idx: x.become(update_dot(i))) for idx in range(len(data_in))
        ]
        self.add(axes_act, act_func_obj, *act_dot_queue)

        # 添加误差曲线
        axes_err = Axes(
            x_range=[0, 600, 20],
            y_range=[0, 0.5, 0.1],
            x_length=5,
            y_length=2,
            y_axis_config={
                "numbers_to_include": np.array([0.1, 0.3, 0.5])
            },
            tips=False
        ).move_to((-3, -2.3, 0))
        axes_err_label = VGroup(
            Text("迭代次数", font="SimSun", font_size=20).move_to(axes_err.coords_to_point(axes_err.x_range[1]/2, -0.08)),
            Text("平均误差", font="SimSun", font_size=20).rotate(90*DEGREES).move_to(axes_err.coords_to_point(-100, axes_err.y_range[1]/2)),
        )
        acc_err = 0.02
        acc_err_line = VGroup(
            Line(
                start=axes_err.coords_to_point(0, acc_err),
                end=axes_err.coords_to_point(axes_err.x_range[1], acc_err),
                color=RED_A,
                stroke_width=0.5
            ),
            Text("容差", font="SimSun", font_size=20, color=RED_A).move_to(axes_err.coords_to_point(0.9*axes_err.x_range[1], acc_err+0.038)),
            Text(f"{acc_err:.2f}", font="Times New Roman", font_size=20, color=RED_A).move_to(axes_err.coords_to_point(0.9*axes_err.x_range[1], acc_err-0.032))
        )
        self.add(axes_err, axes_err_label, acc_err_line)

        # 开始训练神经网络
        def back(input_data: list, except_out_data: list, alpha: float = 0.5, e: float = 1e-2) -> float:
            """误差反向传播"""
            """
            input_data 输入数据集列表
            except_out_data 预期输出列表
            alpha 学习率
            e 容差
            """
            if len(input_data[0]) != bpnn.node_numbers_ls[0]:
                raise ValueError(f"数据输入维度{len(input_data[0])}与输入层节点个数{bpnn.node_numbers_ls[0]}不匹配")
            if len(except_out_data[0]) != bpnn.node_numbers_ls[-1]:
                raise ValueError(f"预期输出维度{len(except_out_data)}与输出层节点个数{bpnn.node_numbers_ls[-1]}不匹配")

            def get_args_ls() -> list:
                """获取参数列表"""
                args_ls = []
                for layer_idx in range(1, bpnn.layer_numbers):
                    for node_idx in range(bpnn.node_numbers_ls[layer_idx]):
                        args_ls.extend(bpnn.node_ls[layer_idx][node_idx].w_ls)
                        args_ls.append(bpnn.node_ls[layer_idx][node_idx].b[0])
                return args_ls

            def update_args(args_ls: list) -> None:
                """依据参数列表，更新节点参数"""
                layer_idx = 1
                node_idx = 0
                wight_idx = 0
                for args_idx in range(len(args_ls)):
                    if wight_idx < len(bpnn.node_ls[layer_idx][node_idx].w_ls):
                        bpnn.node_ls[layer_idx][node_idx].w_ls[wight_idx] = args_ls[args_idx]
                        wight_idx += 1
                    else:  # 权重已完成赋值，更新阈值后转向下一个节点
                        bpnn.node_ls[layer_idx][node_idx].b = [args_ls[args_idx]]
                        wight_idx = 0
                        node_idx += 1
                        if node_idx >= bpnn.node_numbers_ls[layer_idx]:  # 该层节点已完成赋值，转向下一层
                            layer_idx += 1
                            node_idx = 0

            def cal_error(now_output_data: list) -> float:
                """计算误差（损失函数）"""
                err = 0
                for data_idx in range(len(except_out_data)):
                    err += np.linalg.norm(np.array(now_output_data[data_idx]) - np.array(except_out_data[data_idx]))
                return err / len(input_data)

            args_ls = get_args_ls()

            # 梯度下降
            def cal_grad(args_ls, h=1e-6) -> list:
                """计算数值梯度"""
                """
                args_ls 计算位置
                h 数值步长
                """
                grad = []
                # 计算初始误差
                update_args(args_ls)
                error_init = cal_error(bpnn.update_by_data(input_data))
                for args_idx in range(len(args_ls)):
                    args_temp = args_ls[:]
                    args_temp[args_idx] += h
                    update_args(args_temp)
                    grad.append((cal_error(bpnn.update_by_data(input_data)) - error_init) / h)
                return grad

            times_label = Text(" ")
            self.add(times_label)
            def update_times_label(times, err):
                times_label = VGroup(
                    Text(f"梯度下降迭代第{times:0>3}轮", font="SimSun", font_size=20, color=YELLOW_E),
                    Text(f"平均误差{err:.4f}", font="SimSun", font_size=20, color=YELLOW_E)
                ).arrange(DOWN, buff=0.1).move_to(axes.coords_to_point(2, 6))
                return times_label
            times = 0
            while cal_error(bpnn.update_by_data(input_data)) > e:
                times += 1
                grad_now = cal_grad(args_ls)
                args_ls = list(np.array(args_ls) - alpha * np.array(grad_now))  # 向负梯度方向前进
                update_args(args_ls)  # 更新以便于在下轮循环开启前计算输出
                # manim方面
                cut_line.become(update_cut_line())
                area = update_area()
                area1.become(area[0])
                area2.become(area[1])
                w_b_update()  # 展示新的权重和阈值
                err_now = cal_error(bpnn.update_by_data(input_data))
                times_label.become(update_times_label(times, err_now))  # 更新图像上边展示的字
                for dot_obj in act_dot_queue:
                    dot_obj.update()
                self.add(Dot(
                    point=axes_err.coords_to_point(times, err_now),
                    radius=0.02,
                    color=ORANGE
                ))
                self.wait(0.1)
            update_args(args_ls)  # 迭代完成，更新参数
            return cal_error(bpnn.update_by_data(input_data))  # 返回误差
        back(data_in, data_out, 0.5, e=acc_err)
        self.wait()
