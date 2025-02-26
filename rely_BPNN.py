"""BPNN依赖文件"""
from manim import *
import numpy as np


# manim参数
font_type = "Times New Roman"
font_size = 20
shift_b = 0.5       # 权重b标签向右偏移量
shift_w = 0.6       # 阈值w标签横向偏移量
shift_value = 0.4   # 数值value标签向下偏移量


def activation_function(x):
    """BPNN激活函数"""
    return 1/(1+np.exp(-x))


class Node:
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


class BPNN:
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

