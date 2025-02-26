此处提供BP神经网络演示视频的全部manim源码，使用者可免费且无条件地使用所有开源内容。

[![cc0][cc0-image]][cc0]

[cc0]: https://creativecommons.org/public-domain/cc0/
[cc0-image]: https://licensebuttons.net/p/zero/1.0/88x31.png

[![Python](https://camo.githubusercontent.com/36a52016e02020b1b2b3a4b07812957a13bf404e03a8793f1793415a6a40be22/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d76332e31312d677265656e2e7376673f7374796c653d666c6174)](https://www.python.org/) [![manim](https://camo.githubusercontent.com/5d142d7c8431408522b6a907e828d11de85f9ff8e0d679b134dc5e3af1319886/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6d616e696d2d76302e31382e302d677265656e2e7376673f7374796c653d666c6174)](https://github.com/3b1b/manim)



## 分镜表与代码清单

> [!NOTE]
>
> **部分md阅读器不支持嵌入html，分镜表可在*分镜表.html*文件中查看**



<!DOCTYPE html>
<head>
    <meta charset="UTF-8">
</head>
<body>
    <table id="table1" width="90%" border="1px" style="border-collapse: collapse;" align="center">
        <caption><h3>manim镜头分镜设计表</h3></caption>
        <thead>
            <th align="center">镜号<br>(同文件名与类名)</th>
            <th align="center">入点<br>(时:分:秒:帧)</th>
            <th align="center">镜头</th>
        </thead>
        <tbody>
            <tr>
                <td align="left" colspan="3"><b>A 支持向量机</b></td>
            </tr>
            <!-- MA1 -->
            <tr>
                <td align="center">MA1</td>
                <td align="center">00:00:00:00</td>
                <td align="left">两个圆形和两个三角形<br>展示能够完成分类的直线<br>直线两侧填色</td>
            </tr>
            <!-- MA2 -->
            <tr>
                <td align="center">MA2</td>
                <td align="center">00:01:06:55</td>
                <td align="left">描述量化过程<br>展示分类的理论支撑</td>
            </tr>
            <!-- MA3 -->
            <tr>
                <td align="center">MA3</td>
                <td align="center">00:03:47:20</td>
                <td align="left">绘制阶跃函数<br>点从左侧移至右侧</td>
            </tr>
            <!-- MA4 -->
            <tr>
                <td align="center">MA4</td>
                <td align="center">00:04:23:29</td>
                <td align="left">展示各点到直线距离的垂线，直线旋转<br>展示sigmoid函数<br>直线旋转，带出缓冲区虚线</td>
            </tr>
            <!-- MA5 -->
            <tr>
                <td align="center">MA5</td>
                <td align="center">00:05:30:13</td>
                <td align="left">回顾此前过程<br>引出SVM拓扑结构</td>
            </tr>
            <!-- MA6 -->
            <tr>
                <td align="center">MA6</td>
                <td align="center">00:08:10:00</td>
                <td align="left">线性可分情况下，对训练集数据整体训练的BPNN（2→1）</td>
            </tr>
            <!-- MA7 -->
            <tr>
                <td align="center">MA7</td>
                <td align="center">00:09:02:43</td>
                <td align="left">线性可分情况下，对训练集数据逐个训练的BPNN（2→1）</td>
            </tr>
            <!-- MA8 -->
            <tr>
                <td align="center">MA8</td>
                <td align="center">00:09:50:42</td>
                <td align="left">线性可分情况下，对训练集数据整体训练的BPNN（3→1）</td>
            </tr>
            <!-- MA9 -->
            <tr>
                <td align="center">MA9</td>
                <td align="center">00:07:34:54</td>
                <td align="left">展示常见激活函数的名称、表达式、图像</td>
            </tr>
            <tr>
                <td align="left" colspan="3"><b>B 多层感知机</b></td>
            </tr>
            <!-- MB1 -->
            <tr>
                <td align="center">MB1</td>
                <td align="center">00:11:44:47</td>
                <td align="left">矩阵变换的几何示意</td>
            </tr>
            <!-- MB2 -->
            <tr>
                <td align="center">MB2</td>
                <td align="center">00:11:49:41</td>
                <td align="left">矩阵变换的代数注解</td>
            </tr>
            <!-- MB3 -->
            <tr>
                <td align="center">MB3</td>
                <td align="center">00:12:14:49</td>
                <td align="left">不同维度输入到二维的变换矩阵</td>
            </tr>
            <tr>
                <td align="left" colspan="3"><b>C 总结与祛魅</b></td>
            </tr>
            <!-- MC1 -->
            <tr>
                <td align="center">MC1</td>
                <td align="center">00:14:57:43</td>
                <td align="left">激活函数改变，隐藏层增加，节点数量改变</td>
            </tr>
        </tbody>
    </table>
</body>




## 渲染命令


```
manim MA1.py MA1
```

其中，MA1.py为文件名，MA1为类名

亦可在该文件下补充以下代码，直接运行文件：

```python
if __name__ == '__main__':
	import os
    os.system('manim MA1.py MA1')
```

