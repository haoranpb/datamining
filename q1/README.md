# 数据挖掘 Task 1 说明文档

姓名：孙浩然

学号：1652714

## 开发环境：

* 系统：macOS Mojave 10.14.4
* 虚拟环境：pyenv 1.2.9
* Python 3.7.2
* matplotlib 3.0.3

## 运行环境：

- Python3
- matplotlib 3.0.3

Mac 用户注意：`pip install matplotlib` 安装完 `matplotlib` 后，**可能在运行程序时报错或生成全白的图标**，原因是 `matplotlib` 的 `backend` 设置的出错。最简单的解决方法是：

1. 先安装`pyqt5`工具包：`pip install pyqt5`
2. 之后创建 `~/.matplotlib/matplotlibrc` 文件，写入`backend: Qt5Agg`，这时运行就不该有问题了

## 运行方式：

在源代码 `line 105` 前后可以调整一些参数

```shell
cd /path/to/q1/
python3 task1.py
```

注意：`Aggregation.txt` 需要和 `task1.py` 放在同一目录下