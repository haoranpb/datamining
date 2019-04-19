# 数据挖掘 Task 2 说明文档

姓名：孙浩然

学号：1652714

## 开发环境：

* 系统：macOS Mojave 10.14.4
* 虚拟环境：pyenv 1.2.9
* Python 3.7.2
* scikit-learn 0.20.3
* matplotlib 3.0.3
* numpy 1.16.0

## 运行环境：

- Python3
- scikit-learn 0.20.3
- matplotlib 3.0.3
- numpy 1.16.0

## 运行方式：

### 目录结构：

注意：最终结果文件**task2.csv**在`./data`目录下

```
├── README.md
├── 实验报告文档.pdf
├── data
│   ├── parsed_data.json              # 通过
│   ├── task2.csv                     # 最终结果文件
│   ├── task2_tmp.csv                 # 生成最终结果文件的中间文件
│   ├── review_dianping_12992.json    # 原代码
│   ├── stop_word.txt                 # 筛选词汇所用的文件
├── pre_processor.py                  # 预处理脚本文件
├── preview_data.py                   # 预览数据脚本
├── sci2014.py                        # sci2014应用
└── task2.py                          # 剩下五种算法应用
```

### 运行：

注意：`data` 文件夹需要和 `task2.py` 放在同一目录下

请按如下顺序运行文件，否则有可能出错：

**运行过程时间很长，请耐心等待**

```shell
# 先输出预览结果
python3 review_data.py

# 预处理数据
python3 pre_processor.py

# 应用sci2014，并生成./data/task2_tmp.csv文件
python3 sci2014.py

# 应用其他五种算法，最终生成./data/task2.csv文件
python3 task2.py
```

