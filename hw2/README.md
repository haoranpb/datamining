# 数据挖掘 hw 2 说明文档

姓名：孙浩然

学号：1652714

## 开发环境：

* 系统：macOS Mojave 10.14.4
* 虚拟环境：pyenv 1.2.9
* Python 3.7.2
* scikit-learn 0.20.3
* numpy 1.16.0

## 运行环境：

- Python3
- scikit-learn 0.20.3
- numpy 1.16.0

## 运行方式：

### 目录结构：

注意：最终结果文件**dianping.csv**在`./data`目录下

```
├── README.md
├── 实验报告文档.pdf
├── data
│   ├── dianping.csv										# 最终结果文件
│   ├── dianping_figuration_vecs.csv    # 原数据文件
│   ├── ......													# 原数据文件
│   ├── dianping_user_vec.csv						# 原数据文件
│   ├── id_gender.csv
│   ├── id_malicious_fnum_fave.csv      # 特征向量文件
│   ├── id_review_vec.csv								# 特征向量文件
│   └── id_vec.csv											# 特征向量文件
├── get_new_feature.py									# 生成新特征脚本
├── parse_data.py												# 200维坐标处理文件
├── pre_prossesor.py										# 生成 id_mff 特征向量
└── predict.py													# 主预测函数
```

### 运行：

大部分代码已经 `predict.py` 中已经被注释掉了，想要先运行，需要取消注释。

