"""
    Task 2: Preview Data
"""
import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    stop_word = [] # 剔除停用词
    with open('./data/stop_word.txt', 'r') as file:
        for line in file:
            stop_word.append(line)

    with open('./data/review_dianping_12992.json', 'r') as file:
        json_dict = json.load(file)


    all_customer_dict = {} # 所有用户的用词次数统计
    customer_count = {} # 用于分析用户评论次数分布
    word_count = []
    i = 0
    for key in json_dict:
        i += 1
        if i % 1000 == 0:
            print('处理到第' + str(i) + '个顾客')
        if len(json_dict[key]) > 0:
            words = json_dict[key][0].split() # 取出每个用户的所有单词
            LEN = len(words)
            word_count.append(LEN)
            if LEN in customer_count:
                customer_count[LEN] += 1
            else:
                customer_count[LEN] = 1

        customer_dict = {}
        for word in words:
            if word in stop_word or LEN < 2:
                continue
            if word in customer_dict:
                customer_dict[word] += 1
                all_customer_dict[word] += 1
            else:
                customer_dict[word] = 1

                if word in all_customer_dict:
                    all_customer_dict[word] += 1
                else:
                    all_customer_dict[word] = 1

        json_dict[key] = customer_dict

    word_items = []
    for item in all_customer_dict.items():
        if item[1] > 1000:
            word_items.append(item)

    print('使用量最多的五十个词汇')
    j = 0
    for item in sorted(word_items, key=lambda item:item[1], reverse=True):
        j += 1
        if j > 50:
            break
        print(item)

    print('\n\n### 每人词汇使用量情况 ###')
    print('平均每人使用词汇', np.average(word_count))
    print('每人使用词汇中位数', np.median(word_count))
    print('每人使用词汇最大值', np.amax(word_count))
    print('每人使用词汇最小值', np.amin(word_count))
    plt.scatter(list(customer_count.keys()), list(customer_count.values()), s=10)
    plt.show()


