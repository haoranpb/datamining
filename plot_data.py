"""
    Task 2 for the first assignment of data mining
"""
import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    stop_word = []
    with open('./Data/stop_word.txt', 'r') as file:
        for line in file:
            stop_word.append(line)

    with open('./Data/review_dianping_12992.json', 'r') as file:
        json_dict = json.load(file)
    
    all_customer_dict = {}
    customer_count = {}
    word_count = []
    i = 0

    for key in json_dict:
        i += 1
        print('处理到第' + str(i) + '个顾客')
        if len(json_dict[key]) > 0:
            words = json_dict[key][0].split() # 取出所有单词
            word_count.append(len(words))
            if len(words) in customer_count:
                customer_count[len(words)] += 1
            else:
                customer_count[len(words)] = 1

        customer_dict = {}
        for word in words:
            if word in stop_word or len(word) < 2:
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

        # print(sorted(list(customer_dict.items()), key=lambda item:item[1], reverse=True))
        json_dict[key] = customer_dict
        # with open('./Data/test.json', 'w') as file:
        #     json.dump(json_dict, file, indent=2)

    word_items = []
    for item in all_customer_dict.items():
        if item[1] > 100:
            word_items.append(item)
    j = 0
    for item in sorted(word_items, key=lambda item:item[1], reverse=True):
        j += 1
        if j > 50:
            break
        print(item)

    print('### Word Count ###')
    # print(word_count)
    print('average', np.average(word_count))
    print('median', np.median(word_count))
    print('amax', np.amax(word_count))
    print('amin', np.amin(word_count))
    plt.scatter(list(customer_count.keys()), list(customer_count.values()), s=10)

    plt.show()


