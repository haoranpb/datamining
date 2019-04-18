"""
    Task 2 for the first assignment of data mining
"""
import json
import numpy as np


if __name__ == "__main__":
    stop_word = []
    with open('./Data/stop_word.txt', 'r') as file:
        for line in file:
            stop_word.append(line)

    with open('./Data/review_dianping_12992.json', 'r') as file:
        json_dict = json.load(file)
    

    new_dict = {}
    all_customer_dict = {}
    i = 0
    for key in json_dict:
        i += 1
        print('处理到第' + str(i) + '个顾客')
        if len(json_dict[key]) > 0:
            words = json_dict[key][0].split() # 取出所有单词
            if len(words) > 250 and len(words) < 2000: # 合理的范围
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

                new_dict[key] = customer_dict
    with open('./Data/parsed_data.json', 'w') as file:
        json.dump(new_dict, file, indent=2)

    word_items = []
    for item in all_customer_dict.items():
        if item[1] > 1000:
            word_items.append(item)
    print('word_items', len(word_items))
    temp = sorted(word_items, key=lambda item:item[1], reverse=True) # 拿出前一百，随机选取10个
    for i in np.random.randint(0, 600, 12):
        print(temp[i])


# ('调料', 5890)
# ('火锅', 13091)
# ('馄饨', 5762)
# ('小时', 14161)
# ('口感', 22304)
# ('店面', 10081)
# ('时候', 44343)
# ('效果', 7572)
# ('霸王餐', 10979)
# ('牛蛙', 13057)