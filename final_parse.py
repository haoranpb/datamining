"""
    Task 2 for the first assignment of data mining
"""
import json
import numpy as np


if __name__ == "__main__":
    selected_word = ['调料', '火锅', '馄饨', '外皮', '口感', '店面', '评价', '效果', '霸王餐', '原价']

    with open('./Data/review_dianping_12992.json', 'r') as file:
        json_dict = json.load(file)

    parsed_dict = {}
    for key in json_dict:
        if len(json_dict[key]) > 0:
            words = json_dict[key][0].split() # 取出所有单词
            if len(words) > 250 and len(words) < 2000: # 合理的范围
                customer_dict = {}
                for word in selected_word:
                    customer_dict[word] = 0
                for word in words:
                    if word in selected_word:
                        customer_dict[word] += 1

                parsed_dict[key] = customer_dict
    with open('./Data/parsed_data.json', 'w') as file:
        json.dump(parsed_dict, file, indent=2)
