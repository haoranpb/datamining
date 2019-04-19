"""
    Task 2: Pre process the raw data
"""
import json


if __name__ == "__main__":
    selected_words = ['味道', '口感', '服务态度', '服务员', '性价比', '价格']
    with open('./Data/review_dianping_12992.json', 'r') as file:
        json_dict = json.load(file)

    parsed_dict = {}
    for key in json_dict:
        if len(json_dict[key]) > 0:
            words = json_dict[key][0].split() # 取出所有单词
            if len(words) > 280 and len(words) < 800: # 合理的范围
                customer_dict = {}
                for word in selected_words:
                    customer_dict[word] = 0
                for word in words:
                    if word in selected_words:
                        customer_dict[word] += 1

                if sum(customer_dict.values()) > 0:
                    parsed_dict[key] = customer_dict

    print('处理后共有' + str(len(parsed_dict.keys())) + '个用户')
    with open('./data/parsed_data.json', 'w') as file:
        json.dump(parsed_dict, file, indent=2)
