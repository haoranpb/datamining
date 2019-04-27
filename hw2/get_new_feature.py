"""
    Get a new feature from dianping_review.csv
"""

import jieba

if __name__ == "__main__":
    id_gender = {}
    with open('./data/id_gender.csv', 'r') as file:
        for line in file:
            data = line.split(',')
            id_gender[data[0]] = int(data[1])


    # all_usr_count = {}
    per_usr_count = {}
    words = {
        '不错': 0,
        '味道': 1,
        '好吃': 2,
        '可以': 3,
        '还是': 4,
        '没有': 5,
        '就是': 6,
        '感觉': 7,
        '比较': 8,
        '喜欢': 9
    }
    i = 0
    with open('./data/dianping_review.csv', 'r') as file:
        for line in file:
            if i == 0:
                i += 1
                continue
            else:
                i += 1
                if i%100 == 0:
                    print('处理到了第' + str(i) + '行')
                data = line.split(',')
                if data[2] not in id_gender:
                    continue
                if data[2] not in per_usr_count:
                    per_usr_count[data[2]] = [0 for i in range(10)]

                seg_list = list(jieba.cut(data[-1].strip('\n'), cut_all=False))
                for word in seg_list:
                    if len(word) <2:
                        continue
                    else:
                        if word in words:
                            per_usr_count[data[2]][words[word]] += 1


    with open('./data/id_review_vec.csv', 'w') as file:
        for key in per_usr_count:
            file.write(key + ',' + ','.join(list(map(str, per_usr_count[key]))) + '\n')
                        # if word in per_usr_count[data[2]]:
                        #     per_usr_count[data[2]][word] += 1
                        #     all_usr_count[word] += 1
                        # else:
                        #     per_usr_count[data[2]][word] = 1
                        #     if word in all_usr_count:
                        #         all_usr_count[word] += 1
                        #     else:
                        #         all_usr_count[word] = 1

    # temp = []
    # for key, value in all_usr_count.items():
    #     if value < 1000:
    #         continue
    #     else:
    #         temp.append((value, key))
    # temp.sort(reverse=True)
    # for i in range(50):
    #     print(temp[i])
