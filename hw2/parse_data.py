"""
    处理 dianping_user_vec.csv 文件，处理成更简单的格式
"""
id_gender = {}
i = 0
with open('./data/dianping_gender.csv', 'r') as file:
    for line in file:
        if i == 0:
            i += 1
            continue
        else:
            data = line.split(',')
            id_gender[data[1]] =  int(data[2])


id_vec = {}
with open('./data/dianping_user_vec.csv', 'r') as file:
    usr_id = ''
    for line in file:
        if i == 1:
            i += 1
            continue


        if '[' in line:
            tmp = line.split(',')
            usr_id = tmp[1]
            # 如果在ID-性别里没有，那就不要这个数据了
            if usr_id not in id_gender:
                usr_id = '0'
            id_vec[usr_id] = []
            tmp = tmp[2].split('[')[1].split()
        elif ']' in line:
            tmp = line.strip().split(']')
            tmp = tmp[0].split()
        else:
            tmp = line.strip().split()

        for num in tmp:
            id_vec[usr_id].append(num)



with open('./data/id_vec.csv', 'w') as file:
    for key in id_vec:
        if key != '0':
            file.write(key + ',' + ','.join(id_vec[key]) + '\n')

with open('./data/id_gender.csv', 'w') as file:
    for key in id_gender:
        file.write(key + ',' + str(id_gender[key]) + '\n')
    
                
