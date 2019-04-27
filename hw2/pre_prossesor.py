"""

"""
import numpy as np

if __name__ == "__main__":
    id_gender = {}
    with open('./data/id_gender.csv', 'r') as file:
        for line in file:
            data = line.split(',')
            id_gender[data[0]] = int(data[1])


    i = 0
    id_malicious = {}
    with open('./data/dianping_malicious_user.csv', 'r') as file:
        for line in file:
            if i == 0:
                i += 1
                continue
            else:
                data = line.strip('\n').split(',')
                if data[1] in id_gender:
                    id_malicious[data[1]] = data[2]


    id_fnum_fave = {}
    with open('./data/dianping_figuration_vecs.csv', 'r') as file:
        for line in file:
            if i == 1:
                i += 1
                continue
            else:
                data = line.strip('\n').split('[')
                id_parse = data[0].split(',')
                if id_parse[1] in id_gender:
                    figuration = data[1].split(']')[0].split(', ')
                    figuration = list(map(int, figuration))
                    id_fnum_fave[id_parse[1]] = [str(len(figuration)), str(round(np.average(figuration)))]


    with open('./data/id_malicious_fnum_fave.csv', 'w') as file:
        for key in id_gender:
            file.write(key + ',' + id_malicious[key] + ',' + ','.join(id_fnum_fave[key]) + '\n')