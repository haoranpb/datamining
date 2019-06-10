import csv
import utm
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

def check_validation(value_dict):
    for key, value in value_dict.items():
        # print(type(value))
        if 'RNC' in key or 'Cell' in key or 'Dbm' in key or 'Asu' in key or 'Signal' in key or 'Longitude' in key or 'Latitude' in key:
            if value == '-1' or value == '-999':
                return False
    return True

signal_tower_dict = {}
with open('../data/gongcan.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        coordinate = [row['Latitude'], row['Longitude']]
        signal_tower_dict[row['RNCID'] + '|' + row['CellID']] = coordinate
# print(json.dumps(signal_tower_dict, indent=2))

X = []
Y = []
max_1 = 0
max_2 = 0
min_1 = 999
min_2 = 999
with open('../data/train_2g.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if(check_validation(row)): # 如果有无效数据，直接丢弃
            Y.append([row['Latitude'], row['Longitude']])
            max_1 = max(max_1, float(row['Latitude']))
            max_2 = max(max_2, float(row['Longitude']))
            min_1 = min(min_1, float(row['Latitude']))
            min_2 = min(min_2, float(row['Longitude']))

print(min_1, min_2, max_1, max_2)
print(utm.from_latlon(min_1, min_2))
print(utm.from_latlon(max_1, max_2))

            # print(utm.from_latlon(float(row['Latitude']), float(row['Longitude'])))
            # break
            # mr_sample = np.zeros((6,5))

            # for i in range(6):
            #     coordinate = signal_tower_dict[row['RNCID_' + str(i+1)] + '|' + row['CellID_' + str(i+1)]]
            #     mr_sample[i][0] = coordinate[0] # 转换成信号站的经纬度坐标
            #     mr_sample[i][1] = coordinate[1]
            #     mr_sample[i][2] = row['Dbm_' + str(i+1)]
            #     mr_sample[i][3] = row['AsuLevel_' + str(i+1)]
            #     mr_sample[i][4] = row['SignalLevel_' + str(i+1)]

            # X.append(np.array([mr_sample]).T)

# LEN = len(X)
# # print(LEN)
# X = np.array(X).astype('float32')
# Y = np.array(Y).astype('float32')
# # print(Y.shape)
# x_train = X[0: int(0.8*LEN)]
# y_train = Y[0: int(0.8*LEN)]
# x_test = X[int(0.8*LEN):]
# y_test = Y[int(0.8*LEN):]

