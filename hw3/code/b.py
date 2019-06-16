import csv
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten

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

X = []
Y = []
max_longitude = 0
max_latitude = 0
min_latitude = 999
min_longitude = 999
with open('../data/train_2g.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if(check_validation(row)): # 如果有无效数据，直接丢弃
            Y.append([row['Latitude'], row['Longitude']])
            max_latitude = max(max_latitude, float(row['Latitude']))
            max_longitude = max(max_longitude, float(row['Longitude']))
            min_latitude = min(min_latitude, float(row['Latitude']))
            min_longitude = min(min_longitude, float(row['Longitude']))

print('MAX:', max_latitude, max_longitude)
print('MIN:', min_latitude, min_longitude)
print('-latitude', (max_latitude - min_latitude)*1000) # 12 -> 3
print('-longitude', (max_longitude - min_longitude)*1000) # 16 -> 4

la_size = 6
lo_size = 8

la = (max_latitude - min_latitude)/la_size
lo = (max_longitude - min_longitude)/lo_size
tmp_sum = np.zeros((la_size, lo_size))

def coordinate_to_label(latitude, longitude):
    lo_label = int((longitude - min_longitude - 1e-10)/lo)
    la_label = int((latitude - min_latitude - 1e-10)/la)
    # print(la_label, lo_label)
    # if lo_label == 3
    tmp = np.zeros((la_size, lo_size))
    tmp[la_label][lo_label] = 1
    tmp_sum[la_label][lo_label] += 1
    return tmp



for i in range(len(Y)):
    new_y = coordinate_to_label(float(Y[i][0]), float(Y[i][1]))


print(tmp_sum)
print(np.sum(tmp_sum))

def label_to_coordinate(label):
    pass


# print(utm.from_latlon(min_1, min_2))

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

