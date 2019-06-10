import csv
import utm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

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
with open('../data/train_2g.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if(check_validation(row)): # 如果有无效数据，直接丢弃
            Y.append([row['Latitude'], row['Longitude']])
            mr_sample = np.zeros((6, 5))

            for i in range(6):
                coordinate = signal_tower_dict[row['RNCID_' + str(i+1)] + '|' + row['CellID_' + str(i+1)]]
                mr_sample[i][0] = coordinate[0] # 转换成信号站的经纬度坐标
                mr_sample[i][1] = coordinate[1]
                mr_sample[i][2] = row['Dbm_' + str(i+1)]
                mr_sample[i][3] = row['AsuLevel_' + str(i+1)]
                mr_sample[i][4] = row['SignalLevel_' + str(i+1)]

            X.append(np.array([mr_sample]).T)

LEN = len(X)

X = np.array(X).astype('float32')
Y = np.array(Y).astype('float32')


adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-10, amsgrad=False)
model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(5, 6, 1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2))

model.compile(optimizer=adam, loss='mean_squared_error')

model.fit(X, Y, epochs=100, batch_size=32, shuffle=True)
