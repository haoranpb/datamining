import csv
import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D

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
            mr_sample = np.zeros((6,5))
            # print(row)
            for i in range(6):
                coordinate = signal_tower_dict[row['RNCID_' + str(i+1)] + '|' + row['CellID_' + str(i+1)]]
                mr_sample[i][0] = coordinate[0] # 转换成信号站的经纬度坐标
                mr_sample[i][1] = coordinate[1]
                mr_sample[i][2] = row['Dbm_' + str(i+1)]
                mr_sample[i][3] = row['AsuLevel_' + str(i+1)]
                mr_sample[i][4] = row['SignalLevel_' + str(i+1)]
            # print(mr_sample)
            X.append(mr_sample)

LEN = len(X)
print(LEN)
X = np.array(X).astype('float32')
Y = np.array(Y).astype('float32')
x_train = X[0: int(0.8*LEN)]
y_train = Y[0: int(0.8*LEN)]
x_test = X[int(0.8*LEN):]
y_test = Y[int(0.8*LEN):]
print(X.shape)


# model = Sequential()
# model.add(Conv2D(10, 3, 3, activation='relu', input_shape=(6, 5, 1)))


# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])


# model.fit(x_train, y_train, batch_size=16, epochs=2, validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])