import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import utm

max_latitude = 330414.05273900216
max_longitude = 3463503.3311055717
min_latitude = 328787.97245886514
min_longitude = 3462203.7096695383

la_size = 163
lo_size = 130
la = (max_latitude - min_latitude)/la_size
lo = (max_longitude - min_longitude)/lo_size

def coordinate_to_label(coordinate):
    la_label = int((coordinate[0] - min_latitude - 1e-11)/la)
    lo_label = int((coordinate[1] - min_longitude - 1e-11)/lo)
    tmp = np.zeros(la_size * lo_size, dtype=int)
    tmp[la_label+ lo_label*la_size] = 1
    return tmp

def shuffle_train_test(X, Y, LEN):
    random.seed(1)
    random_list = random.sample(range(LEN), k=int(0.1*LEN))
    X_Train = []
    Y_Train = []
    X_Test = []
    Y_Test = []
    for i in range(LEN):
        if i in random_list:
            X_Test.append(X[i])
            Y_Test.append(Y[i])
        else:
            X_Train.append(X[i])
            Y_Train.append(Y[i])
    return np.array(X_Train), np.array(Y_Train), np.array(X_Test), np.array(Y_Test)

# lat, lon, 51 'R'
def check_validation(value_dict):
    validate_list = []
    validate_data = {
        'Latitude': value_dict['Latitude'],
        'Longitude': value_dict['Longitude']
    }
    for i in range(6):
        if value_dict['RNCID_' + str(i+1)] != '-999' and value_dict['CellID_' + str(i+1)] != '-999' and value_dict['Dbm_' + str(i+1)] != '-999' \
            and value_dict['AsuLevel_' + str(i+1)] != '-1' and value_dict['SignalLevel_' + str(i+1)] != '-1':
            
            validate_list.append(i)
            validate_data['RNCID_' + str(i+1)] = value_dict['RNCID_' + str(i+1)]
            validate_data['CellID_' + str(i+1)] = value_dict['CellID_' + str(i+1)]
            validate_data['Dbm_' + str(i+1)] = value_dict['Dbm_' + str(i+1)]
            validate_data['AsuLevel_' + str(i+1)] = value_dict['AsuLevel_' + str(i+1)]
            validate_data['SignalLevel_' + str(i+1)] = value_dict['SignalLevel_' + str(i+1)]

    if len(validate_list) < 5: # 我认为基站数量少于3，即理论上不可能得出手机坐标
        return False, validate_data
    else:
        for i in range(6):
            if i not in validate_list:
                np.random.seed(1)
                k = np.random.choice(validate_list)
                validate_data['RNCID_' + str(i+1)] = value_dict['RNCID_' + str(k+1)]
                validate_data['CellID_' + str(i+1)] = value_dict['CellID_' + str(k+1)]
                validate_data['Dbm_' + str(i+1)] = value_dict['Dbm_' + str(k+1)]
                validate_data['AsuLevel_' + str(i+1)] = value_dict['AsuLevel_' + str(k+1)]
                validate_data['SignalLevel_' + str(i+1)] = value_dict['SignalLevel_' + str(k+1)]
        return True, validate_data


signal_tower_dict = {}
with open('../data/gongcan.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        coordinate = utm.from_latlon(float(row['Latitude']), float(row['Longitude']))[:2]
        signal_tower_dict[row['RNCID'] + '|' + row['CellID']] = coordinate

X = []
Y = []
with open('../data/train_2g.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        validation, parsed_row = check_validation(row)
        if(not validation):
            continue
        coordinate = utm.from_latlon(float(parsed_row['Latitude']), float(parsed_row['Longitude']))[:2]
        Y.append(coordinate_to_label(coordinate))
        mr_sample = np.zeros((6, 5))

        for i in range(6):
            coordinate = signal_tower_dict[parsed_row['RNCID_' + str(i+1)] + '|' + parsed_row['CellID_' + str(i+1)]]
            mr_sample[i][0] = coordinate[0] # 转换成信号站的经纬度坐标
            mr_sample[i][1] = coordinate[1]
            mr_sample[i][2] = parsed_row['Dbm_' + str(i+1)]
            mr_sample[i][3] = parsed_row['AsuLevel_' + str(i+1)]
            mr_sample[i][4] = parsed_row['SignalLevel_' + str(i+1)]

        X.append(np.array(mr_sample))

LEN = len(X)
X = np.array(X).astype('float64')
X = X[:, :, :, np.newaxis]
print(X.shape)
Y = np.array(Y).astype('float64')
print(Y.shape)
X_Train, Y_Train, X_Test, Y_Test = shuffle_train_test(X, Y, LEN)


# adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
model = Sequential()

# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(6, 5, 1)))
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(40000, activation='relu'))
# model.add(Dense(40000, activation='relu'))
# model.add(Dense(21190, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_Train, Y_Train, epochs=30, batch_size=32)