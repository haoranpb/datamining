import csv
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import utm

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

    if len(validate_list) < 3: # 我认为基站数量少于3，即理论上不可能得出手机坐标
        return False, validate_data
    else:
        for i in range(6):
            if i not in validate_list:
                random.seed(1)
                k = random.choice(validate_list)
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
        Y.append([coordinate[0], coordinate[1]])
        mr_sample = np.zeros((6, 5))

        for i in range(6):
            coordinate = signal_tower_dict[parsed_row['RNCID_' + str(i+1)] + '|' + parsed_row['CellID_' + str(i+1)]]
            mr_sample[i][0] = coordinate[0] # 转换成信号站的经纬度坐标
            mr_sample[i][1] = coordinate[1]
            mr_sample[i][2] = parsed_row['Dbm_' + str(i+1)]
            mr_sample[i][3] = parsed_row['AsuLevel_' + str(i+1)]
            mr_sample[i][4] = parsed_row['SignalLevel_' + str(i+1)]

        X.append(np.array([mr_sample]).T)

LEN = len(X)

X = np.array(X).astype('float32')
Y = np.array(Y).astype('float32')
X_Train = X[:int(0.9*LEN)]
Y_Train = Y[:int(0.9*LEN)]

X_Test = X[:int(0.1*LEN):]
Y_Test = Y[:int(0.1*LEN):]

adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(5, 6, 1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2))

model.compile(optimizer=adam, loss='mean_squared_error')

model.fit(X_Train, Y_Train, epochs=300, batch_size=32, shuffle=True)

result = model.predict(X_Test)


def calcu_distance(true_latitude, true_longitude, pred_latitude, pred_longitude):
    vector1 = np.array([true_latitude, true_longitude])
    vector2 = np.array([pred_latitude, pred_longitude])
    return np.sqrt(np.sum(np.square(vector1 - vector2)))

error_list = []
for i in range(int(0.1*LEN)):
    error = calcu_distance(Y_Test[i][0], Y_Test[i][1], result[i][0], result[i][1])
    error_list.append(error)

print(np.median(error_list)) # 这就是中位误差？
print(np.mean(error_list)) # 这就是中位误差？