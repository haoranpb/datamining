import csv
import random
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import Adam
import utm

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
        'Longitude': value_dict['Longitude'],
        'MRTime': value_dict['MRTime'],
        'TrajID': value_dict['TrajID']
    }
    if float(value_dict['Accuracy']) > 30:
        return False, validate_data

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
features = 32
with open('../data/train_2g.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        validation, parsed_row = check_validation(row)
        if(not validation):
            continue
        coordinate = utm.from_latlon(float(parsed_row['Latitude']), float(parsed_row['Longitude']))[:2]
        Y.append([coordinate[0], coordinate[1]])
        mr_sample = np.zeros(features)
        mr_sample[0] = parsed_row['MRTime']
        mr_sample[1] = parsed_row['TrajID']
        for i in range(6):
            coordinate = signal_tower_dict[parsed_row['RNCID_' + str(i+1)] + '|' + parsed_row['CellID_' + str(i+1)]]
            mr_sample[i*5 + 2] = coordinate[0] # 转换成信号站的经纬度坐标
            mr_sample[i*5 + 3] = coordinate[1]
            mr_sample[i*5 + 4] = parsed_row['Dbm_' + str(i+1)]
            mr_sample[i*5 + 5] = parsed_row['AsuLevel_' + str(i+1)]
            mr_sample[i*5 + 6] = parsed_row['SignalLevel_' + str(i+1)]
        X.append(np.array(mr_sample))

LEN = len(X)
slice_length = 6

X = np.array(X).astype('float64')
Y = np.array(Y).astype('float64')
print(X.shape)
print(Y.shape)


# 按照轨迹ID进行切割
tmp_count = 1
tmp_X = []
tmp_Y = []
x_slice_list = []
y_slice_list = []
tmp_traj_label = int(X[0][1])
for i in range(LEN):
    traj_id = int(X[i][1])
    if tmp_count == slice_length:
        if traj_id == tmp_traj_label:
            x_slice_list.append(X[i])
            y_slice_list.append(Y[i])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            x_slice_list = []
            y_slice_list = []
            tmp_count = 1
            if (i+1) < LEN:
                tmp_traj_label = int(X[i+1][1])
        else:
            x_slice_list.insert(0, X[i-slice_length])
            y_slice_list.insert(0, Y[i-slice_length])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            x_slice_list = [X[i]]
            y_slice_list = [Y[i]]
            tmp_count = 2
            tmp_traj_label = traj_id
    else:
        if traj_id == tmp_traj_label:
            tmp_count += 1
            x_slice_list.append(X[i])
            y_slice_list.append(Y[i])
        else:
            x_slice_list = []
            y_slice_list = []
            for k in range(slice_length):
                x_slice_list.append(X[i+k-slice_length])
                y_slice_list.append(Y[i+k-slice_length])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            x_slice_list = [X[i]]
            y_slice_list = [Y[i]]
            tmp_count = 2
            tmp_traj_label = traj_id

LEN = len(tmp_X)
X, Y = np.array(tmp_X), np.array(tmp_Y)
X = np.delete(X, [1], axis=1)

scalerX = preprocessing.StandardScaler()
scalerY = preprocessing.StandardScaler()

X = scalerX.fit_transform(X)
Y = scalerY.fit_transform(Y)

X = X.reshape((round(LEN/slice_length), slice_length, features-1))
Y = Y.reshape((round(LEN/slice_length), slice_length, 2))

X_Train, Y_Train, X_Test, Y_Test = shuffle_train_test(X, Y, round(LEN/slice_length))

adam = Adam(lr=4e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
model = Sequential()

model.add(LSTM(slice_length*40, input_shape=(X_Train.shape[1], X_Train.shape[2]), return_sequences=True))
model.add(TimeDistributed(Dense(72, activation='relu')))
model.add(TimeDistributed(Dense(2, activation='linear')))

model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(X_Train, Y_Train, epochs=500, batch_size=6)
result = model.predict(X_Test)

result = result.reshape(Y_Test.shape[0]*slice_length, 2)
Y_Test = Y_Test.reshape(Y_Test.shape[0]*slice_length, 2)

result = scalerY.inverse_transform(result)
Y_Test = scalerY.inverse_transform(Y_Test)

def calcu_distance(true_latitude, true_longitude, pred_latitude, pred_longitude):
    vector1 = np.array([true_latitude, true_longitude])
    vector2 = np.array([pred_latitude, pred_longitude])
    return np.sqrt(np.sum(np.square(vector1 - vector2)))

error_list = []
error_dict = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0
}
for i in range(Y_Test.shape[0]):
    error = calcu_distance(Y_Test[i][0], Y_Test[i][1], result[i][0], result[i][1])
    if error <= 20:
        error_dict['0'] += 1
    elif error <= 40:
        error_dict['1'] += 1
    elif error <= 60:
        error_dict['2'] += 1
    elif error <= 80:
        error_dict['3'] += 1
    elif error <= 100:
        error_dict['4'] += 1
    error_list.append(error)

error_list.sort()
print(np.median(error_list)) # 中位误差
print(np.mean(error_list)) # 平均误差
print(error_list[int(len(error_list)*0.9)]) # 90%误差

error_ratio = []
for i in range(5):
    if i > 0:
        error_dict[str(i)] += error_dict[str(i-1)]
    error_ratio.append(error_dict[str(i)]/len(error_list))


plt.figure('figure 1')
plt.scatter(Y_Test[:,0], Y_Test[:,1], c='blue', s=5)
plt.scatter(result[:,0], result[:,1], c='red', s=3)

plt.figure('figure 2')
plt.plot([20, 40, 60, 80, 100], error_ratio, marker='o')

plt.show()
