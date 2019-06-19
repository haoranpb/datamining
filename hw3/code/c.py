import csv
import random
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from keras.optimizers import Adam
import utm

def shuffle_train_test(X, Y, LEN):
    random.seed(1)
    random_list = random.sample(range(LEN), k=int(0.1*LEN))
    # random_list.sort()
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
    if float(value_dict['Accuracy']) > 30 or int(value_dict['TrajID']) > 69:
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

X.sort(key=lambda row: row[0]) # 按照时间戳排序
X = np.array(X).astype('float64')
Y = np.array(Y).astype('float64')
print(X.shape)

# 按照轨迹ID进行切割
tmp_count = 1
tmp_X = []
tmp_Y = []
x_slice_list = []
y_slice_list = []
tmp_traj_label = round(X[0][1])
for i in range(LEN):
    traj_id = round(X[i][1])
    if tmp_count == slice_length:
        if traj_id == tmp_traj_label:
            x_slice_list.append(X[i])
            tmp_X += x_slice_list
            x_slice_list = []
            tmp_count = 1
            tmp_traj_label = round(X[i+1][1])
            break
        else:
            x_slice_list.insert(0, X[i-5])
            tmp_X += x_slice_list
            x_slice_list = []
            tmp_count = 1
            tmp_traj_label = traj_id
    else:
        if traj_id == tmp_traj_label:
            tmp_count += 1
            x_slice_list.append(X[i])
        else:
            x_slice_list = []
            for k in range(1, 7): # 从这个位置起，回退六个
                x_slice_list.append(X[i+k-7])
                tmp_X += x_slice_list
                tmp_traj_label = traj_id
                x_slice_list = []
print(len(X))
print(len(tmp_X))

#     if tmp_traj_label == '' or tmp_traj_label != X[i][1]:
#         tmp_traj_label = X[i][1]
#         tmp_count = 1
#     else:
#         tmp_count += 1

#     if tmp_count


# LEN = int(LEN/slice_length)*slice_length
# X = X[:LEN]
# Y = Y[:LEN]




# scalerX = preprocessing.StandardScaler()
# scalerY = preprocessing.StandardScaler()

# X = scalerX.fit_transform(X)
# Y = scalerY.fit_transform(Y)

# X = X.reshape((round(LEN/slice_length), slice_length, features))
# Y = Y.reshape((round(LEN/slice_length), slice_length*2))

# X_Train, Y_Train, X_Test, Y_Test = shuffle_train_test(X, Y, round(LEN/slice_length))

# adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
# model = Sequential()

# model.add(LSTM(slice_length*10, input_shape=(X_Train.shape[1], X_Train.shape[2])))
# model.add(Dense(2*slice_length))
# model.compile(loss='mean_squared_error', optimizer=adam)
# model.fit(X_Train, Y_Train, epochs=500, batch_size=64)
# result = model.predict(X_Test)

# result = result.reshape(int(0.1*LEN), 2)
# Y_Test = Y_Test.reshape(int(0.1*LEN), 2)

# result = scalerY.inverse_transform(result)
# Y_Test = scalerY.inverse_transform(Y_Test)

# def calcu_distance(true_latitude, true_longitude, pred_latitude, pred_longitude):
#     vector1 = np.array([true_latitude, true_longitude])
#     vector2 = np.array([pred_latitude, pred_longitude])
#     return np.sqrt(np.sum(np.square(vector1 - vector2)))

# error_list = []
# for i in range(int(0.1*LEN)):
#     error = calcu_distance(Y_Test[i][0], Y_Test[i][1], result[i][0], result[i][1])
#     error_list.append(error)

# print(np.median(error_list)) # 这就是中位误差？
# print(np.mean(error_list)) # 这就是中位误差？

# plt.figure()
# plt.scatter(Y_Test[:,0], Y_Test[:,1], c='blue', s=5)
# # plt.scatter(Y_Train[:,0], Y_Train[:,1], c='red', s=3)
# plt.scatter(result[:,0], result[:,1], c='red', s=3)

# plt.show()
