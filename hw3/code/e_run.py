import csv
import random
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Conv2D, Flatten
from keras.optimizers import Adam
import utm

max_latitude = 330414.05273900216
max_longitude = 3463503.3311055717
min_latitude = 328787.97245886514
min_longitude = 3462203.7096695383

la_size = 16*2
lo_size = 13*2
la = (max_latitude - min_latitude)/la_size
lo = (max_longitude - min_longitude)/lo_size
print(lo)
print(la)

def coordinate_to_label(coordinate):
    la_label = int((coordinate[0] - min_latitude - 1e-12)/la)
    lo_label = int((coordinate[1] - min_longitude - 1e-12)/lo)
    tmp = np.zeros(la_size * lo_size, dtype=int)
    tmp[la_label + lo_label*la_size] = 1
    return tmp
def shuffle_train_test(X, Y, LEN,rate):
    random.seed(1)
    random_list = random.sample(range(LEN), k=int(rate*LEN))
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
            tmp_label = np.zeros((lo_size*la_size))
            #print(Y[i])
            tmp_label = coordinate_to_label(Y[i])
            Y_Train.append(tmp_label)
            # Y_Train.append(Y[i])
    return np.array(X_Train), np.array(Y_Train), np.array(X_Test), np.array(Y_Test),random_list
def shuffle_train_testlstm(X, Y, LEN,rate):
    random.seed(1)
    random_list = random.sample(range(LEN), k=int(rate * LEN))
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
            tmp_label = np.zeros((6, lo_size * la_size))
            for k in range(6):
                tmp_label[k] = coordinate_to_label(Y[i, k])
            Y_Train.append(tmp_label)
            # tmp_label = coordinate_to_label(Y[i])
            # Y_Train.append(tmp_label)
            # Y_Train.append(Y[i])
    return np.array(X_Train), np.array(Y_Train), np.array(X_Test), np.array(Y_Test), random_list
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
with open('./data/gongcan.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        coordinate = utm.from_latlon(float(row['Latitude']), float(row['Longitude']))[:2]
        signal_tower_dict[row['RNCID'] + '|' + row['CellID']] = coordinate

X = []
Y = []
features = 32
with open('./data/train_2g.csv', 'r') as file:
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

# 按照轨迹ID进行切割


scalerX = preprocessing.MinMaxScaler()

Xtrailid = X[:,1]
Xtimestep = X[:,0]

X = scalerX.fit_transform(X)
X = np.delete(X, [1], axis=1)
X_CNN = np.delete(X, [0], axis=1) # CNN训练时，不需要时间戳



X_CNN = X_CNN.reshape((round(LEN),  6, 5))
X_Train, Y_Train, X_Test, Y_Test, randomlist = shuffle_train_test(X_CNN, Y, round(LEN),rate = 0.6)
randomlist = np.sort(randomlist)
print(X_Train.shape,Y_Train.shape,Y_Test.shape)
X_Train = X_Train.reshape((X_Train.shape[0],30))
X_Test = X_Test.reshape((X_Test.shape[0],30))
scalerX = preprocessing.StandardScaler()
X_Train = scalerX.fit_transform(X_Train)
X_Test = scalerX.fit_transform(X_Test)

X_Train = X_Train.reshape((X_Train.shape[0],6,5,1))
X_Test = X_Test.reshape((X_Test.shape[0],6,5,1))
print(X_Train.shape,Y_Train.shape,Y_Test.shape)
#scalerY = preprocessing.StandardScaler()
#Y_Train = scalerY.fit_transform(Y_Train)


adam_cnn = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None,amsgrad=False)
model_cnn = Sequential()
model_cnn.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(6, 5, 1)))
model_cnn.add(Conv2D(256, kernel_size=3, activation='relu'))
model_cnn.add(Flatten())
model_cnn.add(Dense(512, activation='relu'))
model_cnn.add(Dense(256, activation='relu'))
model_cnn.add(Dense(la_size*lo_size, activation='softmax'))
model_cnn.compile(optimizer=adam_cnn, loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_Train, Y_Train, epochs=200, batch_size=64)
'''test part'''
print(X_Train.shape,Y_Train.shape,Y_Test.shape)
cnn_result = model_cnn.predict(X_Test)
print(cnn_result.shape) # 5548, 832
resulttemp = np.zeros(cnn_result.shape[0])
for i in range(cnn_result.shape[0]):
    resulttemp[i] = np.argmax(cnn_result[i])
cnn_result = resulttemp
print(cnn_result.shape,np.max(cnn_result),np.min(cnn_result))
def label_to_coordinate(label):
    #label = np.argmax(label_list)
    la_label = label % la_size
    lo_label = (label - la_label) / la_size
    return np.array([(0.5+la_label)*la + min_latitude, (0.5+lo_label)*lo + min_longitude])
def label_to_coordinate2(label_list):
    label = np.argmax(label_list)
    la_label = label % la_size
    lo_label = (label - la_label) / la_size
    return np.array([(0.5+la_label)*la + min_latitude, (0.5+lo_label)*lo + min_longitude])
def calcu_distance(true_latitude, true_longitude, pred_latitude, pred_longitude):
    vector1 = np.array([true_latitude, true_longitude])
    vector2 = np.array([pred_latitude, pred_longitude])
    return np.sqrt(np.sum(np.square(vector1 - vector2)))

resulttemp2 = np.zeros((cnn_result.shape[0],2))
print(resulttemp2.shape)
for i in range(cnn_result.shape[0]):
    resulttemp2[i] = label_to_coordinate(cnn_result[i])
print(resulttemp2.shape)
error_list = np.zeros(Y_Test.shape[0])
for i in range(Y_Test.shape[0]):
    error = calcu_distance(Y_Test[i][0], Y_Test[i][1], resulttemp2[i][0], resulttemp2[i][1])
    error_list[i] = error

print(np.median(error_list)) # 这就是中位误差？
print(np.mean(error_list)) # 这就是中位误差？
plt.figure()
plt.scatter(Y_Test[:,0], Y_Test[:,1], c='blue', s=5)
plt.scatter(resulttemp2[:,0], resulttemp2[:,1], c='red', s=3)
plt.show()

'''LSTM'''
cnn_proba = model_cnn.predict_proba(X_Test)
#切分
print(cnn_proba.shape)

tmp_count = 1
tmp_X = []
tmp_Y = []
x_slice_list = []
y_slice_list = []
tmp_traj_label = int(Xtrailid[randomlist[0]])
for i in range(len(cnn_proba)):
    traj_id = int(Xtrailid[randomlist[i]])
    if tmp_count == slice_length:
        #print(i,len(tmp_X))
        if traj_id == tmp_traj_label:
            x_slice_list.append(cnn_proba[i])
            y_slice_list.append(Y_Test[i])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            x_slice_list = []
            y_slice_list = []
            tmp_count = 1
            tmp_traj_label = int(Xtrailid[randomlist[i+1]])
        else:
            x_slice_list.insert(0, cnn_proba[i-slice_length])
            y_slice_list.insert(0, Y_Test[i-slice_length])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            x_slice_list = [cnn_proba[i]]
            y_slice_list = [Y_Test[i]]
            tmp_count = 2
            tmp_traj_label = traj_id
    else:
        if traj_id == tmp_traj_label:
            tmp_count += 1
            x_slice_list.append(cnn_proba[i])
            y_slice_list.append(Y_Test[i])
        else:
            x_slice_list = []
            y_slice_list = []
            for k in range(slice_length):
                x_slice_list.append(cnn_proba[i+k-slice_length])
                y_slice_list.append(Y_Test[i+k-slice_length])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            #print(i, len(tmp_X))
            x_slice_list = [cnn_proba[i]]
            y_slice_list = [Y_Test[i]]
            tmp_count = 2
            tmp_traj_label = traj_id


'''
for i in range(len(cnn_proba)):
    traj_id = int(Xtrailid[randomlist[i]])
    #print(randomlist[i],traj_id)
    if tmp_count == slice_length:
        if traj_id == tmp_traj_label:
            x_slice_list.append(cnn_proba[i])
            y_slice_list.append(Y_Test[i])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            x_slice_list = []
            y_slice_list = []
            tmp_count = 1
            tmp_traj_label = int(Xtrailid[randomlist[i+1]])
        else:
            x_slice_list.insert(0, cnn_proba[i-slice_length])
            y_slice_list.insert(0, Y_Test[i-slice_length])
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
            x_slice_list = [cnn_proba[i]]
            y_slice_list = [Y_Test[i]]
            tmp_count = 2
            tmp_traj_label = traj_id
    else:
        if traj_id == tmp_traj_label:
            tmp_count += 1
            x_slice_list.append(cnn_proba[i])
            y_slice_list.append(Y_Test[i])
        else:
            trailid_iflargethanslice_flag = False
            x_slice_list = []
            y_slice_list = []
            for k in range(slice_length):
                print(Xtrailid[randomlist[i+k-slice_length]],traj_id,k,tmp_traj_label)
                if Xtrailid[randomlist[i+k-slice_length]] == traj_id:
                    x_slice_list.append(cnn_proba[i+k-slice_length])
                    y_slice_list.append(Y_Test[i+k-slice_length])
                else:
                    print("duankai")
                    trailid_iflargethanslice_flag = True
                    break
            x_slice_list = [cnn_proba[i]]
            y_slice_list = [Y_Test[i]]
            tmp_count = 2
            tmp_traj_label = traj_id
            if trailid_iflargethanslice_flag == True:
                continue
            tmp_X += x_slice_list
            tmp_Y += y_slice_list
'''
LEN = len(tmp_X)
X, Y = np.array(tmp_X), np.array(tmp_Y)
print(X.shape,Y.shape,LEN)
X = X.reshape((round(LEN/slice_length), slice_length, lo_size*la_size))
Y = Y.reshape((round(LEN/slice_length), slice_length, 2))
print(X.shape,Y.shape,LEN)
'''
Y = Y.reshape((round(LEN/slice_length)*slice_length, 2))
tempy = []
for i in range(Y.shape[0]):
    tempy.append(coordinate_to_label(Y[i]))
Y = np.array(tempy).reshape((round(LEN/slice_length), slice_length, 832))
'''
lstmep = 20
lstmbatch_size = 6
print(Y.shape)
X_Train_LSTM,Y_Train_LSTM,X_Test_LSTM,Y_test_LSTM,randomlistLSTM = shuffle_train_testlstm(X,Y,LEN=len(X),rate=0.1)
adam_lstm = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
model_lstm = Sequential()
model_lstm.add(LSTM(slice_length*20, input_shape=(X_Train_LSTM.shape[1], X_Train_LSTM.shape[2]), return_sequences=True))
model_lstm.add(TimeDistributed(Dense(2*la_size*lo_size, activation='relu')))
model_lstm.add(TimeDistributed(Dense(la_size*lo_size, activation='softmax')))
model_lstm.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam_lstm)
model_lstm.fit(X_Train_LSTM, Y_Train_LSTM, epochs=lstmep, batch_size=lstmbatch_size)
lstm_result = model_lstm.predict(X_Test_LSTM)
print(lstm_result.shape)
lstm_result = lstm_result.reshape(Y_test_LSTM.shape[0]*slice_length, lo_size*la_size)
Y_test_LSTM = Y_test_LSTM.reshape(Y_test_LSTM.shape[0]*slice_length, 2)

print(Y_test_LSTM.shape)
error_list = []
coordinate_result = []
for i in range(Y_test_LSTM.shape[0]):
    tmp = label_to_coordinate2(lstm_result[i])
    coordinate_result.append(tmp)
    error = calcu_distance(Y_test_LSTM[i][0], Y_test_LSTM[i][1], tmp[0], tmp[1])
    error_list.append(error)

coordinate_result = np.array(coordinate_result)
print(coordinate_result.shape)
print(Y_test_LSTM.shape)
print(np.median(error_list)) # 这就是中位误差？
print(np.mean(error_list)) # 这就是中位误差？

plt.figure()
plt.scatter(Y_test_LSTM[:,0], Y_test_LSTM[:,1], c='blue', s=5)
plt.scatter(coordinate_result[:,0], coordinate_result[:,1], c='red', s=3)

plt.show()


# cnn_result.reshape(())
# X_CNN = np.delete(X, [0], axis=1) # CNN训练时，不需要时间戳
# X_CNN = X_CNN.reshape((round(LEN/slice_length), slice_length, 6, 5))
# X_CNN = X_CNN[:, :, :,: , np.newaxis]
# Y = Y.reshape((round(LEN/slice_length), slice_length, 2))

#切割
# for i in range(LEN):
#     traj_id = int(X[i][1])
#     if tmp_count == slice_length:
#         if traj_id == tmp_traj_label:
#             x_slice_list.append(X[i])
#             y_slice_list.append(Y[i])
#             tmp_X += x_slice_list
#             tmp_Y += y_slice_list
#             x_slice_list = []
#             y_slice_list = []
#             tmp_count = 1
#             tmp_traj_label = int(X[i+1][1])
#         else:
#             x_slice_list.insert(0, X[i-slice_length])
#             y_slice_list.insert(0, Y[i-slice_length])
#             tmp_X += x_slice_list
#             tmp_Y += y_slice_list
#             x_slice_list = [X[i]]
#             y_slice_list = [Y[i]]
#             tmp_count = 2
#             tmp_traj_label = traj_id
#     else:
#         if traj_id == tmp_traj_label:
#             tmp_count += 1
#             x_slice_list.append(X[i])
#             y_slice_list.append(Y[i])
#         else:
#             x_slice_list = []
#             y_slice_list = []
#             for k in range(slice_length):
#                 x_slice_list.append(X[i+k-slice_length])
#                 y_slice_list.append(Y[i+k-slice_length])
#             tmp_X += x_slice_list
#             tmp_Y += y_slice_list
#             x_slice_list = [X[i]]
#             y_slice_list = [Y[i]]
#             tmp_count = 2
#             tmp_traj_label = traj_id

#LEN = len(tmp_X)
#X, Y = np.array(tmp_X), np.array(tmp_Y)
#X = np.delete(X, [1], axis=1)

#print(X.shape,Y.shape)
# scalerX = preprocessing.StandardScaler()
# X = scalerX.fit_transform(X)
# X_CNN = np.delete(X, [0], axis=1) # CNN训练时，不需要时间戳
# X_CNN = X_CNN.reshape((round(LEN/slice_length), slice_length, 6, 5))
# X_CNN = X_CNN[:, :, :,: , np.newaxis]
# Y = Y.reshape((round(LEN/slice_length), slice_length, 2))

# X_Train, Y_Train, X_Test, Y_Test = shuffle_train_test(X_CNN, Y, round(LEN/slice_length))

# X_Train = X_Train.reshape((X_Train.shape[0]*X_Train.shape[1], 6, 5, 1))
# X_Test = X_Test.reshape((X_Test.shape[0]*X_Test.shape[1], 6, 5, 1))
# Y_Train = Y_Train.reshape((Y_Train.shape[0]*Y_Train.shape[1], la_size*lo_size))
# Y_Test = Y_Test.reshape((Y_Test.shape[0]*Y_Test.shape[1], 2))

# adam_cnn = Adam(lr=6e-4, beta_1=0.9, beta_2=0.999, epsilon=None,amsgrad=False)
# model_cnn = Sequential()

# model_cnn.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(6, 5, 1)))
# model_cnn.add(Conv2D(256, kernel_size=3, activation='relu'))
# model_cnn.add(Flatten())
# model_cnn.add(Dense(512, activation='relu'))
# model_cnn.add(Dense(256, activation='relu'))
# model_cnn.add(Dense(la_size*lo_size, activation='softmax'))

# model_cnn.compile(optimizer=adam_cnn, loss='categorical_crossentropy', metrics=['accuracy'])
# model_cnn.fit(X_Train, Y_Train, epochs=1, batch_size=64)
# cnn_result = model_cnn.predict(X_Test)
# print(cnn_result.shape) # 942, 832
# # cnn_result.reshape(())

# X = X.reshape((round(LEN/slice_length), slice_length, 31))
# X_Train, Y_Train, X_Test, Y_Test = shuffle_train_test(X, Y, round(LEN/slice_length))
# print(X_Train.shape)
# print(Y_Train.shape)


# adam_lstm = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False)
# model_lstm = Sequential()

# model_lstm.add(LSTM(slice_length*20, input_shape=(X_Train.shape[1], X_Train.shape[2]), return_sequences=True))
# model_lstm.add(TimeDistributed(Dense(2*la_size*lo_size, activation='relu')))
# model_lstm.add(TimeDistributed(Dense(la_size*lo_size, activation='softmax')))
# model_lstm.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam_lstm)
# model_lstm.fit(X_Train, Y_Train, epochs=1, batch_size=6)
# lstm_result = model_lstm.predict(X_Test)
# print(lstm_result.shape)

# def label_to_coordinate(label_list): # error!
#     label = np.argmax(label_list)
#     la_label = label % la_size
#     lo_label = (label - la_label) / la_size
#     return ((0.5+la_label)*la + min_latitude, (0.5+lo_label)*lo + min_longitude)

# def calcu_distance(true_latitude, true_longitude, pred_latitude, pred_longitude):
#     vector1 = np.array([true_latitude, true_longitude])
#     vector2 = np.array([pred_latitude, pred_longitude])
#     return np.sqrt(np.sum(np.square(vector1 - vector2)))

# error_list = []
# coordinate_result = []
# print(result.shape[0])
# for i in range(result.shape[0]):
#     tmp = label_to_coordinate(result[i])
#     coordinate_result.append(tmp)
#     error = calcu_distance(Y_Test[i][0], Y_Test[i][1], tmp[0], tmp[1])
#     error_list.append(error)

# coordinate_result = np.array(coordinate_result)
# print(np.median(error_list)) # 这就是中位误差？
# print(np.mean(error_list)) # 这就是中位误差？
# # print(coordinate_result.shape)
# plt.figure()
# plt.scatter(Y_Test[:,0], Y_Test[:,1], c='blue', s=5)
# plt.scatter(coordinate_result[:,0], coordinate_result[:,1], c='red', s=3)

# plt.show()
