import json
import numpy as np
import operator
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
from sklearn.metrics import roc_auc_score


data = []
# with open('./data/modcloth_final_data.json','r') as file:
with open('./data/renttherunway_final_data.json','r') as file:
    for line in file:
        data.append(json.loads(line))

np.random.seed(10) # 随机值种子设定为5
np.random.shuffle(data) # 打乱元素
train_data = data[:int(0.9*len(data))] # 前90%的数据用于train
test_data = data[int(0.9*len(data)):] # 后10%用于测试

cloth_data = {}
customer_data = {}
cloth_order = {}
customer_order = {}
parent_cloth_order = {} 
customer_counter = 0
cloth_counter = 0
parent_cloth_counter = 0
sizes = {}
size_counter = 0

for lines in data:
    # 记录全部size
    if lines['size'] not in sizes:
        sizes[lines['size']] = size_counter
        size_counter += 1
    # 对于训练数据中的每一行数据，按照 id|size 的label，存储在itemdata字典中，label是id|size，值是对应的一行数据，同时将该商品第一次被读取的顺序用相同的label
    # 即 id | size ，存储在index字典中
    if lines['item_id'] + '|' + str(lines['size']) not in cloth_data:
        cloth_data[lines['item_id'] + '|' + str(lines['size'])] = [lines]
        cloth_order[lines['item_id'] + '|' + str(lines['size'])] = cloth_counter
        cloth_counter += 1
    else:
        cloth_data[lines['item_id'] + '|' + str(lines['size'])].append(lines)
        
    if lines['item_id'] not in parent_cloth_order:
        parent_cloth_order[lines['item_id']] = parent_cloth_counter
        parent_cloth_counter += 1
    # 同时按照userid分类，user也按照类似处理方式进行处理，label换成userid
    if lines['user_id'] not in customer_data:
        customer_data[lines['user_id']] = [lines]
        customer_order[lines['user_id']] = customer_counter
        customer_counter += 1
    else:
        customer_data[lines['user_id']].append(lines)

cloth_size = {}
#对于每一个商品，记录其全部的size
for lines in data:
    if lines['item_id'] not in cloth_size:
        cloth_size[lines['item_id']] = [lines['size']]
    else:
        cloth_size[lines['item_id']].append(lines['size'])

#对该dict的任一
for k in cloth_size:
    cloth_size[k] = list(sorted(set(cloth_size[k]))) # 对每个商品的全部size去重从小到大排序

cloth_smaller = {}
cloth_larger = {}

for k in cloth_size:
    # 对于全部的商品
    for i in range(len(cloth_size[k])):
        # 对于每一个size
        if len(cloth_size[k]) == 1: # 如果就一行，那smaller和larger里面对应的商品id|尺寸的第一次被读取的顺序的label的值都 = -1
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1
        # 对于第一个值（最小值），smaller里面id|size第一被读取的值 = -1 larger对应值 = 该商品+尺寸更大的size第一次被读取的顺序（也是在data中该商品+size第一次出现的位置
        elif i == 0:
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i+1])]
        # 对于最后一个值（最大值），larger里面id|size第一次被读取的值 = -1 smaller对应值 = 比该商品+尺寸小的尺寸第一次被读取的数据（也是在data中该商品+size第一次出现的位置
        elif i == len(cloth_size[k]) - 1:
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i-1])]
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1
        else:
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i-1])]
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i+1])]

# smaller中存的是每个index对应比自己小的index，larger对应的是比自己大的larger，最大或最小都 = -1
# 将s，t对应位置相乘所得向量与a，bu，bi三个权值合并，再与w做点乘
def ftscore(a, b_i, b_u, s, t):
    return np.dot(w, np.concatenate(([a, b_u, b_i], np.multiply(s, t))))

K = 10
alpha = 1
viat = np.random.normal(size = (len(cloth_order), K)) * 0.1 # 高斯分布，返回array的size为[商品id+size的总数，K(10)]
viac = np.random.normal(size = (len(customer_order), K), loc=1, scale=0.1) # 高斯分布，返回array的size为[user的总数，K(10)]
cloth_bias = np.random.normal(size = (len(cloth_order))) * 0.1 # 第一个偏差值为高斯分布，长度为商品id+size的总数，结果与0.1相乘
customer_bias = np.random.normal(size = (len(customer_order))) * 0.1 # 第二个偏差值为高斯分布，长度为user的总数，结果与0.1相乘
b_1 = -5
b_2 = 5
lamda = 2
w = np.ones(K+3) # 其他变量，其中w是长度为13的全1矩阵，lamda影响后面的学习效率
# 修改viat的值，对于cloth_size里面的每个商品id+size，viat对应的indexlabel的值赋值为size为(1,10)的高斯分布，分布的均值是当前id总size数目 - 当前size的顺序
for k in cloth_size:
    start = len(cloth_size[k])
    for size in cloth_size[k]:
        viat[cloth_order[k + '|' + str(size)]] = np.random.normal(size=(1, K), loc=start, scale=0.1)
        start -= 1

# 体征提取
def features_extraction(data):
    X = []
    Y = []
    Y_auc = []
    for lines in data:
        features = []
        features.append(w[1] * customer_bias[customer_order[lines['user_id']]])
        features.append(w[2] * cloth_bias[parent_cloth_order[lines['item_id']]])
        features.extend(np.multiply(w[3:], np.multiply(viac[customer_order[lines['user_id']]], viat[cloth_order[lines['item_id'] + '|' + str(lines['size'])]])))
        X.append(features)
        # onehot
        if 'small' in lines['fit']:
            Y.append(0)
            Y_auc.append([1, 0, 0])
        elif 'fit' in lines['fit']:
            Y.append(1)
            Y_auc.append([0, 1, 0])
        else:
            Y.append(2)
            Y_auc.append([0, 0, 1])
    return np.array(X),np.array(Y), np.array(Y_auc)

# 从不同类别的data中筛选一些值出来
def select_prototype(data):
    chosen_data = []
    X_small,_,_ = features_extraction(data) # 提取的特征
    small_mean = np.mean(X_small, axis = 0)
    dist = []
    for i in range(len(data)): # features里面每类数据的方差，包括2者偏差与高斯分布乘积
        dist.append((i, sum([(X_small[i][j]-small_mean[j])**2 for j in range(len(small_mean))])))
    # 判断该组数据类别
    if 'fit' in data[0]['fit']: # small_temp
        small_temp = sorted(dist, key=operator.itemgetter(1))[1300:] # 对方差排序
        interval = int(len(data)/5000)
        for i in range(100):
            chosen_data.append(data[small_temp[int(i*interval)][0]]) # 以400为step，从data/5000位置开始添加data第x个的值
            interval += 4
    elif 'small' in data[0]['fit']:
        small_temp = sorted(dist, key=operator.itemgetter(1))[1200:]
        # print(len(small_temp))
        interval = int(len(data)/5000)
        # interval = int(len(data)/1000)
        for i in range(100):
            if int(i*interval) > len(small_temp):
                print(i,interval,int(i*interval),len(small_temp))
            chosen_data.append(data[small_temp[int(i*interval)][0]])
            interval += 0.8  #这里改为modcloth数据还保持为1.15的话会下标越界，我估计是因为small凑不够11500个,事实上确实如此，按照546行的输出，
            # 发现仅有9258条，循环100次的话，从66232/1000开始，即66，每次增加1.15,换句话说从6600开始，很快就超了，所以这里要缩小初始值，同时也要缩小step
            #interval += 1.15#step为115
    else:
        small_temp = sorted(dist, key=operator.itemgetter(1))[1200:]
        interval = int(len(data)/500)
        for i in range(100):
            chosen_data.append(data[small_temp[int(i*interval)][0]])
            interval += 0.5 # step为50
    return chosen_data

# LMNN fit 并计算AUC
def LMNNcal():
    small = []
    fit = []
    large= []
    # 3个矩阵分别存储对应评价的条目
    for lines in range(len(train_data)):
        if 'small' in train_data[lines]['fit']:
            small.append(train_data[lines])
        elif 'fit' in train_data[lines]['fit']:
            fit.append(train_data[lines])
        else:
            large.append(train_data[lines])
    # 打乱
    np.random.shuffle(small)
    np.random.shuffle(fit)
    np.random.shuffle(large)
    data_training = []
    data_training.extend(select_prototype(small))
    data_training.extend(select_prototype(fit))
    data_training.extend(select_prototype(large))
    #训练数据拿出这些值
    np.random.shuffle(data_training)
    #给训练数据打乱
    X_train, Y_train, _ = features_extraction(data_training)
    #处理训练数据
    X_test, _, Y_test_auc = features_extraction(test_data)
    #处理测试数据
    #LMNN
    clf_kLF = LMNN(n_neighbors=53, max_iter=50, n_features_out=X_train.shape[1], verbose=0)
    clf_kLF = clf_kLF.fit(X_train, Y_train)
    pred = clf_kLF.predict_proba(X_test)
    AUC = []
    for i in range(3):
        AUC.append(roc_auc_score(Y_test_auc[:,i], pred[:,i], average='weighted'))
    print('LMNN Average AUC for 3 fits', np.mean(AUC),'and Every AUC',  AUC)

    return np.mean(AUC)

# 损失函数，用文中提到的方法计算，铰链loss
def calculoss():
    finalloss = 0
    for item in cloth_data:
        for lines in cloth_data[item]:
            s = viac[customer_order[lines['user_id']]]
            t = viat[cloth_order[lines['item_id'] + '|' + str(lines['size'])]]
            b_i = cloth_bias[parent_cloth_order[lines['item_id']]]
            b_u = customer_bias[customer_order[lines['user_id']]]

            if 'large' in lines['fit']:
                finalloss += max(0, 1 + ftscore(alpha, b_i, b_u, s, t) - b_1)
            else:
                if 'small' in lines['fit']:
                    finalloss += max(0, 1 - ftscore(alpha,b_i,b_u,s,t) + b_2)
                elif 'fit' in lines['fit']:
                    finalloss += max(0, 1 + ftscore(alpha, b_i, b_u, s, t) - b_2)
                    finalloss += max(0, 1 - ftscore(alpha, b_i, b_u, s, t) + b_1)
    return finalloss

# 发现450次的时候loss还在下降，没有收敛，因此增加到500
for iterr in range(0, 500):
    learning_rate1 = 0.00025
    learning_rate2 = 0.00001/np.sqrt(iterr+1)
    # 学习率
    Gradient_of_b1 = 0
    Gradient_of_b2 = 0 # 分别存储结果使用b1，b2的次数
    Gradient_of_s = np.zeros((len(customer_order), K))
    Gradient_of_t = np.zeros((len(cloth_order), K))
    Gradient_of_bu = np.zeros(len(customer_order))
    Gradient_of_bi = np.zeros(len(parent_cloth_order))
    Gradient_of_bs = np.zeros(len(sizes))
    Gradient_of_w = np.zeros((K + 3))

    for lines in train_data:
        s = viac[customer_order[lines['user_id']]]
        t = viat[cloth_order[lines['item_id'] + '|' + str(lines['size'])]]
        b_i = cloth_bias[parent_cloth_order[lines['item_id']]]
        b_u = customer_bias[customer_order[lines['user_id']]]
        #变量初始化
        #下面开始计算损失函数
        # 文中提到的loss计算方式，当计算LOSS>0时，才会根据学习率进行对变量的更新操作，以此来缩小loss
        if 'small' in lines['fit']:
            #如果是small的情况
            small_loss = 1 - ftscore(alpha, b_i, b_u, s, t) + b_2
            if small_loss > 0:
                Gradient_of_s[customer_order[lines['user_id']]] += -1*np.multiply(w[3:], t)
                Gradient_of_t[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += -1*np.multiply(w[3:], s)
                Gradient_of_bu[customer_order[lines['user_id']]] += -1*w[1]
                Gradient_of_bi[parent_cloth_order[lines['item_id']]] += -1*w[2]
                Gradient_of_w += -1*np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                Gradient_of_b2 += 1
        elif 'fit' in lines['fit']:
            #fit有两种情况
            fit_loss_1 = 1 + ftscore(alpha, b_i, b_u, s, t) - b_2
            fit_loss_2 = 1 - ftscore(alpha, b_i, b_u, s, t) + b_1
            if fit_loss_1 > 0:
                Gradient_of_s[customer_order[lines['user_id']]] += np.multiply(w[3:], t)
                Gradient_of_t[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += np.multiply(w[3:], s)
                Gradient_of_bu[customer_order[lines['user_id']]] += w[1]
                Gradient_of_bi[parent_cloth_order[lines['item_id']]] += w[2]
                Gradient_of_w += np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                Gradient_of_b2 += -1
            if fit_loss_2 > 0:
                Gradient_of_s[customer_order[lines['user_id']]] += -1*np.multiply(w[3:], t)
                Gradient_of_t[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += -1*np.multiply(w[3:], s)
                Gradient_of_bu[customer_order[lines['user_id']]] += -1*w[1]
                Gradient_of_bi[parent_cloth_order[lines['item_id']]] += -1*w[2]
                Gradient_of_w += -1*np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                Gradient_of_b1 += 1
        elif 'large' in lines['fit']:
            large_loss = 1 + ftscore(alpha, b_i, b_u, s, t) - b_1
            if large_loss > 0:
                Gradient_of_s[customer_order[lines['user_id']]] += np.multiply(w[3:], t)
                Gradient_of_t[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += np.multiply(w[3:], s)
                Gradient_of_bu[customer_order[lines['user_id']]] += w[1]
                Gradient_of_bi[parent_cloth_order[lines['item_id']]] += w[2]
                Gradient_of_w += np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                Gradient_of_b1 += -1
    # 对于商品，训练更新item和对应size的bias矩阵数据和vias数据
    for i in range(len(cloth_order)):
        ## constraint update
        # 比较优化后的viat[i]与比他更大/更小的被读取的顺序，取较小/较大者作为结果更新viat[i]的值
        temp = viat[i] - learning_rate1*(Gradient_of_t[i] + 2*lamda*viat[i])
        if cloth_smaller[i] != -1:
            temp = np.maximum(temp, viat[cloth_smaller[i]])
        if cloth_larger[i] != -1:
            temp = np.minimum(temp, viat[cloth_larger[i]])
        viat[i] = temp
    for i in range(len(parent_cloth_order)):
        cloth_bias[i] -= learning_rate1*(Gradient_of_bi[i] + 2*lamda*cloth_bias[i])
    # 对于userindex，训练更新users的bias矩阵数据和vias数据
    for i in range(len(customer_order)):
        viac[i] -= learning_rate1*(Gradient_of_s[i] + 2*lamda*viac[i])
        customer_bias[i] -= learning_rate1*(Gradient_of_bu[i] + 2*lamda*customer_bias[i])

    
    b_1 -= learning_rate2*(Gradient_of_b1 + 2*lamda*b_1)
    b_2 -= learning_rate2*(Gradient_of_b2 + 2*lamda*b_2)
    w -= learning_rate2*(Gradient_of_w + 2*lamda*w) #更新w
    if iterr % 10 == 0:
        # 计算损失函数，用文中提到的方法计算，锁链loss
        finalloss = 0
        for item in cloth_data:
            for lines in cloth_data[item]:
                s = viac[customer_order[lines['user_id']]]
                t = viat[cloth_order[lines['item_id'] + '|' + str(lines['size'])]]
                b_i = cloth_bias[parent_cloth_order[lines['item_id']]]
                b_u = customer_bias[customer_order[lines['user_id']]]
                if 'large' in lines['fit']:
                    finalloss += max(0, 1 + ftscore(alpha, b_i, b_u, s, t) - b_1)
                else:
                    if 'small' in lines['fit']:
                        finalloss += max(0, 1 - ftscore(alpha, b_i, b_u, s, t) + b_2)
                    elif 'fit' in lines['fit']:
                        finalloss += max(0, 1 + ftscore(alpha, b_i, b_u, s, t) - b_2)
                        finalloss += max(0, 1 - ftscore(alpha, b_i, b_u, s, t) + b_1)
        print('迭代次数：', iterr, '，铰链损失：', finalloss)
        LMNNcal()