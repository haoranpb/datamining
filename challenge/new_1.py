import json
import numpy as np
import string
import random
import operator
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
from sklearn.metrics import roc_auc_score
'''change'''
np.random.seed(5)#随机值种子设定为5

'''change'''
data = []



#with open('./modcloth_final_data.json','r') as file:
with open('./data/modcloth_final_data.json','r') as file:
    for line in file:
        data.append(json.loads(line))

random.seed(1)
random.shuffle(data)#打乱元素

train_data = data[:int(0.9*len(data))]#前90%的数据用于train
test_data = data[int(0.9*len(data)):]#后10%用于测试

print(len(train_data), len(test_data))

#以下是优化方法
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
    '''记录全部size'''
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
    '''
    同时按照userid分类，user也按照类似处理方式进行处理，label换成userid
    '''
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
    cloth_size[k] = list(sorted(set(cloth_size[k])))#对每个商品的全部size去重从小到大排序


cloth_smaller = {}
cloth_larger = {}

for k in cloth_size:
    #对于全部的商品
    for i in range(len(cloth_size[k])):
        #对于每一个size
        if len(cloth_size[k]) == 1:#如果就一行，那smaller和larger里面对应的商品id|尺寸的第一次被读取的顺序的label的值都 = -1
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1

        elif i == 0:#对于第一个值（最小值），smaller里面id|size第一被读取的值 = -1 larger对应值 = 该商品+尺寸更大的size第一次被读取的顺序（也是在data中该商品+size第一次出现的位置
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i+1])]
        elif i == len(cloth_size[k]) - 1:#对于最后一个值（最大值），larger里面id|size第一次被读取的值 = -1 smaller对应值 = 比该商品+尺寸小的尺寸第一次被读取的数据（也是在data中该商品+size第一次出现的位置
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i-1])]
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = -1
        else:#
            cloth_smaller[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i-1])]
            cloth_larger[cloth_order[k + '|' + str(cloth_size[k][i])]] = cloth_order[k + '|' + str(cloth_size[k][i+1])]

#smaller中存的是每个index对应比自己小的index，larger对应的是比自己大的larger，最大或最小都 = -1

'''将s，t对应位置相乘所得向量与a，bu，bi三个权值合并，再与w做点乘'''
def ftscore(a,b_i,b_u,s,t):
    return np.dot(w, np.concatenate(([a, b_u, b_i], np.multiply(s, t))))

K = 10
alpha = 1
viat = np.random.normal(size = (len(cloth_order), K))*0.1#高斯分布，返回array的size为[商品id+size的总数，K(10)]
viac = np.random.normal(size = (len(customer_order), K), loc=1, scale=0.1)#高斯分布，返回array的size为[user的总数，K(10)]
cloth_bias = np.random.normal(size = (len(cloth_order)))*0.1#第一个偏差值为高斯分布，长度为商品id+size的总数，结果与0.1相乘
customer_bias = np.random.normal(size = (len(customer_order)))*0.1#第二个偏差值为高斯分布，长度为user的总数，结果与0.1相乘
b_1 = -5
b_2 = 5
lamda = 2
w = np.ones(K+3)#其他变量，其中w是长度为13的全1矩阵，lamda影响后面的学习效率
'''修改viat的值，对于cloth_size里面的每个商品id+size，viat对应的indexlabel的值赋值为size为(1,10)的高斯分布，分布的均值是当前id总size数目 - 当前size的顺序'''
for k in cloth_size:
    start = len(cloth_size[k])
    for size in cloth_size[k]:
        viat[cloth_order[k + '|' + str(size)]] = np.random.normal(size=(1,K), loc=start, scale=0.1)
        start -= 1

#目的：通过其他user对一个item的一个确定size的评论，来确定其他user对该size的评论是fit/small/large

def features_extraction(data):
    X = []
    Y = []
    Y_auc = []
    for lines in data:
        features = []
        features.append(w[1]*customer_bias[customer_order[lines['user_id']]])
        features.append(w[2]*cloth_bias[parent_cloth_order[lines['item_id']]])
        features.extend(np.multiply(w[3:], np.multiply(viac[customer_order[lines['user_id']]], viat[cloth_order[lines['item_id'] + '|' + str(lines['size'])]])))
        X.append(features)
        #onehot
        if 'small' in lines['fit']:
            Y.append(0)
            Y_auc.append([1,0,0])
        elif 'fit' in lines['fit']:
            Y.append(1)
            Y_auc.append([0,1,0])
        else:
            Y.append(2)
            Y_auc.append([0,0,1])
    return np.array(X),np.array(Y), np.array(Y_auc)

def select_prototype(data):#从data中筛选一些值出来
    chosen_data = []
    X_small,_,_ = features_extraction(data)#提取的fe
    small_mean = np.mean(X_small, axis = 0)
    dist = []
    for i in range(len(data)):#fe里面每类数据的方差，包括2者偏差与高斯分布乘积
        dist.append((i, sum([(X_small[i][j]-small_mean[j])**2 for j in range(len(small_mean))])))
    #判断该组数据类别
    if 'fit' in data[0]['fit']:#small_temp
        small_temp = sorted(dist, key=operator.itemgetter(1))[1300:]#对方差排序
        interval = int(len(data)/5000)
        for i in range(100):
            chosen_data.append(data[small_temp[int(i*interval)][0]])#以400为step，从data/5000位置开始添加data第x个的值
            interval += 4
    elif 'small' in data[0]['fit']:
        small_temp = sorted(dist, key=operator.itemgetter(1))[1200:]
        #print(len(small_temp))
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
            interval += 0.5#step为50
    return chosen_data

def LMNNcal():#LMNN fit 并计算AUC
    small = []
    fit = []
    large= []
    #3个矩阵分别存储对应评价的条目
    for i in range(len(train_data)):
        if 'small' in train_data[i]['fit']:
            small.append(train_data[i])
        elif 'fit' in train_data[i]['fit']:
            fit.append(train_data[i])
        else:
            large.append(train_data[i])
    #打乱
    random.shuffle(small); random.shuffle(fit); random.shuffle(large)
    data_training = []
    data_training.extend(select_prototype(small))
    data_training.extend(select_prototype(fit))
    data_training.extend(select_prototype(large))
    #训练数据拿出这些值
    random.shuffle(data_training)
    #给训练数据打乱
    X_train, Y_train, Y_train_auc = features_extraction(data_training)
    #处理训练数据
    X_test, Y_test, Y_test_auc = features_extraction(test_data)
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

def calculoss():#损失函数，用文中提到的方法计算，铰链loss
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

for iterr in range(0, 1000):#发现450次的时候loss还在下降，没有收敛，因此增加到500
    learning_rate1 = 0.00025
    learning_rate2 = 0.00001/np.sqrt(iterr+1)
    #学习率
    innerb1 = 0
    innerb2 = 0#分别存储结果使用b1，b2的次数
    inners = np.zeros((len(customer_order), K))
    innert = np.zeros((len(cloth_order), K))
    innerbu = np.zeros(len(customer_order))
    innerbi = np.zeros(len(parent_cloth_order))
    innerbs = np.zeros(len(sizes))
    innerw = np.zeros((K+3))
    for lines in train_data:
        s = viac[customer_order[lines['user_id']]]#
        t = viat[cloth_order[lines['item_id'] + '|' + str(lines['size'])]]#
        b_i = cloth_bias[parent_cloth_order[lines['item_id']]]#
        b_u = customer_bias[customer_order[lines['user_id']]]#
        #变量初始化
        #下面开始计算损失函数
        # 文中提到的loss计算方式，当计算LOSS>0时，才会根据学习率进行对变量的更新操作，以此来缩小loss
        if 'small' in lines['fit']:
            #如果是small的情况
            A = 1 - ftscore(alpha,b_i,b_u,s,t) + b_2
            if A>0:
                inners[customer_order[lines['user_id']]] += -1*np.multiply(w[3:],t)
                innert[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += -1*np.multiply(w[3:],s)
                innerbu[customer_order[lines['user_id']]] += -1*w[1]
                innerbi[parent_cloth_order[lines['item_id']]] += -1*w[2]
                innerw += -1*np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                innerb2 += 1
        elif 'fit' in lines['fit']:
            #fit有两种情况
            B = 1 + ftscore(alpha,b_i,b_u,s,t) - b_2
            C = 1 - ftscore(alpha,b_i,b_u,s,t) + b_1
            if B>0:
                inners[customer_order[lines['user_id']]] += np.multiply(w[3:],t)
                innert[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += np.multiply(w[3:],s)
                innerbu[customer_order[lines['user_id']]] += w[1]
                innerbi[parent_cloth_order[lines['item_id']]] += w[2]
                innerw += np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                innerb2 += -1
            if C>0:
                inners[customer_order[lines['user_id']]] += -1*np.multiply(w[3:],t)
                innert[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += -1*np.multiply(w[3:],s)
                innerbu[customer_order[lines['user_id']]] += -1*w[1]
                innerbi[parent_cloth_order[lines['item_id']]] += -1*w[2]
                innerw += -1*np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                innerb1 += 1
        elif 'large' in lines['fit']:
            D = 1 + ftscore(alpha,b_i,b_u,s,t) - b_1
            if D>0:
                inners[customer_order[lines['user_id']]] += np.multiply(w[3:],t)
                innert[cloth_order[lines['item_id'] + '|' + str(lines['size'])]] += np.multiply(w[3:],s)
                innerbu[customer_order[lines['user_id']]] += w[1]
                innerbi[parent_cloth_order[lines['item_id']]] += w[2]
                innerw += np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                innerb1 += -1
    # 对于商品，训练更新item和对应size的bias矩阵数据和vias数据
    for i in range(len(cloth_order)):
        ## constraint update
        #比较优化后的viat[i]与比他更大/更小的被读取的顺序，取较小/较大者作为结果更新viat[i]的值
        temp = viat[i] - learning_rate1*(innert[i] + 2*lamda*viat[i])
        if cloth_smaller[i] != -1:
            temp = np.maximum(temp, viat[cloth_smaller[i]])
        if cloth_larger[i] != -1:
            temp = np.minimum(temp, viat[cloth_larger[i]])
        viat[i] = temp
    for i in range(len(parent_cloth_order)):
        cloth_bias[i] -= learning_rate1*(innerbi[i] + 2*lamda*cloth_bias[i])    
    #对于userindex，训练更新users的bias矩阵数据和vias数据
    for i in range(len(customer_order)):
        viac[i] -= learning_rate1*(inners[i] + 2*lamda*viac[i])
        customer_bias[i] -= learning_rate1*(innerbu[i] + 2*lamda*customer_bias[i])

    
    b_1 -= learning_rate2*(innerb1 + 2*lamda*b_1)
    b_2 -= learning_rate2*(innerb2 + 2*lamda*b_2)
    w -= learning_rate2*(innerw + 2*lamda*w)#更新w
    if iterr%5 == 0:
        '''计算损失函数，用文中提到的方法计算，锁链loss？'''
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
        print(iterr, finalloss, b_1, b_2,w[:4])
        if iterr % 20 == 0:
            LMNNcal()

# LMNNcal()

# 总体模型是一个两层训练，第一层训练是针对用户偏差，商品偏差，w，alpha等参数的训练，然后在这些参数训练的基础上，对第二层参数进行继续训练
# 第二层严格意义上不是一个训练过程，他不改变任何存储在当前系统中的变量，仅仅是根据现有变量，使用LR/LMNN模型fit来得出结果，并进行预测，而在循环中，他每一次的fit都是不存储变量的
# 因此从宏观上讲，第二层我认为不是一个训练过程

# 调参来看我认为有两种方式，一种是更改lr这种训练过程中不变的值，即通常意义上的调参，这里可以调参：lr、训练次数、LMNN方法计算AUC中的取值位置
# 还有一种办法是，最后将w，alpha等值输出一下，然后直接在初始化的时候将他初始化为上一次训练的最终值，相当于实现了模型存储（其实不算是调参

# 还有一个很有意思的现象，用LR训练效果好于LMNN，LMNN训练最后结果输出达不到该github上显示的0.7，而是和最后一次输出一样大概在0.68左右

#简单记录一下，对于modcloth，使用当前模型，LR运行结果为0.59 LMNN结果为0.612

#训练500次，loss都无法收敛还在减小，可以考虑提高laerning rate