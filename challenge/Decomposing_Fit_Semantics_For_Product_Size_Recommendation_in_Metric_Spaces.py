
import json
import gzip
import numpy as np
import string
import random
import operator
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
from collections import defaultdict

def parseData(file):#分批次返回文件每一行内容
    for l in open(file,'r'):
        yield json.loads(l)
        
def remove_punctuation(text):#移除标点符号
    return ''.join([c.lower() for c in text if c not in set(string.punctuation)])

'''change'''
np.random.seed(5)#随机值种子设定为5

#data = list(parseData('./renttherunway_final_data.json'))#加载数据
data = list(parseData('./modcloth_final_data.json'))

'''change'''
random.seed(1)
random.shuffle(data)#打乱元素

train_data = data[:int(0.8*len(data))]#前80%的数据用于train
val_data = data[int(0.8*len(data)):int(0.9*len(data))]#0.8~0.9用于验证
test_data = data[int(0.9*len(data)):]#后10%用于测试

print(len(train_data), len(val_data), len(test_data))


'''

item_data = {}
item_index = {}
user_index = {}
user_data = {}
u_index = 0
i_index = 0

for r in train_data:
    if r['item_id'] + '|' + str(r['size']) not in item_data:
        item_data[r['item_id'] + '|' + str(r['size'])] = [r]
        item_index[r['item_id'] + '|' + str(r['size'])] = i_index
        i_index += 1
    else:
        item_data[r['item_id'] + '|' + str(r['size'])].append(r)

    if r['user_id'] not in user_data:
        user_data[r['user_id']] = [r]
        user_index[r['user_id']] = u_index
        u_index += 1
    else:
        user_data[r['user_id']].append(r)





print(len(user_data), len(user_index), len(item_data), len(item_index))



true_size_item = np.zeros(len(item_data))#child商品数量
true_size_cust = np.zeros(len(user_data))#user数量
w = 1; b_1 = -1; b_2 = 1; lamda = 2#参数初始值


for item in item_data:
    true_size_item[item_index[item]] = int(item.split('|')[1])




from sklearn.linear_model import LogisticRegression
def calc_auc():#计算AUC
    train_features = []
    train_labels = []
    for r in train_data:
        fe = []
        fe.append(true_size_cust[user_index[r['user_id']]])
        fe.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]])
        train_features.append(fe)

        if 'small' in r['fit']:
            train_labels.append(0)
        elif 'fit' in r['fit']:
            train_labels.append(1)
        elif 'large' in r['fit']:
            train_labels.append(2)

    c = 1
    clf_1LV = LogisticRegression(fit_intercept=True, multi_class='ovr', C=c)
    clf_1LV.fit(train_features, train_labels)

    test_features = []; test_labels = []; test_labels_auc = []
    for r in test_data:
        fe = []
        try:
            fe.append(true_size_cust[user_index[r['user_id']]])
        except KeyError:
            fe.append(np.mean(true_size_cust))
        try:
            fe.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]])
        except KeyError:
            fe.append(np.mean(true_size_item))

        test_features.append(fe)
        label = [0, 0, 0]
        if 'small' in r['fit']:
            test_labels.append(0)
            label[0] = 1
        elif 'fit' in r['fit']:
            test_labels.append(1)
            label[1] = 1
        elif 'large' in r['fit']:
            test_labels.append(2)
            label[2] = 1
        test_labels_auc.append(label)

    test_labels_auc = np.array(test_labels_auc)

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    pred = clf_1LV.predict_proba(test_features)
    AUC = []
    for i in range(3):
        AUC.append(roc_auc_score(test_labels_auc[:,i], pred[:,i], average='weighted'))
    print('Average AUC', np.mean(AUC), AUC)


def f(s,t):
    return w*(s-t)

def cal_loss_user(user, cust_size):
    loss = 0
    for r in user_data[user]:
        if 'small' in r['fit']:
            loss += max(0, 1 - f(cust_size, true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]) + b_2)
        elif 'fit' in r['fit']:
            loss += max(0, 1 + f(cust_size, true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]) - b_2)
            loss += max(0, 1 - f(cust_size, true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]) + b_1)
        elif 'large' in r['fit']:
            loss += max(0, 1 + f(cust_size, true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]) - b_1)
    return loss
            
def cal_loss_item(item, product_size):
    loss = 0
    for r in item_data[item]:
        if 'small' in r['fit']:
            loss += max(0, 1 - f(true_size_cust[user_index[r['user_id']]], product_size) + b_2)
        elif 'fit' in r['fit']:
            loss += max(0, 1 + f(true_size_cust[user_index[r['user_id']]], product_size) - b_2)
            loss += max(0, 1 - f(true_size_cust[user_index[r['user_id']]], product_size) + b_1)
        elif 'large' in r['fit']:
            loss += max(0, 1 + f(true_size_cust[user_index[r['user_id']]], product_size) - b_1)
    return loss

def total_loss():
    loss = 0
    for item in item_data:
        for r in item_data[item]:
            product_size = true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]
            if 'small' in r['fit']:
                loss += max(0, 1 - f(true_size_cust[user_index[r['user_id']]], product_size) + b_2)
            elif 'fit' in r['fit']:
                loss += max(0, 1 + f(true_size_cust[user_index[r['user_id']]], product_size) - b_2)
                loss += max(0, 1 - f(true_size_cust[user_index[r['user_id']]], product_size) + b_1)
            elif 'large' in r['fit']:
                loss += max(0, 1 + f(true_size_cust[user_index[r['user_id']]], product_size) - b_1)
    return loss

for iterr in range(0,220):
    
    ## Phase 1
    for user in user_data:
        candidate_sizes = []
        for r in user_data[user]:
            if 'small' in r['fit']:
                candidate_sizes.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]] + ((b_2+1)/w))
            elif 'fit' in r['fit']:
                candidate_sizes.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]] + ((b_1+1)/w))
                candidate_sizes.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]] + ((b_2-1)/w))
            elif 'large' in r['fit']:
                candidate_sizes.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]] + ((b_1-1)/w))

        flag = 0
        candidate_sizes = list(set(candidate_sizes))
        candidate_sizes = sorted(candidate_sizes)

        if len(candidate_sizes) == 1:
            true_size_cust[user_index[user]] = candidate_sizes[0]
        else:
            for s in range(1, len(candidate_sizes)):
                slope = (cal_loss_user(user, candidate_sizes[s]) - cal_loss_user(user, candidate_sizes[s-1]))/(candidate_sizes[s] - candidate_sizes[s-1])
                if slope>=0:
                    flag=1
                    true_size_cust[user_index[user]] = candidate_sizes[s-1]
                    break

            if flag==0:
                true_size_cust[user_index[user]] = candidate_sizes[-1]

    ## Phase 2
    for item in item_data:
        candidate_sizes = []
        for r in item_data[item]:
            if 'small' in r['fit']:
                candidate_sizes.append(true_size_cust[user_index[r['user_id']]] - ((b_2+1)/w))
            elif 'fit' in r['fit']:
                candidate_sizes.append(true_size_cust[user_index[r['user_id']]] - ((b_1+1)/w))
                candidate_sizes.append(true_size_cust[user_index[r['user_id']]] - ((b_2-1)/w))
            elif 'large' in r['fit']:
                candidate_sizes.append(true_size_cust[user_index[r['user_id']]] - ((b_1-1)/w))

        flag = 0
        candidate_sizes = list(set(candidate_sizes))
        candidate_sizes = sorted(candidate_sizes)
        if len(candidate_sizes) == 1:
            true_size_item[item_index[item]] = candidate_sizes[0]
        else:
            for s in range(1, len(candidate_sizes)):
                slope = (cal_loss_item(item, candidate_sizes[s]) - cal_loss_item(item, candidate_sizes[s-1]))/(candidate_sizes[s] - candidate_sizes[s-1])
                if slope>=0:
                    flag=1
                    true_size_item[item_index[item]] = candidate_sizes[s-1]
                    break

            if flag==0:
                true_size_item[item_index[item]] = candidate_sizes[-1]

    ## Phase 3
    learning_rate = 0.0000005/np.sqrt(iterr+1)
    grad_w = 0
    grad_b1 = 0
    grad_b2 = 0
    for r in train_data:
        s = true_size_cust[user_index[r['user_id']]]
        t = true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]

        if 'small' in r['fit']:
            A = 1 - f(s, t) + b_2
            if A>0:
                grad_w += -1*(s - t)
                grad_b2 += 1
        elif 'fit' in r['fit']:
            B = 1 + f(s, t) - b_2
            C = 1 - f(s, t) + b_1
            if B>0:
                grad_w += (s - t)
                grad_b2 += -1
            if C>0:
                grad_w += -1*(s - t)
                grad_b1 += 1
        elif 'large' in r['fit']:
            D = 1 + f(s, t) - b_1
            if D>0:
                grad_w += (s - t)
                grad_b1 += -1

    w -= learning_rate*(grad_w + 2*lamda*w)
    b_1 -= learning_rate*(grad_b1 + 2*lamda*b_1)
    b_2 -= learning_rate*(grad_b2 + 2*lamda*b_2)
    if iterr%5 == 0:
        print(iterr, total_loss())
        calc_auc()





from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

train_features = []; train_labels = []
for r in train_data:
    fe = []
    fe.append(true_size_cust[user_index[r['user_id']]])
    fe.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]])
    train_features.append(fe)

    if 'small' in r['fit']:
        train_labels.append(0)
    elif 'fit' in r['fit']:
        train_labels.append(1)
    elif 'large' in r['fit']:
        train_labels.append(2)

c = 1
clf_1LV = LogisticRegression(fit_intercept=True, multi_class='ovr', C=c)
clf_1LV.fit(train_features, train_labels)

test_features = []; test_labels = []; test_labels_auc = []
for r in val_data:
    fe = []
    try:
        u = user_index[r['user_id']]
        fe.append(true_size_cust[u])
    except KeyError:
        fe.append(np.mean(true_size_cust))
    try:
        fe.append(true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]])
    except KeyError:
        fe.append(np.mean(true_size_item))

    test_features.append(fe)
    label = [0, 0, 0]
    if 'small' in r['fit']:
        test_labels.append(0)
        label[0] = 1
    elif 'fit' in r['fit']:
        test_labels.append(1)
        label[1] = 1
    elif 'large' in r['fit']:
        test_labels.append(2)
        label[2] = 1
    test_labels_auc.append(label)

test_labels_auc = np.array(test_labels_auc)

pred = clf_1LV.predict_proba(test_features)
AUC = []
for i in range(3):
    AUC.append(roc_auc_score(test_labels_auc[:,i], pred[:,i], average='weighted'))
print('Average AUC', np.mean(AUC), AUC)

'''

#以下是优化方法
item_data = {}; user_data = {}
item_index = {}; user_index = {}; p_item_index = {} ## p_ is for parent 
u_index = 0; i_index = 0; p_i_index = 0
sizes = {}
s_index = 0

for r in data:
    '''记录全部size'''
    if r['size'] not in sizes:
        sizes[r['size']] = s_index
        s_index += 1
    # 对于训练数据中的每一行数据，按照 id|size 的label，存储在itemdata字典中，label是id|size，值是对应的一行数据，同时将该商品第一次被读取的顺序用相同的label
    # 即 id | size ，存储在index字典中
    if r['item_id'] + '|' + str(r['size']) not in item_data:
        item_data[r['item_id'] + '|' + str(r['size'])] = [r]
        item_index[r['item_id'] + '|' + str(r['size'])] = i_index
        i_index += 1
    else:
        item_data[r['item_id'] + '|' + str(r['size'])].append(r)
        
    if r['item_id'] not in p_item_index:
        p_item_index[r['item_id']] = p_i_index
        p_i_index += 1
    '''
    同时按照userid分类，user也按照类似处理方式进行处理，label换成userid
    '''
    if r['user_id'] not in user_data:
        user_data[r['user_id']] = [r]
        user_index[r['user_id']] = u_index
        u_index += 1
    else:
        user_data[r['user_id']].append(r)
#len(user_data), len(user_index), len(item_data), len(item_index), len(p_item_index)



product_sizes = {}; product_sizes_rev = {}
#对于每一个商品，记录其全部的size
for r in data:
    if r['item_id'] not in product_sizes:
        product_sizes[r['item_id']] = [r['size']]
    else:
        product_sizes[r['item_id']].append(r['size'])
#对该dict的任一
for k in product_sizes:
    product_sizes[k] = list(sorted(set(product_sizes[k])))#对每个商品的全部size去重从小到大排序
    product_sizes_rev[k] = list(sorted(set(product_sizes[k]), reverse=True))#逆序


product_smaller = {}
product_larger = {}

for k in product_sizes:
    #对于全部的商品
    for i in range(len(product_sizes[k])):
        #对于每一个size
        if len(product_sizes[k]) == 1:#如果就一行，那smaller和larger里面对应的商品id|尺寸的第一次被读取的顺序的label的值都 = -1
            product_smaller[item_index[k + '|' + str(product_sizes[k][i])]] = -1
            product_larger[item_index[k + '|' + str(product_sizes[k][i])]] = -1

        elif i == 0:#对于第一个值（最小值），smaller里面id|size第一被读取的值 = -1 larger对应值 = 该商品+尺寸更大的size第一次被读取的顺序（也是在data中该商品+size第一次出现的位置
            product_smaller[item_index[k + '|' + str(product_sizes[k][i])]] = -1
            product_larger[item_index[k + '|' + str(product_sizes[k][i])]] = item_index[k + '|' + str(product_sizes[k][i+1])]
        elif i == len(product_sizes[k]) - 1:#对于最后一个值（最大值），larger里面id|size第一次被读取的值 = -1 smaller对应值 = 比该商品+尺寸小的尺寸第一次被读取的数据（也是在data中该商品+size第一次出现的位置
            product_smaller[item_index[k + '|' + str(product_sizes[k][i])]] = item_index[k + '|' + str(product_sizes[k][i-1])]
            product_larger[item_index[k + '|' + str(product_sizes[k][i])]] = -1
        else:#
            product_smaller[item_index[k + '|' + str(product_sizes[k][i])]] = item_index[k + '|' + str(product_sizes[k][i-1])]
            product_larger[item_index[k + '|' + str(product_sizes[k][i])]] = item_index[k + '|' + str(product_sizes[k][i+1])]

#smaller中存的是每个index对应比自己小的index，larger对应的是比自己大的larger，最大或最小都 = -1


'''将s，t对应位置相乘所得向量与a，bu，bi三个权值合并，再与w做点乘'''
def f(a,b_i,b_u,s,t):
    return np.dot(w, np.concatenate(([a, b_u, b_i], np.multiply(s, t))))

K = 10; learning_rate = 0.000005; alpha = 1
true_size_item = np.random.normal(size = (len(item_index), K))*0.1#高斯分布，返回array的size为[商品id+size的总数，K(10)]
true_size_cust = np.random.normal(size = (len(user_index), K), loc=1, scale=0.1)#高斯分布，返回array的size为[user的总数，K(10)]
bias_i = np.random.normal(size = (len(item_index)))*0.1#第一个偏差值为高斯分布，长度为商品id+size的总数，结果与0.1相乘
bias_u = np.random.normal(size = (len(user_index)))*0.1#第二个偏差值为高斯分布，长度为user的总数，结果与0.1相乘
b_1 = -5; b_2 = 5; lamda = 2; w = np.ones(K+3)#其他变量，其中w是长度为13的全1矩阵
'''修改truesizeitem的值，对于productsize里面的每个商品id+size，truesizeitem对应的indexlabel的值赋值为size为(1,10)的高斯分布，分布的均值是当前id总size数目 - 当前size的顺序'''
for k in product_sizes:
    start = len(product_sizes[k])
    for size in product_sizes[k]:
        true_size_item[item_index[k + '|' + str(size)]] = np.random.normal(size=(1,K), loc=start, scale=0.1)
        start -= 1


#目的：通过其他user对一个item的一个确定size的评论，来确定其他user对该size的评论是fit/small/large
def calc_auc():#LR训练，并计算AUC
    train_features = []; train_labels = []
    #
    for r in train_data:
        fe = []
        fe.append(w[1]*bias_u[user_index[r['user_id']]])#添加对应user的bias与w[1]相乘
        fe.append(w[2]*bias_i[p_item_index[r['item_id']]])#添加对应item的bias
        fe.extend(np.multiply(w[3:], np.multiply(true_size_cust[user_index[r['user_id']]], true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]])))
        #添加w后10位和（true_size_cust，true_size_item中对应的userid和itemid|size的长度为10的矩阵按元素相乘的结果）按元素相乘
        train_features.append(fe)

        #small->1 fit->2 large->3
        if 'small' in r['fit']:
            train_labels.append(1)
        elif 'fit' in r['fit']:
            train_labels.append(2)
        elif 'large' in r['fit']:
            train_labels.append(3)
    #将二者做逻辑回归，按照文中提到的K个算法
    clf = LogisticRegression(fit_intercept=True, multi_class='ovr')
    clf.fit(train_features, train_labels)

    test_features = []; test_labels = []; test_labels_auc = []
    for r in test_data:
        fe = []
        flag = 0
        fe.append(w[1]*bias_u[user_index[r['user_id']]])
        fe.append(w[2]*bias_i[p_item_index[r['item_id']]])
        fe.extend(np.multiply(w[3:], np.multiply(true_size_cust[user_index[r['user_id']]], true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]])))

        test_features.append(fe)
        label = [0, 0, 0]
        if 'small' in r['fit']:
            test_labels.append(1)
            label[0] = 1
        elif 'fit' in r['fit']:
            test_labels.append(2)
            label[1] = 1
        elif 'large' in r['fit']:
            test_labels.append(3)
            label[2] = 1
        test_labels_auc.append(label)
    #onehot处理一下testlabels
    test_labels_auc = np.array(test_labels_auc)

    from sklearn.metrics import roc_auc_score
    pred = clf.predict_proba(test_features)
    AUC = []
    for i in range(3):
        AUC.append(roc_auc_score(test_labels_auc[:,i], pred[:,i], average='weighted'))
    print('Average AUC', np.mean(AUC), AUC)




def calc_metric_auc():#LMNN fit 并计算AUC
    U = true_size_cust; V = true_size_item
    def prepare_features(data):#数据预处理
        X = []
        Y = []
        Y_auc = []
        item_l = []
        item_n = []
        items = {}
        item_count = defaultdict(int)#统计traindata中的item和size的出现总次数
        item_small = defaultdict(int)#统计对应item和size的评价中small占比
        frac_small = []#包括所有商品的small占比
        for r in data:
            fe = []
            fe.append(w[1]*bias_u[user_index[r['user_id']]])
            fe.append(w[2]*bias_i[p_item_index[r['item_id']]])
            fe.extend(np.multiply(w[3:], np.multiply(true_size_cust[user_index[r['user_id']]], true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]])))
            X.append(fe)
            item_l.append(np.multiply(w[3:],V[item_index[r['item_id'] + '|' + str(r['size'])]]))#id和size的高斯分布与后10位的乘积
            item_n.append(str(r['category']))#类别
            items[r['item_id'] + '|' + str(r['size'])] = 1
            #onehot
            if 'small' in r['fit']:
                Y.append(0)
                Y_auc.append([1,0,0])
            elif 'fit' in r['fit']:
                Y.append(1)
                Y_auc.append([0,1,0])
            else:
                Y.append(2)
                Y_auc.append([0,0,1])

        for r in train_data:
            if (r['item_id'] + '|' + str(r['size'])) in items:
                item_count[r['item_id'] + '|' + str(r['size'])] += 1
                if 'small' in r['fit']:
                    item_small[r['item_id'] + '|' + str(r['size'])] += 1
        for k in item_count:
            item_small[k] = item_small[k]/item_count[k]#计算占比

        for r in data:
            frac_small.append(item_small[r['item_id'] + '|' + str(r['size'])])

        return np.array(X),np.array(Y), np.array(Y_auc), np.array(item_l), item_n, frac_small
    
    def select_prototype(data):#从data中筛选一些值出来

        chosen_data = []
        X_small,_,_,_,_,_ = prepare_features(data)#提取的fe
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

    small = []; true = []; large= []
    #3个矩阵分别存储对应评价的条目
    for r in range(len(train_data)):
        if 'small' in train_data[r]['fit']:
            small.append(train_data[r])
        elif 'fit' in train_data[r]['fit']:
            true.append(train_data[r])
        else:
            large.append(train_data[r])
    #打乱
    random.shuffle(small); random.shuffle(true); random.shuffle(large)
    data_training = []
    data_training.extend(select_prototype(small)); data_training.extend(select_prototype(true)); data_training.extend(select_prototype(large))
    #训练数据拿出这些值
    random.shuffle(data_training)
    #给训练数据打乱
    X_train, Y_train, Y_train_auc, item_l, item_n, frac_small = prepare_features(data_training)
    #处理训练数据
    X_test, Y_test, Y_test_auc, _, _, _ = prepare_features(test_data)
    #处理测试数据
    #LMNN
    clf_kLF = LMNN(n_neighbors=53, max_iter=50, n_features_out=X_train.shape[1], verbose=0)
    clf_kLF = clf_kLF.fit(X_train, Y_train)

    from sklearn.metrics import roc_auc_score
    pred = clf_kLF.predict_proba(X_test); AUC = []
    for i in range(3):
        AUC.append(roc_auc_score(Y_test_auc[:,i], pred[:,i], average='weighted'))
    print('Average AUC', np.mean(AUC), AUC)
    return np.mean(AUC)





def total_loss():#损失函数，用文中提到的方法计算，锁链loss？
    loss = 0
    for item in item_data:
        for r in item_data[item]:
            s = true_size_cust[user_index[r['user_id']]]
            t = true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]
            b_i = bias_i[p_item_index[r['item_id']]]
            b_u = bias_u[user_index[r['user_id']]]
            
            if 'small' in r['fit']:
                loss += max(0, 1 - f(alpha,b_i,b_u,s,t) + b_2)
            elif 'fit' in r['fit']:
                loss += max(0, 1 + f(alpha,b_i,b_u,s,t) - b_2)
                loss += max(0, 1 - f(alpha,b_i,b_u,s,t) + b_1)
            elif 'large' in r['fit']:
                loss += max(0, 1 + f(alpha,b_i,b_u,s,t) - b_1)
    return loss

for iterr in range(0,500):#发现450次的时候loss还在下降，没有收敛，因此增加到500
    learning_rate1 = 0.00025
    learning_rate2 = 0.00001/np.sqrt(iterr+1)
    #学习率
    grad_b1 = 0
    grad_b2 = 0#分别存储结果使用b1，b2的次数
    grad_s = np.zeros((len(user_index), K))
    grad_t = np.zeros((len(item_index), K))
    grad_bu = np.zeros(len(user_index))
    grad_bi = np.zeros(len(p_item_index))
    grad_bs = np.zeros(len(sizes))
    grad_w = np.zeros((K+3))
    for r in train_data:
        s = true_size_cust[user_index[r['user_id']]]#
        t = true_size_item[item_index[r['item_id'] + '|' + str(r['size'])]]#
        b_i = bias_i[p_item_index[r['item_id']]]#
        b_u = bias_u[user_index[r['user_id']]]#
        #变量初始化
        #下面开始计算损失函数
        if 'small' in r['fit']:

            #如果是small的情况
            A = 1 - f(alpha,b_i,b_u,s,t) + b_2
            if A>0:
                grad_s[user_index[r['user_id']]] += -1*np.multiply(w[3:],t)
                grad_t[item_index[r['item_id'] + '|' + str(r['size'])]] += -1*np.multiply(w[3:],s)
                
                grad_bu[user_index[r['user_id']]] += -1*w[1]
                grad_bi[p_item_index[r['item_id']]] += -1*w[2]
                grad_w += -1*np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                grad_b2 += 1
        elif 'fit' in r['fit']:
            #fit有两种情况
            B = 1 + f(alpha,b_i,b_u,s,t) - b_2
            C = 1 - f(alpha,b_i,b_u,s,t) + b_1
            if B>0:
                grad_s[user_index[r['user_id']]] += np.multiply(w[3:],t)
                grad_t[item_index[r['item_id'] + '|' + str(r['size'])]] += np.multiply(w[3:],s)
                
                grad_bu[user_index[r['user_id']]] += w[1]
                grad_bi[p_item_index[r['item_id']]] += w[2]
                grad_w += np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                grad_b2 += -1
            if C>0:
                grad_s[user_index[r['user_id']]] += -1*np.multiply(w[3:],t)
                grad_t[item_index[r['item_id'] + '|' + str(r['size'])]] += -1*np.multiply(w[3:],s)
                
                grad_bu[user_index[r['user_id']]] += -1*w[1]
                grad_bi[p_item_index[r['item_id']]] += -1*w[2]
                grad_w += -1*np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                grad_b1 += 1
        elif 'large' in r['fit']:
            D = 1 + f(alpha,b_i,b_u,s,t) - b_1
            if D>0:
                grad_s[user_index[r['user_id']]] += np.multiply(w[3:],t)
                grad_t[item_index[r['item_id'] + '|' + str(r['size'])]] += np.multiply(w[3:],s)
                
                grad_bu[user_index[r['user_id']]] += w[1]
                grad_bi[p_item_index[r['item_id']]] += w[2]
                grad_w += np.concatenate(([alpha, b_u, b_i], np.multiply(s, t)))
                grad_b1 += -1
    #对于userindex，训练更新users的bias矩阵数据
    for i in range(len(user_index)):
        true_size_cust[i] -= learning_rate1*(grad_s[i] + 2*lamda*true_size_cust[i])
        bias_u[i] -= learning_rate1*(grad_bu[i] + 2*lamda*bias_u[i])
    # 对于item，训练更新item和对应size的bias矩阵数据
    for i in range(len(item_index)):
        ## constraint update
        temp = true_size_item[i] - learning_rate1*(grad_t[i] + 2*lamda*true_size_item[i])
        if product_smaller[i] != -1:
            temp = np.maximum(temp, true_size_item[product_smaller[i]])
        if product_larger[i] != -1:
            temp = np.minimum(temp, true_size_item[product_larger[i]])
        true_size_item[i] = temp
        
    for i in range(len(p_item_index)):
        bias_i[i] -= learning_rate1*(grad_bi[i] + 2*lamda*bias_i[i])        
    
    b_1 -= learning_rate2*(grad_b1 + 2*lamda*b_1)
    b_2 -= learning_rate2*(grad_b2 + 2*lamda*b_2)
    w -= learning_rate2*(grad_w + 2*lamda*w)#更新w
    if iterr%5 == 0:
        print(iterr, total_loss(), b_1, b_2,w[:4])
        if iterr % 20 == 0:
            calc_auc()
            auc = calc_metric_auc()



calc_metric_auc()

# 总体模型是一个两层训练，第一层训练是针对用户偏差，商品偏差，w，alpha等参数的训练，然后在这些参数训练的基础上，对第二层参数进行继续训练
# 第二层严格意义上不是一个训练过程，他不改变任何存储在当前系统中的变量，仅仅是根据现有变量，使用LR/LMNN模型fit来得出结果，并进行预测，而在循环中，他每一次的fit都是不存储变量的
# 因此从宏观上讲，第二层我认为不是一个训练过程

# 调参来看我认为有两种方式，一种是更改lr这种训练过程中不变的值，即通常意义上的调参，这里可以调参：lr、训练次数、LMNN方法计算AUC中的取值位置
# 还有一种办法是，最后将w，alpha等值输出一下，然后直接在初始化的时候将他初始化为上一次训练的最终值，相当于实现了模型存储（其实不算是调参

# 还有一个很有意思的现象，用LR训练效果好于LMNN，LMNN训练最后结果输出达不到该github上显示的0.7，而是和最后一次输出一样大概在0.68左右

#简单记录一下，对于modcloth，使用当前模型，LR运行结果为0.59 LMNN结果为0.612

#训练500次，loss都无法收敛还在减小，可以考虑提高laerning rate