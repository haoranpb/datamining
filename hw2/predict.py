"""
    Test
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np

def get_id_gender():
    id_gender = {}
    with open('./data/id_gender.csv', 'r') as file:
        for line in file:
            data = line.split(',')
            id_gender[data[0]] = int(data[1])
    return id_gender

def get_id_vec():
    id_vec = {}
    with open('./data/id_vec.csv', 'r') as file:
        for line in file:
            data = line.split(',')
            id_vec[data[0]] = []
            for num in data[1:]:
                id_vec[data[0]].append(float(num))
    return id_vec

def get_id_review_vec():
    id_review_vec = {}
    with open('./data/id_review_vec.csv', 'r') as file:
        for line in file:
            data = line.split(',')
            id_review_vec[data[0]] = []
            for num in data[1:]:
                id_review_vec[data[0]].append(float(num))
    return id_review_vec

def get_X_y(id_gender, id_vec, id_review_vec, id_mff):
    X = []
    X_2 = []
    X_3 = []
    y = []
    for key in id_gender:
        y.append(id_gender[key])
        X.append(id_vec[key])
        X_2.append(id_review_vec[key])
        X_3.append(id_mff[key])
    X = np.array(X)
    X_2 = np.array(X_2)
    X_3 = np.array(X_3)
    y = np.array(y)
    return (X, X_2, X_3, y)

def get_id_mff():
    id_mff = {}
    with open('./data/id_malicious_fnum_fave.csv', 'r') as file:
        for line in file:
            data = line.split(',')
            id_mff[data[0]] = []
            for num in data[1:]:
                id_mff[data[0]].append(float(num))
    return id_mff

if __name__ == "__main__":

    id_gender = get_id_gender()
    id_vec = get_id_vec()
    id_review_vec = get_id_review_vec()
    id_mff = get_id_mff()
    (X_1, X_2, X_3, y) = get_X_y(id_gender, id_vec, id_review_vec, id_mff)
    LEN = len(y)

    X = X_1
    # X = [[] for i in range(LEN)]
    # for i in range(LEN):
    #     X[i] = list(X_2[i])+ list(X_1[i])
    # X = np.array(X)

    # clf = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    # print('\nLogistic Regression:')
    # print('Accuracy: ', clf.score(X[int(LEN*0.9):, :], y[int(LEN*0.9):]))
    # print('Precision: ', precision_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('Recall: ', recall_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('F1-Score: ', f1_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))

    # clf = SVC(gamma='auto').fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    # print('\nSVM:')
    # print('Accuracy: ', clf.score(X[int(LEN*0.9):, :], y[int(LEN*0.9):]))
    # print('Precision: ', precision_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('Recall: ', recall_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('F1-Score: ', f1_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))

    # clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0).fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    # print('\nRandom Forest Classifier:')
    # print('Accuracy: ', clf.score(X[int(LEN*0.9):, :], y[int(LEN*0.9):]))
    # print('Precision: ', precision_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('Recall: ', recall_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('F1-Score: ', f1_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))

    # clf = MLPClassifier(max_iter=450).fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    # print('\nML-ReLu::')
    # print('Accuracy: ', clf.score(X[int(LEN*0.9):, :], y[int(LEN*0.9):]))
    # print('Precision: ', precision_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('Recall: ', recall_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))
    # print('F1-Score: ', f1_score(clf.predict(X[int(LEN*0.9):, :]), y[int(LEN*0.9):]))

    # clf_l = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    # clf_n = MLPClassifier(max_iter=550).fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    # proba_l = clf_l.predict_proba(X[int(LEN*0.9):, :])
    # proba_n = clf_n.predict_proba(X[int(LEN*0.9):, :])

    # y_proba = []
    # for i in range(len(proba_l)):
    #     proba = proba_l[i][0]*0.65 + proba_n[i][0]*0.35
    #     if proba > 0.5:
    #         y_proba.append(0)
    #     else:
    #         y_proba.append(1)
    # print('Accuracy: ', accuracy_score(y[int(LEN*0.9):], y_proba))
    # print('Precision: ', precision_score(y[int(LEN*0.9):], y_proba))
    # print('Recall: ', recall_score(y[int(LEN*0.9):], y_proba))
    # print('F1-Score: ', f1_score(y[int(LEN*0.9):], y_proba))

    clf_l = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    clf_n = MLPClassifier(max_iter=650).fit(X[:int(LEN*0.8), :], y[:int(LEN*0.8)])
    proba_l = clf_l.predict_proba(X)
    proba_n = clf_n.predict_proba(X)

    y_proba = []
    for i in range(LEN):
        proba = proba_l[i][0]*0.65 + proba_n[i][0]*0.35
        if proba > 0.5:
            y_proba.append(0)
        else:
            y_proba.append(1)

    i = 0
    with open('./data/dianping.csv', 'w') as file:
        for key in id_gender:
            file.write(key + ', ' + str(y_proba[i]) + ', ' + str(y[i]) + '\n')
            i += 1