"""
    Task 2: Please run sci2014.py before you run this program!
        Implement the other algorithm
"""
import json, random
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn import metrics
from sklearn.mixture import GaussianMixture


if __name__ == "__main__":
    with open('./data/parsed_data.json', 'r') as file:
        json_dict = json.load(file)

    X = []
    for usr in json_dict:
        X.append(list(json_dict[usr].values()))

    ### K-means
    class_num = 3
    labels_1 = KMeans(n_clusters=class_num).fit_predict(X)
    print('K-Means', class_num, metrics.silhouette_score(X, labels_1, metric='euclidean'))

    ### DBSCAN
    labels_2 = DBSCAN(eps=2.5, min_samples=9).fit_predict(X)
    print('DBSCAN', max(labels_2) + 1,  metrics.silhouette_score(X, labels_2, metric='euclidean'))

    ### Hierarchical
    labels_3 = AgglomerativeClustering().fit_predict(X)
    print('Hierarchical', max(labels_3) + 1, metrics.silhouette_score(X, labels_3, metric='euclidean'))

    ### SpectralClustering
    labels_4 = SpectralClustering(n_clusters=3, random_state=0).fit_predict(X)
    print(metrics.silhouette_score(X, labels_4, metric='euclidean'))

    ### EM-GMM
    labels_5 = GaussianMixture(n_components=3, max_iter=20, random_state=0).fit_predict(X)
    print('EM-GMM', max(labels_5) + 1, metrics.silhouette_score(X, labels_5, metric='euclidean'))


    i = 0
    with open('./data/task2.csv', 'w') as file_tmp:
        with open('./data/task2_tmp.csv', 'r') as file:
            for line in file:
                file_tmp.write(line.strip('\n') + ', ' + str(labels_1[i]) + ', ' + str(labels_2[i]) + ', ' + str(labels_3[i]) + ', ' + str(labels_4[i]) + ', ' + str(labels_5[i]) + ']\n')
                i += 1
