"""
    Task 2: Implement sci2014 for task2 data
"""
import math, json
from sklearn import metrics


def calcu_local_density(i ,dc):
    """
        计算第 i 个点的 local density
    """
    sum = 0
    for t in range(LEN):
        if t == i: # 为了防止正向无限趋勤于0
            continue
        elif calcu_distance(t, i) - dc < 0:
            sum += 1
        else:
            sum += 0
    return sum


def calcu_distance(p1, p2):
    """
        计算两个点之间的距离，勾股定理。适应多维坐标点
    """
    temp_sum = 0
    for i in range(len(POINTS[p1])):
        temp_sum += (POINTS[p1][i] - POINTS[p2][i])**2
    return math.sqrt(temp_sum)


def calcu_dc(p_densities):
    """
        计算dc： 计算每两个点之间的距离，之后从小到大排序，选取第 T% 的点作为 dc 的值。
        注意：论文里写的是 T 属于 1-2%
        顺便将每两个点的距离存入 points_distance
    """
    d_i_j = []
    for i in range(LEN):
        temp_list = [-1] * (i + 1)
        for j in range(i + 1, LEN):
            temp_dist = calcu_distance(i, j)
            d_i_j.append(temp_dist)
            temp_list.append(temp_dist)
        p_densities.append(temp_list)
    d_i_j.sort()
    return d_i_j[round(T*LEN/2*(LEN - 1))]


def calcu_min_distance_points(p_distance, p_densities):
    """
        对每个点计算，到该点的最小距离和那个点的序号（那个点的 local density 应该大于他）
    """
    min_distance_point = []
    for i in range(LEN):
        min_point = -1
        min_distance = 9999
        for j in range(LEN):
            if p_densities[j] <= p_densities[i]:
                continue
            points_distance = 10000
            if i == j:
                continue
            elif i < j:
                points_distance = p_distance[i][j]
            else:
                points_distance = p_distance[j][i]
            
            if points_distance < min_distance:
                min_point = j
                min_distance = points_distance
        min_distance_point.append(min_point)
    return min_distance_point


def calcu_clusters(clusters, next_point, i):
    if clusters[i] == -1:
        clusters[i] = -2
        clusters[i] = calcu_clusters(clusters, next_point, next_point[i])
        return clusters[i]
    elif clusters[i] >= 0:
        # print(i, clusters[i])
        return clusters[i]
    elif clusters[i] == -2:
        return -2

### Const Varibles
POINTS = []
LEN = 0
T = 0.02
CLASSES = 3

if __name__ == "__main__":

    ### Get points coordinates from file
    points_distance = [] # 存有任意两点的距离，一个上三角矩阵
    with open('./data/parsed_data.json', 'r') as file:
        json_dict = json.load(file)

    for usr in json_dict:
        POINTS.append(list(json_dict[usr].values()))

    LEN = len(POINTS)
    dc = calcu_dc(points_distance)

    ### Calculate each point's local density
    points_densities = [] # 存放每个点的 local_densities
    max_local_density = 0 # 存放最大的 local_densities
    for i in range(LEN):
        temp_density = calcu_local_density(i, dc)
        max_local_density = max(temp_density, max_local_density)
        points_densities.append(temp_density)

    min_distance_points = calcu_min_distance_points(points_distance, points_densities) # 距离每个点最近的点的序号


    ### Calculate each point's sigema
    sigemas = []
    for i in range(LEN):
        if points_densities[i] == max_local_density:
            max_distance = 0
            for j in range(LEN):
                if i == j:
                    continue
                elif i < j: # 由于距离的倒三角矩阵
                    max_distance = max(points_distance[i][j], max_distance)
                else:
                    max_distance = max(points_distance[j][i], max_distance)
            sigemas.append(max_distance)
        else:
            min_distance = 9999.0
            for j in range(LEN):
                if points_densities[j] > points_densities[i]:
                    if i < j:
                        min_distance = min(points_distance[i][j], min_distance)
                    else:
                        min_distance = min(points_distance[j][i], min_distance)
            sigemas.append(min_distance)

    gamma_with_num = [] # 为了之后选择前几大的中心点
    density_with_num = [] # 为了聚类的时候按 local_density 从小到大选
    for i in range(LEN):
        gamma_with_num.append([points_densities[i]*sigemas[i], i])
        density_with_num.append([points_densities[i], i])
    gamma_with_num.sort(reverse=True)
    density_with_num.sort(reverse=True)

    clusters = [-1] * LEN
    for i in range(CLASSES):
        num = gamma_with_num[i][1] # 第 i cluster的中心点的序号
        clusters[num] = i

    for i in range(LEN):
        num = density_with_num[i][1] # 按 density 从大到下取出序号
        calcu_clusters(clusters, min_distance_points, num)

    print(metrics.silhouette_score(POINTS, clusters, metric='euclidean'))

    with open('./data/task2_tmp.csv', 'w') as file:
        keys = list(json_dict.keys())
        for i in range(LEN):
            file.write('[' + keys[i] + ', ' + str(clusters[i]) + '\n')
