"""
    Task 1 for the first assignment of data mining
"""
import matplotlib.pyplot as plt
import math


def calcu_local_density(points_x, points_y ,i ,dc):
    """
        计算第 i 个点的 local density
    """
    sum = 0
    for t in range(0, len(points_x)):
        if t == i: # 为了防止正向无限趋勤于0
            continue
        elif calcu_distance((points_x[t], points_y[t]), (points_x[i], points_y[i])) - dc < 0:
            sum += 1
        else:
            sum += 0
    return sum


def calcu_distance(point1, point2):
    """
        计算两个点之间的距离，勾股定理
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calcu_dc(x, y, d):
    """
        计算dc： 计算每两个点之间的距离，之后从小到大排序，选取第 2% 的点作为 dc 的值。
        注意：论文里写的是 1-2%。但是经过尝试，2%还算比较合适。
    """
    d_i_j = []
    for i in range(len(x)):
        temp_list = [-1] * (i + 1)
        for j in range(i + 1, len(x)):
            temp = calcu_distance((x[i], y[i]), (x[j], y[j]))
            d_i_j.append(temp)
            temp_list.append(temp)
        d.append(temp_list)
    d_i_j.sort()
    return(d_i_j[round(T*len(x)/2*(len(x) - 1))])


def calcu_min_distance_point(d, p_densities):
    """
        对每个点计算，到该点的最小距离和那个点的序号
    """
    ans = []
    for i in range(len(d)):
        min_point = -1
        min_distance = 9999
        for j in range(len(d)):
            if p_densities[j] <= p_densities[i]:
                continue
            distance = 10000
            if i == j:
                continue
            elif i < j:
                distance = d[i][j]
            else:
                distance = d[j][i]
            
            if distance < min_distance:
                min_point = j
                min_distance = distance
        ans.append(min_point)
    return ans


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
COLORS = ['#7f7f7f',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',]
T = 0.02
CLASSES = 6

if __name__ == "__main__":

    ### Get points coordinates from file
    x = []
    y = []
    COLORS = COLORS[0:CLASSES]
    distance = []
    with open('./Data/Aggregation.txt', 'r') as file:
        for line in file:
            point = line.strip().split(',')
            x.append(float(point[0]))
            y.append(float(point[1]))

    dc = calcu_dc(x, y, distance)


    ### Calculate each point's density
    points_densities = []
    max_density = 0
    for t in range(len(x)):
        temp = calcu_local_density(x, y, t, dc)
        if temp > max_density:
            max_density = temp
        points_densities.append(temp)

    min_distance_point = calcu_min_distance_point(distance, points_densities) # 距离每个点最近的点的序号


    ### Calculate each point's sigema
    sigemas = []
    for i in range(len(x)):
        if points_densities[i] == max_density:
            # print(max_density, i)
            max_distance = 0
            for j in range(len(x)):
                if i == j:
                    continue
                elif i < j:
                    max_distance = max(distance[i][j], max_distance)
                else:
                    max_distance = max(distance[j][i], max_distance)
            sigemas.append(max_distance)
        else:
            min_distance = 9999.0
            for j in range(len(x)):
                if points_densities[j] > points_densities[i]:
                    if i < j:
                        min_distance = min(distance[i][j], min_distance)
                    else:
                        min_distance = min(distance[j][i], min_distance)
            sigemas.append(min_distance)


    # Plot graph for gamma
    plt.figure()
    plt.title('The value of γi = pi*di in decreasing order for the data')
    temp_x = []
    gamma = []
    gamma_with_num = []
    density_with_num = []
    for i in range(len(x)):
        temp_x.append(i)
        gamma.append(points_densities[i]*sigemas[i])
        gamma_with_num.append([points_densities[i]*sigemas[i], i])
        density_with_num.append([points_densities[i], i])
    gamma.sort(reverse=True)
    gamma_with_num.sort(reverse=True)
    density_with_num.sort(reverse=True)

    plt.scatter(temp_x, gamma, s=20)


    # Plot Decision graph
    plt.figure()
    plt.title('Decision Graph')
    center_points_x = []
    center_points_y = []
    center_points_densities = []
    center_points_sigemas = []
    clusters = [-1] * len(x)

    for i in range(CLASSES):
        num = gamma_with_num[i][1] # 第 i cluster的中心点的序号
        clusters[num] = i
        center_points_x.append(x[num])
        center_points_y.append(y[num])
        center_points_densities.append(points_densities[num])
        center_points_sigemas.append(sigemas[num])
    # for i in range(8):
    #     if i == 1: # 根据图像结果，手动筛选掉一个重复的中心点
    #         continue
    #     num = gamma_with_num[i][1] # 第 i cluster的中心点的序号
    #     if i > 1:
    #         clusters[num] = i - 1
    #     else:
    #         clusters[num] = i
    #     center_points_x.append(x[num])
    #     center_points_y.append(y[num])
    #     center_points_densities.append(points_densities[num])
    #     center_points_sigemas.append(sigemas[num])

    plt.scatter(points_densities, sigemas, s=25)
    plt.scatter(center_points_densities, center_points_sigemas, s=80, c=COLORS)


    # Plot all of the points
    plt.figure()
    plt.title('Points Distributions')
    clusters_x = [[] for i in range(CLASSES)]
    clusters_y = [[] for i in range(CLASSES)]
    for i in range(len(x)):
        num = density_with_num[i][1] # 按 density 从大到下取出序号
        calcu_clusters(clusters, min_distance_point, num)

    for i in range(len(x)):
        if clusters[i] >= 0:
            clusters_x[clusters[i]].append(x[i])
            clusters_y[clusters[i]].append(y[i])

    for i in range(CLASSES):
        plt.scatter(clusters_x[i], clusters_y[i], s=25, c=COLORS[i])

    plt.show()
