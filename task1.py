"""
    Task 1 for the first assignment of data mining
"""
import matplotlib.pyplot as plt
import math


def calcu_local_density(points_x, points_y ,i ,dc):
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
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calcu_dc(x, y, d):
    d_i_j = []
    for i in range(len(x)):
        temp_list = [-1] * (i + 1)
        for j in range(i + 1, len(x)):
            temp = calcu_distance((x[i], y[i]), (x[j], y[j]))
            d_i_j.append(temp)
            temp_list.append(temp)
        d.append(temp_list)
    d_i_j.sort()
    return(d_i_j[round(0.015*len(x)/2*(len(x) - 1))])


def calcu_min_distance_point(d):
    ans = []
    for i in range(len(d)):
        min_point = -1
        min_distance = 9999
        for j in range(len(d)):
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


if __name__ == "__main__":

    ### Get points coordinates from file
    x = []
    y = []
    distance = []
    with open('./Data/Aggregation.txt', 'r') as file:
        for line in file:
            point = line.strip().split(',')
            x.append(float(point[0]))
            y.append(float(point[1]))

    dc = calcu_dc(x, y, distance)
    min_distance_point = calcu_min_distance_point(distance)


    ### Calculate each point's density
    points_densities = []
    max_density = 0
    for t in range(len(x)):
        temp = calcu_local_density(x, y, t, dc)
        if temp > max_density:
            max_density = temp
        points_densities.append(temp)


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


    # Plot graph for center point
    plt.figure()
    temp_x = []
    temp_ans = []
    test = []
    for i in range(len(x)):
        temp_x.append(i)
        temp_ans.append(points_densities[i]*sigemas[i])
        test.append([points_densities[i]*sigemas[i], i])
    temp_ans.sort(reverse=True)
    test.sort(reverse=True)

    test_x = []
    test_y = []
    t_x = []
    t_y = []
    for i in range(12):
        num = test[i][1] # 序号
        test_x.append(x[num])
        test_y.append(y[num])
        t_x.append(points_densities[num])
        t_y.append(sigemas[num])

    plt.scatter(temp_x, temp_ans)


    # Plot Decision graph
    plt.figure()
    plt.scatter(points_densities, sigemas)
    plt.scatter(t_x, t_y)


    # Plot all of the points
    plt.figure()
    plt.scatter(x, y)
    plt.scatter(test_x, test_y)

    plt.show()
