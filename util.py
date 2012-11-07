#coding=utf-8

from consts import *

def cal_vectors_average_distance(vectors, distance):
    print len(vectors)
    sum_distance = 0
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            sum_distance += distance(vectors[i], vectors[j])
    return float(sum_distance) / (len(vectors) * len(vectors))

#TODO 处理len(nonsparse_vector)为0的情况
def nonsparse_vector(vectors, max_limit_zero):
    print max_limit_zero
    nonsparse_vector = list()
    for vector in vectors:
        if vector.tolist().count(0) <= max_limit_zero:
            nonsparse_vector.append(vector)
    print len(nonsparse_vector)
    return nonsparse_vector

def max_density_means(vectors, average_distance, distance):

    average_distance = average_distance * THETA
    density = list()
    for i in range(len(vectors)):
        tmp_density = 0
        for j in range(len(vectors)):
            if distance(vectors[i], vectors[j]) <= average_distance:
                tmp_density += 1
        density.append(tmp_density)

    average_density = BETA * float(sum(density)) / len(density)

    S_d = list()
    means = list()
    max_density = (0, -1)
    for i in range(len(vectors)):
        if density[i] > average_density:
            if density[i] > max_density[0]:
                max_density = (density[i], i)
            S_d.append(i)
    print 'S_d', len(S_d)
    
    means.append(max_density[1])
    for i in range(int(len(S_d) * PHI)):
        max_choose = (0, -1)
        for vector_num in S_d:
            min_choose = INF
            if vector_num not in means:
                for mean_num in means:
                    d = distance(vectors[vector_num], vectors[mean_num])
                    if d < min_choose:
                        min_choose = d
            if min_choose > max_choose[0] and min_choose != INF:
                max_choose = (min_choose, vector_num)
        if max_choose[0] > 0:
            means.append(max_choose[1])
        else:
            break

    return map(lambda x:vectors[x], means)

def find_max_density(vectors, distance):
    number_of_zero = 0
    for vector in vectors:
        number_of_zero += vector.tolist().count(0)
    number_of_zero = float(number_of_zero) / len(vectors)

    max_limit_zero = number_of_zero * GAMMA
    new_vectors = nonsparse_vector(vectors, max_limit_zero)

    average_distance = cal_vectors_average_distance(new_vectors, distance)

    means = max_density_means(new_vectors, average_distance, distance)
    return means
    
