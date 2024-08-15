# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:46:00 2022

@author: xjnine
"""


from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


def division(hb_list, hb_list_not):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) > 0:
            ball_1, ball_2 = spilt_ball(hb)
            dm_parent = get_dm(hb)
            dm_child_1 = get_dm(ball_1)
            dm_child_2 = get_dm(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = w1 * dm_child_1 + w2 * dm_child_2
            t2 = w_child < dm_parent
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)
    return gb_list_new, hb_list_not


def spilt_ball(data):
    ball1 = []
    ball2 = []
    # n, m = data.shape
    # x_mat = data.T
    # g_mat = np.dot(x_mat.T, x_mat)
    # h_mat = np.tile(np.diag(g_mat), (n, 1))
    # d_mat = np.sqrt(h_mat + h_mat.T - g_mat * 2)

    # 调用pdist计算距离矩阵
    A=pdist(data)
    d_mat=squareform(A)
    r, c = np.where(d_mat == np.max(d_mat))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if d_mat[j, r1] < d_mat[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_dm(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center-hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = 0
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    if num > 2:
        return mean_radius
    else:
        return 1


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center-hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius


def plot_dot(data):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    plt.legend(loc=1)


def draw_ball(hb_list):
    is_isolated = False
    for data in hb_list:
        if len(data) > 1:
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, ls='-', color='black', lw=0.7)
        else:
            plt.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=3)
            is_isolated = True
    plt.plot([], [], ls='-', color='black', lw=1.2, label='hyper-ball boundary')
    plt.legend(loc=1)
    if is_isolated:
        plt.scatter([], [], marker='*', color='#0000EF', label='isolated point')
        plt.legend(loc=1)
    plt.show()


def normalized_ball(hb_list, hb_list_not, radius_detect):
    hb_list_temp = []
    for hb in hb_list:
        if len(hb) < 2:
            hb_list_not.append(hb)
        else:
            if get_radius(hb) <= 2 * radius_detect:
                hb_list_not.append(hb)
            else:
                ball_1, ball_2 = spilt_ball(hb)
                hb_list_temp.extend([ball_1, ball_2])
    
    return hb_list_temp, hb_list_not


def hbc(keys, data_path):
    for d in range(len(keys)):
        df = pd.read_csv(data_path + keys[d] + ".csv", header=None)
        data = df.values
        data = np.unique(data, axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        start_time = datetime.datetime.now()
        hb_list_temp = [data]
        hb_list_not_temp = []
        looptime = 1
        # 按照质量分化
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break            


        # 全局归一化
        radius = []  
        for hb in hb_list_temp:
            if len(hb) >= 2:
                radius.append(get_radius(hb))
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius_detect = max(radius_median, radius_mean)
        hb_list_not_temp = []
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = normalized_ball(hb_list_temp, hb_list_not_temp, radius_detect)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            looptime = looptime+1
            print('looptime',looptime)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break             
        end_time = datetime.datetime.now()
        print("优化后：", keys[d], "：", end_time - start_time)
    return hb_list_temp, data
