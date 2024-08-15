# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:46:00 2022

@author: xjnine
"""


from scipy.spatial.distance import pdist, squareform
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from mayavi import mlab


def plot_dot(data):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="GREEN", linewidths=3, alpha=0.6, marker='o', label='data point')
    plt.legend(loc=1)

def plot_dot1(data):
    total_points = len(data)
    noise_start_index = int(total_points * 0.9)  # 计算前90%数据点的数量
    normal_points = data[:noise_start_index]
    noise_points = data[noise_start_index:]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(normal_points[:, 0], normal_points[:, 1], s=7, c="GREEN", linewidths=3, alpha=0.6, marker='o', label='data point')
    plt.scatter(noise_points[:, 0], noise_points[:, 1], s=7, c="RED", linewidths=3, alpha=0.6, marker='o', label='noise point')
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
            plt.plot(x, y, ls='-', color='blue', lw=0.7)
        else:
            #plt.plot(data[0][0], data[0][1], marker='*', color="RED", markersize=3)
            is_isolated = True
    plt.plot([], [], ls='-', color='blue', lw=1.2, label='granular-ball boundary')
    plt.legend(loc=1)
    if is_isolated:
        #plt.scatter([], [], marker='*', color='red', label='noise point')
        plt.legend(loc=1)
    plt.axis('off')
    plt.savefig('example_gb.png', dpi=300,
                bbox_inches='tight')
    plt.show()


def plot_dot_3d(data):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:,2], c="GREEN", linewidths=0.0001, alpha=0.5, marker='.', label='data point')
    plt.legend(loc=1)
    plt.show()


def draw_ball_3d(hb_list,datasets):
    fig = mlab.figure(size=(800, 800),bgcolor=(1,1,1))

    for data in hb_list:
        if len(data) > 1:
            center = np.array(data).mean(0)
            radius = np.max((((np.array(data) - center) ** 2).sum(axis=1) ** 0.5))
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = center[0] + radius * np.outer(np.sin(v), np.cos(u))
            y = center[1] + radius * np.outer(np.sin(v), np.sin(u))
            z = center[2] + radius * np.outer(np.cos(v), np.ones_like(u))

            mlab.mesh(x, y, z, color=(1, 0, 0), opacity=0.2)
    if datasets is not None:
        mlab.points3d(datasets[:, 0], datasets[:, 1], datasets[:, 2], color=(0, 1, 0), mode='sphere', scale_factor=0.005)
    # 调整视角，使球体占据整个图像
    mlab.view(azimuth=10, elevation=90, distance=1.5)

    # 保存图像
    mlab.savefig('E:/Desktop/个人论文/figures/registration/Granular_ball_mode_2d.eps',magnification=2)
    mlab.show()


def get_dm(hb): #计算子球的质量,hb为粒球中所有的点，目前其实就是标准的密度计算
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center-hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = 0
    mean_radius = sum(distances)/num #下面的语句没有必要使用for循环
    # for i in distances:
    #     sum_radius = sum_radius + i
    # mean_radius = sum_radius / num
    # print('%%%%%%%mean_radius:', mean_radius)
    if num > 2:
        return mean_radius
    else:
        return 1

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
            w_child = w1 * dm_child_1 + w2 * dm_child_2 #某一个子球的质量
            # print('np.shape(dm_child_1),np.shape(dm_child_2)',dm_child_1,dm_child_2)
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
    ball3 = []
    ball4 = []
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
    temp = d_mat[:, r1] < d_mat[:, c1]
    temp2 = d_mat[:, r1] >= d_mat[:, c1]
    ball1.extend([data[temp, :]])
    ball2.extend([data[temp2, :]])
    ball1 = np.array(ball1[0][:][:]) #类型转换兼容其他程序
    ball2 = np.array(ball2[0][:][:])
    return [ball1, ball2]



def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center-hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius

def normalized_ball(hb_list, hb_list_not, radius_detect, radius,whileflag=0):
    hb_list_temp = []
    if whileflag != 1:
        for hb in hb_list:
            if len(hb) < 2:
                hb_list_not.append(hb)
            else:
                # print('小循环radiusradiusradiusradius', get_radius(hb))
                if get_radius(hb) <= 2 * radius_detect:
                    hb_list_not.append(hb)
                else:
                    ball_1, ball_2 = spilt_ball(hb)
                    hb_list_temp.extend([ball_1, ball_2])

    if whileflag==1:
        for i, hb in enumerate(hb_list):
            if len(hb) < 2:
                hb_list_not.append(hb)
            else:
                # print('小循环radiusradiusradiusradius', get_radius(hb))
                # print(np.shape(radius),i)
                if radius[i] <= 2 * radius_detect:
                    hb_list_not.append(hb)
                else:
                    ball_1, ball_2 = spilt_ball(hb)
                    hb_list_temp.extend([ball_1, ball_2])

    return hb_list_temp, hb_list_not


def hbc(keys, data_path):
    looptime = 1 #normalized_ball函数中，控制第一次的getradius调用两次
    for d in range(len(keys)):
        df = pd.read_csv(data_path + keys[d] + ".csv", header=None)
        data = df.values
        data = np.unique(data, axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        start_time = datetime.datetime.now()
        hb_list_temp = [data]
        hb_list_not_temp = []
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
        # print('大循环radiusradiusradiusradius',radius)
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius_detect = max(radius_median, radius_mean)
        hb_list_not_temp = []
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = normalized_ball(hb_list_temp, hb_list_not_temp, radius_detect,radius,whileflag = looptime)
            # looptime控制第一次的getradius调用两次
            looptime=looptime+1
            # print('looptimelooptime',looptime)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break             
        end_time = datetime.datetime.now()
        print("优化后：", keys[d], "：", end_time - start_time)
    return hb_list_temp, data
