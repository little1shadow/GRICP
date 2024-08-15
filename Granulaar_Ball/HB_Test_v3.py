# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:43:58 2022

@author: xjnine
"""
import matplotlib
matplotlib.use('TkAgg')
import time
import numpy as np

from HyperBallClustering_acceleration_v3 import *
import datetime

def findNearestPoint(hb):
    hb_center=np.mean(hb, axis=0)
    hb=hb-hb_center
    l2_norms=np.linalg.norm(hb,axis=1)
    min_index=np.argmin(l2_norms)
    max_index=np.argmax(l2_norms)
    return [min_index,max_index]


def main():
    #keys = ['bunny']
    keys=['D5']
    data_path = "./synthetic/"
    for key in keys:
        begin = time.time()
        hb_list_temp, data = hbc([key], data_path)
        end = time.time()
        print('time cost:', end-begin)
        print("----------------------------------")

        hb_list_temp_represent=[]
        hb_radius=[]
        hb_nums=[]
        
        for hb in hb_list_temp:
            hb_list_temp_represent.append(hb[findNearestPoint(hb)[0]])
            hb_radius.append(get_radius(hb))
            hb_nums.append(len(hb))      
        draw_ball(hb_list_temp)
        plot_dot(data)
        
        filepath_radius="radius.txt"
        filepath_pointNums="pointNum.txt"
          
        with open(filepath_radius, 'w') as f:  
            for item in hb_radius:  
                f.write(str(item))  
                f.write('\n')  
        print("radius has been saved")
        
        with open(filepath_pointNums, 'w') as f:  
            for item in hb_nums:  
                f.write(str(item))  
                f.write('\n')  
        print("the point numbers has been saved")
if __name__ == '__main__':
    main()
