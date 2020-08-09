# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:13:21 2020

@author: wangxin
"""
#%% load modules
from dataset import *
from models import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
#%% hyper prameters
folder_names = ["coil-20", "101_ObjectCategories"]
cluster_num = [20, 101]
ks = [5, 10, 15, 20, 25]
regs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4]
shapes = [(64, 64), (200, 200)]
times = 5
#%% parameter select
for i in range(len(folder_names)):
    X, y = read_pic(os.path.join(".", folder_names[i]), shapes[i])
    # 5 ranks, 8 regs, 5 times
    acc = np.zeros((len(ks), len(regs), times), dtype = np.float32)
    for j in range(len(ks)):
        for k in range(len(regs)):
            for t in range(times):
                X, y = shuffle_data(X, y)
                U, V, _ = GNMF(X, y, k = ks[j], reg = regs[k])
                y_ = K_means(V, cluster_num = cluster_num[i])
                tmpAcc, _ = cluster_acc(y_, y)
                acc[j,k,t] = tmpAcc
                
                print("on dataset {0}, k={1}, lambda={2},acc = {3}".format(folder_names[i], ks[j], regs[k], tmpAcc))
    save(acc, "./acc_select_{0}.pkl".format(folder_names[i]))

#%% model compare
folder_names = ["coil-20", "101_ObjectCategories"]
cluster_num = [20, 101]
ks = [10, 15, 20, 25]
times = 5

#%%

for i in range(len(folder_names)):
    X, y = read_pic(os.path.join(".", folder_names[i]), shapes[i])
    # 4 models, 4 ranks, 5 times
    acc = np.zeros((4, len(ks), times), dtype = np.float32)
    y_ = K_means(X, cluster_num = cluster_num[i])
    acc0, _ = cluster_acc(y_, y)
    
    for j in range(len(ks)):
        for t in range(times):
            X, y = shuffle_data(X, y)

            acc[0, j, t] = acc0
            
            newX = PCA(X, k = ks[j])
            y_ = K_means(newX, cluster_num = cluster_num[i])
            acc1, _ = cluster_acc(y_, y)
            acc[1, j, t] = acc1
            
            U, V, _ = GNMF(X, y, k = ks[j], reg = 0)
            y_ = K_means(V, cluster_num = cluster_num[i])
            acc2,_ = cluster_acc(y_, y)
            acc[2, j, t] = acc2
            
            # the result of GNMF has been obtained in last cell
            
            print("fold:{4}, k={0}, K_means:{1}, PCA:{2},NMF:{3}".format(
                    ks[j],acc0, acc1, acc2, times))
            
    save(acc, "acc_model_compare_{0}.pkl".format(folder_names[i]))

#%% plot data
# dim : rank, reg, times
dbs = ["coil-20", "101_ObjectCategories"]
labels = ["COIL-20", "Caltech-101"]    
ks = [5, 10, 15, 20, 25]
regs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4]
xlim = [0.5, 0]
ylim = [0.9, 0.9]
for k in range(len(dbs)):
    acc = load("./acc_select_{0}.pkl".format(dbs[k]))
    mean_acc = np.mean(acc, axis = -1)
    std_acc = np.std(acc, axis = -1)
    
    color = ["purple", "red", "blue", "green", "coral"]
    x = range(mean_acc.shape[1])
    plt.figure(k+1)
    plt.ylim(xlim[k], ylim[k])
    plt.xticks(range(mean_acc.shape[1]), [r"$10^{0}$".format(i) for i in range(-3, 5)])
    plt.title("Model Paramter Select on {0}".format(labels[k]), 
              fontproperties='Times New Roman',
              fontsize = 14)
    plt.ylabel("Accuracy",
               fontproperties='Times New Roman',
               fontsize = 14)
    plt.xlabel(r"Graph Regular Parameter $\lambda$",
               fontproperties='Times New Roman',
               fontsize = 14)
    plt.grid(color = "k", linestyle = ":")
    for i in range(mean_acc.shape[0]):
        label = "$K = {0}$".format(ks[i])
        plt.plot(x, mean_acc[i,:], 
                marker = "o", markersize = 8,
                linestyle = "--", 
                c = color[i], label = label)
    plt.legend(loc = "best")
    plt.savefig("./{0}_model_select.png".format(dbs[k]), dpi=1024)
    plt.show()   