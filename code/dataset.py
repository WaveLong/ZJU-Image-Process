# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:22:57 2020

@author: wangxin
"""
from PIL import Image
import numpy as np
from glob import glob
import os
import re
import random
import pickle

def save(data, path):
    fr = open(path, "wb")
    pickle.dump(data, fr)
    fr.close()
    print("Successfully save!")
    
def load(path):
    fr = open(path, "rb")
    res = pickle.load(fr)
    fr.close()
    print("Successfully load!")
    return res

def shuffle_data(X, y):
    idx = np.array(range(len(y)))
    np.random.shuffle(idx)
    X = X[idx, :]
    y = y[idx]
    return X, y

def read_pic(folder_name, new_shape = (32, 32), shuffle = False):
    db_name = folder_name.split("\\")[-1]
    print(db_name)
    if db_name == "101_ObjectCategories":
        X,y = read_OC(folder_name, new_shape = (32, 32))
    elif db_name == "mnist":
        X,y = read_coil20_mnist(folder_name, new_shape)
    elif db_name == "coil-20":
        X,y = read_coil20_mnist(folder_name, new_shape)
    print("Successfully load dataset!")
    return X, y-min(y)
def read_coil20_mnist(folder_name, new_shape):
    db_name = folder_name.split("\\")[-1]
    if db_name == "mnist":
        pics = glob(os.path.join(folder_name, "*\*\*"))
    else:
        pics = glob(os.path.join(folder_name, "*"))
        
    pic_num = len(pics)
    
    X = np.zeros((pic_num, new_shape[0]*new_shape[1]), dtype = np.float64)
    y = np.zeros(pic_num, dtype = np.uint8)
    
    for i in range(pic_num):
        tmp = Image.open(pics[i]).convert("L")
        tmp = tmp.resize(new_shape, Image.ANTIALIAS)
        X[i, :] = np.array(tmp).reshape(1, -1) / 255
        y[i] = re.findall(r"\d+\.?\d*",pics[i])[-2]
    
    return X, y

def read_OC(folder_name, new_shape):
    pics_folders = glob(os.path.join(folder_name, "*"))
    pics_folders_num = len(pics_folders)
    X = np.zeros((1, new_shape[0]*new_shape[1]), dtype = np.float32)
    y = np.zeros(1, dtype = np.uint8)
    for i in range(pics_folders_num):
        pics = glob(os.path.join(pics_folders[i], "*"))
        pic_num = len(pics)
        tmpX = np.zeros((pic_num, new_shape[0]*new_shape[1]), dtype = np.float64)
        tmpY = np.zeros(pic_num, dtype = np.uint8)
        for j in range(pic_num):
            tmp = Image.open(pics[j]).convert("L")
            tmp = tmp.resize(new_shape, Image.ANTIALIAS)
            tmpX[j, :] = np.array(tmp).reshape(1, -1) / 255
            tmpY[j] = i
        X = np.concatenate((X, tmpX), axis = 0)
        y = np.concatenate((y, tmpY))
    
    return X[1:, :], y[1:]
    
    