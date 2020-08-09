# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:13:14 2020

@author: wangxin
"""
import numpy as np

def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def compute_kernel_weight(X, y):
    norm_X = np.sum(X**2, axis=1, keepdims=True)
    W = norm_X - 2 * np.dot(X, X.T) + norm_X.T
    mask = compute_mask(y)
    sigma = 0.1 * np.mean(W)
    W = np.exp(-W / sigma)
    W = np.multiply(W, mask)
    return W

def compute_dot_weight(X, y):
    normed_X = X / np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    W = np.dot(normed_X, normed_X.T)
    mask = compute_mask(y)
    W = np.multiply(W, mask)
    return W

def compute_01_weight(X, y):
    mask = compute_maks(y)
    norm_X = np.sum(X**2, axis=1, keepdims=True)
    W = norm_X - 2 * np.dot(X, X.T) + norm_X.T
    W = np.multiply(W, mask)
            
    return W

def compute_mask(y):
    n_samples = y.shape[0]
    mask = np.zeros((n_samples, n_samples), dtype=np.int16)
    for i in range(n_samples):
        for j in range(n_samples):
            if y[i] == y[j]: mask[i][j] = 1
    return mask

def NMF(X, k = 10, lr = 0.1, reg = 0.1, iters = 100):
    X = X.T
    (m, n) = X.shape
    U = np.random.random((m, k))
    V = np.random.random((n, k))
    grad_U2 = np.zeros(U.shape)
    grad_V2 = np.zeros(V.shape)
    cnt = 0
    while(cnt < iters):
        X_hat = np.dot(U, V.T)
        E = X - X_hat
        grad_U = -np.dot(E, V) + reg * U
        grad_U2 += grad_U ** 2
        grad_V = -np.dot(E.T, U) + reg * V
        grad_V2 += grad_V ** 2
        
        lu = lr / (np.sqrt(grad_U2) + 1)
        lv = lr / (np.sqrt(grad_V2) + 1)
        
        U = U - np.multiply(lu, grad_U)
        V = V - np.multiply(lv, grad_V)
        
#        U = np.multiply(U, 
#                        np.divide(np.dot(X, V), 
#                                  np.dot(np.dot(U, V.T), V)))
#        V = np.multiply(V, 
#                        np.divide(np.dot(X.T, U), 
#                                  np.dot(np.dot(V, U.T), U)))
        cnt += 1
        X_hat = np.dot(U, V.T)
        Loss = np.sum((X - X_hat)**2)
        rmse = np.sqrt(np.sum((X - X_hat)**2) / (m * n))
        print("updating, cnt = {0}, rmse = {1}".format(cnt, rmse, Loss))
    return U, V
    
def GNMF(X, y, k = 10, reg = 1, iters = 300, dist_type = "kernel weight", log = False):
    if dist_type == "kernel weight":
        W = compute_kernel_weight(X, y)
    elif dist_type == "dot weight":
        W = compute_dot_weight(X, y)
    else:
        W = compute_01_weight(X, y)
    
    D = np.diag(np.sum(W, axis = 1))
    L = D - W 
    X = X.T
    (m, n) = X.shape
    U = np.random.random((m, k))
    V = np.random.random((n, k))
    rmse_log = []
    cnt = 0
    while(cnt < iters):
        U = np.multiply(U, 
                        np.divide(np.dot(X, V), 
                                  np.dot(np.dot(U, V.T), V)))
        V = np.multiply(V, np.divide(np.dot(X.T, U) + reg * np.dot(W, V), 
                                     np.dot(np.dot(V, U.T), U) + reg * np.dot(D, V)))
        
        cnt += 1
        if cnt % 20 == 0 and cnt != 0:
            X_hat = np.dot(U, V.T)
            rmse = np.sqrt(np.sum((X - X_hat)**2) / (m * n))
            print("updating, cnt = {0}, rmse = {1}".format(cnt, rmse))
            if log == True:
                rmse_log.append(rmse)
        
    return U, V, rmse_log

def PCA(X, k = 10):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = k)
    pca.fit(X)
    return pca.transform(X)

def K_means(X, cluster_num = 10):
    from sklearn.cluster import KMeans
    y_pred = KMeans(n_clusters=cluster_num, random_state=10).fit_predict(X)
    return y_pred
    
    
            
    