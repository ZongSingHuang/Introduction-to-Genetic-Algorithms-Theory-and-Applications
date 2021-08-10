# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:54:43 2021

@author: zongsing.huang
"""

import numpy as np

def fitness(X):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    F = np.sum(X, axis=1)
    
    return F

def selection(X, F):
    if F.min()<0:
        F = F + np.abs( F.min() )
    F_sum = np.sum(F)
    
    normalized_F = F/F_sum
    idx = np.argsort(normalized_F)[::-1]
    sorted_F = np.sort(normalized_F)[::-1]
    
    cumsum_F = np.cumsum(sorted_F)[::-1]
    cumsum_F = np.hstack([cumsum_F[1:], 0.0])
    print(cumsum_F)
    
    new_idx = -1*np.zeros(2).astype(int)
    r = np.random.uniform(size=[2])
    print(r)
    for i in range(len(r)):
        for j in range(len(cumsum_F)):
            if r[i]>cumsum_F[j]:
                new_idx[i] = idx[j]
                break
    
    print(new_idx)
    p1, p2 = X[new_idx]
    
    return p1, p2

# 參數設定
P = 10
D = 5

# 初始化
X = np.random.randint(low=-10, high=10, size=[P, D])
F = fitness(X)

for k in range(0, P, 2):
    # 選擇
    p1, p2 = selection(X, F)