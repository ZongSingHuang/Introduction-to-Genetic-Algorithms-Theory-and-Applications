# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:42:00 2021

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
    
    new_idx = -1*np.zeros(2).astype(int)
    r = np.random.uniform(size=2)
    for i in range(len(r)):
        for j in range(len(cumsum_F)):
            if r[i]>cumsum_F[j]:
                new_idx[i] = idx[j]
                break
    
    p1, p2 = X[new_idx]
    
    return p1, p2

def crossover(p1, p2, pc, species='single_point'):
    D = len(p1)
    
    if species=='single_point':
        cut_point = np.random.randint(D-1)
        new_p1 = np.hstack([ p1[:cut_point], p2[cut_point:] ])
        new_p2 = np.hstack([ p2[:cut_point], p1[cut_point:] ])
    elif species=='double_point':
        cut_point1, cut_point2 = np.sort( np.random.choice(range(1, D), size=2, replace=False) )
        new_p1 = np.hstack([ p1[:cut_point1], p2[cut_point1:cut_point2], p1[cut_point2:] ])
        new_p2 = np.hstack([ p2[:cut_point1], p1[cut_point1:cut_point2], p2[cut_point2:] ])
    
    r1 = np.random.uniform()
    if r1<pc:
        c1 = new_p1
    else:
        c1 = p1
    r2 = np.random.uniform()
    
    if r2<pc:
        c2 = new_p2
    else:
        c2 = p2
    
    return c1, c2

# 參數設定
P = 10
D = 5
pc = 0.5

# 初始化
X = np.random.randint(low=-10, high=10, size=[P, D])
F = fitness(X)
new_X = np.zeros_like(X)

for k in range(0, P, 2):
    # 選擇
    p1, p2 = selection(X, F)
    
    # 交配
    c1, c2 = crossover(p1, p2, pc, 'single_point')
    
    new_X[k] = c1
    new_X[k+1] = c2