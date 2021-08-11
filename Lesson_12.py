# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:42:00 2021

@author: zongsing.huang
"""

# =============================================================================
# #適應值計算+選擇+交配+突變(沒有迴圈)
# =============================================================================

import numpy as np

def fitness(X):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    F = np.sum(X, axis=1)
    
    return F

def selection(X, F):
    P = X.shape[0]
    
    if F.min()<0:
        F = F + np.abs( F.min() )
    F_sum = np.sum(F)
    
    normalized_F = F/F_sum
    idx = np.argsort(normalized_F)[::-1]
    sorted_F = np.sort(normalized_F)[::-1]
    
    cumsum_F = np.cumsum(sorted_F)[::-1]
    cumsum_F = np.hstack([cumsum_F[1:], 0.0])
    
    new_idx = -1*np.zeros(P).astype(int)
    r = np.random.uniform(size=P)
    for i in range(len(r)):
        for j in range(len(cumsum_F)):
            if r[i]>cumsum_F[j]:
                new_idx[i] = idx[j]
                break
    
    p1 = X[new_idx][:int(P/2)]
    p2 = X[new_idx][int(P/2):]
    
    return p1, p2

def crossover(p1, p2, pc, species='single_point'):
    P = p1.shape[0]
    D = p1.shape[1]
    new_p1 = np.zeros_like(p1)
    new_p2 = np.zeros_like(p2)
    c1 = np.zeros_like(p1)
    c2 = np.zeros_like(p2)
    
    if species=='single_point':
        for i in range(P):
            cut_point = np.random.randint(D-1)
            new_p1[i] = np.hstack([ p1[i, :cut_point], p2[i, cut_point:] ])
            new_p2[i] = np.hstack([ p2[i, :cut_point], p1[i, cut_point:] ])
    elif species=='double_point':
        for i in range(P):
            cut_point1, cut_point2 = np.sort( np.random.choice(range(1, D), size=2, replace=False) )
            new_p1[i] = np.hstack([ p1[i, :cut_point1], p2[i, cut_point1:cut_point2], p1[i, cut_point2:] ])
            new_p2[i] = np.hstack([ p2[i, :cut_point1], p1[i, cut_point1:cut_point2], p2[i, cut_point2:] ])
    
    for i in range(P):
        r1 = np.random.uniform()
        if r1<pc:
            c1[i] = new_p1[i]
        else:
            c1[i] = p1[i]
            
        r2 = np.random.uniform()
        if r2<pc:
            c2[i] = new_p2[i]
        else:
            c2[i] = p2[i]
    
    return c1, c2

def mutation(c1, pm, lb, ub):
    P = c1.shape[0]
    D = c1.shape[1]
    
    for i in range(P):
        for j in range(D):
            r = np.random.uniform()
            if r<=pm:
                c1[i, j] = np.random.randint(low=lb[j], high=ub[j])
    
    return c1

# 參數設定
P = 10
D = 5
pc = 0.5
pm = 0.5
lb = -10*np.ones(D)
ub = 10*np.ones(D)

# 初始化
X = np.random.randint(low=lb, high=ub, size=[P, D])
F = fitness(X)

# 選擇
p1, p2 = selection(X, F)

# 交配
c1, c2 = crossover(p1, p2, pc, 'single_point')

# 突變
c1 = mutation(c1, pm, lb, ub)
c2 = mutation(c2, pm, lb, ub)