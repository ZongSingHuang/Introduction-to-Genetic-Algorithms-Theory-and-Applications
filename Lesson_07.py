# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:54:43 2021

@author: zongsing.huang
"""

# =============================================================================
# #適應值計算+選擇(沒有迴圈)
# =============================================================================

import numpy as np

def fitness(X):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    F = np.sum(X, axis=1)
    
    return F

def selection(X, F):
    P = X.shape[0]
    
    # 修正適應值包含負值BUG
# =============================================================================
    if F.min()<0:
        F = F + np.abs( F.min() )
# =============================================================================
    F_sum = np.sum(F)
    
    normalized_F = F/F_sum
    idx = np.argsort(normalized_F)[::-1]
    sorted_F = np.sort(normalized_F)[::-1]
    
    cumsum_F = np.cumsum(sorted_F)[::-1]
    cumsum_F = np.hstack([cumsum_F[1:], 0.0])
    print(cumsum_F)
    
    new_idx = -1*np.zeros(P).astype(int)
    r = np.random.uniform(size=[P])
    print(r)
    for i in range(len(r)):
        for j in range(len(cumsum_F)):
            if r[i]>cumsum_F[j]:
                new_idx[i] = idx[j]
                break
    
    print(new_idx)
    p1 = X[new_idx][:int(P/2)]
    p2 = X[new_idx][int(P/2):]
    
    return p1, p2

# 參數設定
P = 10
D = 5
lb = -10*np.ones(D)
ub = 10*np.ones(D)

# 初始化
X = np.random.randint(low=lb, high=ub, size=[P, D])
F = fitness(X)

# 選擇
p1, p2 = selection(X, F)