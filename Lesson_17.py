# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:42:00 2021

@author: zongsing.huang
"""

# =============================================================================
# x in {0, 1}
# 最大化問題的最佳適應值為-D；最小化問題的最佳適應值為0
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

def fitness(X):
    # Sphere
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    F = np.sum(X**2, axis=1).astype(float)
# =============================================================================
    # # 求max problem
    # F = -1*F
# =============================================================================
    
    return F

def selection(X, F):
    P = X.shape[0]
    
    if F.min()<0:
        F = F + np.abs( F.min() )
    F_sum = np.sum(F)
    
    if F_sum==0:
        # 因為題目太簡單，所以還沒迭代完成就會使所有染色體都達到最佳解
        # 因為F_sum=0，所以F/F_sum = 0/0 會跳警告
        # 因此這邊下一個機制
        normalized_F = np.zeros(P)
    else:
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
    # Bit Flip
    P = c1.shape[0]
    D = c1.shape[1]
    
    for i in range(P):
        for j in range(D):
            r = np.random.uniform()
            if r<=pm:
                c1[i, j] = np.abs(c1[i, j]-1)
    
    return c1

def elitism(X, F, new_X, new_F, er):
    P = X.shape[0]
    elitism_size = int(P*er)
    new_X2 = new_X.copy()
    new_F2 = new_F.copy()
    
    if elitism_size>0:
        idx = np.argsort(F)
        elitism_idx = idx[:elitism_size]
        new_X2[:elitism_size] = X[elitism_idx]
        new_F2[:elitism_size] = F[elitism_idx]
    
    return new_X2, new_F2

def immigrant(new_X, new_F, ir, lb, ub):
    P = new_X.shape[0]
    D = new_X.shape[1]
    immigrant_size = int(P*er)
    
    if immigrant_size>0:
        immigrant_X = np.random.choice(2, size=[immigrant_size, D])
        immigrant_F = fitness(immigrant_X)
        idx = np.argsort(new_F)
        worst_idx = idx[-immigrant_size:]
        new_X[worst_idx] = immigrant_X
        new_F[worst_idx] = immigrant_F
    
    return new_X, new_F

#%% 參數設定
P = 20 # 一定要偶數
D = 20
G = 50
pc = 0.85
pm = 0.01
er = 0.05
ir = 0.05
lb = 0*np.ones(D)
ub = 1*np.ones(D)

#%% 初始化
X = np.random.choice(2, size=[P, D])
gbest_X = np.zeros(D)
gbest_F = np.inf
loss_curve = np.zeros(G)

# 若P不是偶數，則進行修正
if P%2!=0:
    P = 2 * (P//2)
    
#%% 迭代
# 適應值計算
F = fitness(X)
    
for g in range(G):
    # 更新F
    if F.min()<gbest_F:
        best_idx = np.argmin(F)
        gbest_X = X[best_idx]
        gbest_F = F[best_idx]
    loss_curve[g] = gbest_F
    
    # 選擇
    p1, p2 = selection(X, F)
    
    # 交配
    c1, c2 = crossover(p1, p2, pc, 'single_point')
    
    # 突變
    c1 = mutation(c1, pm, lb, ub)
    c2 = mutation(c2, pm, lb, ub)
    
    # 更新X
    new_X = np.vstack([c1, c2])
    np.random.shuffle(new_X)
    
    # 適應值計算
    new_F = fitness(new_X)
    
    # 菁英
    new_X, new_F = elitism(X, F, new_X, new_F, er)
    
    # 移民
    new_X, new_F = immigrant(new_X, new_F, ir, lb, ub)
    
    X = new_X.copy()
    F = new_F.copy()
    
#%% 作畫
plt.figure()
plt.plot(loss_curve)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')
