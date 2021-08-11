# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:54:43 2021

@author: zongsing.huang
"""

# =============================================================================
# #適應值計算(沒有迴圈)
# =============================================================================

import numpy as np

def fitness(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    F = (x[:, 0]+x[:, 1]) * 12.3 * np.pi
    
    return F

# 參數設定
P = 10
D = 2

# 初始化
X = np.round( np.random.uniform(size=[P, D]) )
F = fitness(X)