# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:14:28 2021

@author: zongsing.huang
"""

# =============================================================================
# #交配
# =============================================================================

import numpy as np

def single_point(p1, p2):
    D = len(p1)
    cut_point = np.random.randint(D-1) # 一定會交換
    
    new_p1 = np.hstack([ p1[:cut_point], p2[cut_point:] ])
    new_p2 = np.hstack([ p2[:cut_point], p1[cut_point:] ])
    print(cut_point)
    print('='*20)
    print(p1)
    print(p2)
    print('='*20)
    print(new_p1)
    print(new_p2)
    
    return new_p1, new_p2

def double_point(p1, p2):
    D = len(p1)
    cut_point1, cut_point2 = np.sort( np.random.choice(range(1, D), size=2, replace=False) ) # 一定會交換

    new_p1 = np.hstack([ p1[:cut_point1], p2[cut_point1:cut_point2], p1[cut_point2:] ])
    new_p2 = np.hstack([ p2[:cut_point1], p1[cut_point1:cut_point2], p2[cut_point2:] ])
    print(cut_point1, cut_point2)
    print('='*20)
    print(p1)
    print(p2)
    print('='*20)
    print(new_p1)
    print(new_p2)
    
    return new_p1, new_p2

p1 = np.random.randint(low=0, high=30, size=7)
p2 = np.random.randint(low=0, high=30, size=7)
new_p1, new_p2 = single_point(p1, p2)
print('='*20)
print('='*20)
print('='*20)
p1 = np.random.randint(low=0, high=30, size=7)
p2 = np.random.randint(low=0, high=30, size=7)
new_p1, new_p2 = double_point(p1, p2)