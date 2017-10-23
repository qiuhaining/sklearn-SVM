# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:06:34 2017

@author: qiu
"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit#分层洗牌分割交叉验证
from sklearn.svm import SVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

digits = load_digits()

#网格搜索可视化——热力图
pipe = Pipeline(steps=[
                       
    ('classify',  SVC())
])
C_range = np.logspace(-2, 1, 4)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份
gamma_range = np.logspace(-9, -6, 4)
param_grid = [
    {
        'classify__C': C_range,
        'classify__gamma': gamma_range
    },
]

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)#基于交叉验证的网格搜索。

grid.fit(digits.data, digits.target)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))#找到最佳超参数