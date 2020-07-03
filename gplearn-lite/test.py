# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:58:12 2019

@author: lzy
"""

from gplearn.genetic import SymbolicTransformer
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import glob

files = glob.glob('sectional_pred/*')
for file in files:
    os.remove(file)



X = np.load('X.npy')
y = np.load('y_10.npy')
#panel = list(range(2901-1, 5170-1, 5))
#panel = list(range(2901-1, 5170-1, 10))
panel = list(range(5000-1, 5170-1, 10))
print('data loaded')

function_set = ['add', 'sub', 'mul', 'div','rank_add', 'rank_sub', 'rank_mul', 'rank_div', 'ts_max', 'ts_min', 'ts_nanmean', 'ts_prod', 'ts_rank', 'rank', 'ts_stddev', 'ts_sum', 'ts_corr', 'ts_cov','ts_argmax', 'ts_argmin', 'delta', 'delay']

gp = SymbolicTransformer(generations=2, population_size=500,
                         hall_of_fame=30, n_components=25,
                         function_set=function_set,
                         parsimony_coefficient=0.005,
                         max_samples=1, verbose=1,
                         random_state=688,
                         init_depth=(1, 3),
                         metric = 'rank_ic',
                         const_range = None,
                         feature_names = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'return1'],
                         p_crossover = 0.4,
                         p_subtree_mutation=0.01,
                         p_hoist_mutation=0,
                         p_point_mutation=0.01,
                         p_point_replace=0.4)

gp.fit(X, y, panel)

for program in gp:
    print(program)
    print(program.raw_fitness_)
    #print(program.oob_fitness_)
    print("--------------------------")