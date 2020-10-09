#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:36:29 2020

@author: ydzhao
"""

import semkittieval

# set path to proposals and to dataset
proposals_path = '/media/ydzhao/Seagate/results/data/c745d813fa58e6bfbaaeaecb949bd435e6441fc0/proposals'
semantic_kitti_base ='/media/ydzhao/Seagate/kitti_datasets/semantic_kitti/dataset'

# evaluation
'''
The returned eval_results object is a dict-object, whose keys are the object classes like car, person, etc.,
as well as an additional "mean", which means the mean average recall of all objects. The values of this dict
are numpy arrays, which record how the recall change when number of proposal grows. For example,
eval_results['car'][99] means the average recall of cars when 100 objects are accepted.

'''
eval_results = semkittieval.evaluate_proposals(proposals_path, semantic_kitti_base, nthreads=4, nproposals=400, split='test')


# display eval_results, print a table and show a plot
import numpy as np
from matplotlib import pyplot as plt

# print a table of average recalls
nproposals = np.array([20, 30, 50, 100, 200]) - 1 # consider AR_20, AR_30, AR_50, AR_100, AR_200
table_headers = '\t'.join(["AR_%d"%(np+1) for np in nproposals])
print("\t", table_headers)

for objtype, evalres in eval_results.items():
    # for evaluation result of each class
    res_str_list = ['%.4f'%x for x in evalres[nproposals]] # AR values
    res_str = '\t'.join(res_str_list)
    if objtype != "mean":
        # make sure mean average recall occupies the last row
        print(objtype, "\t", res_str)
    else:
        mean_res_str = res_str
print("mean", "\t", mean_res_str) # print mean average recall

# plot recall vs. number of proposals
plt.figure()
legends = []
for objtype, evalres in eval_results.items():
    plt.plot(np.arange(400)+1, evalres) # plot average recall from 1 to 400
    legends.append(objtype)
plt.legend(legends)
plt.xlabel("#proposals")
plt.ylabel("AR")
plt.show()
