#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:36:29 2020

@author: ydzhao
"""

import semkittieval


proposals_path = '/media/ydzhao/Seagate/results/data/c745d813fa58e6bfbaaeaecb949bd435e6441fc0/proposals'
semantic_kitti_base ='/media/ydzhao/Seagate/kitti_datasets/semantic_kitti/dataset'

# evaluation
eval_results = semkittieval.evaluate_proposals(proposals_path, semantic_kitti_base, nthreads=4, nproposals=400, split='train')

# display eval_results, print a table and show a plot
import numpy as np
from matplotlib import pyplot as plt

# print table
nproposals = np.array([20, 30, 50, 100, 200]) - 1
table_headers = '\t'.join(["AR_%d"%np for np in nproposals])
print("\t", table_headers)
for objtype, evalres in eval_results.items():
    res_str_list = ['%.4f'%x for x in evalres[nproposals]]
    res_str = '\t'.join(res_str_list)
    if objtype != "mean":
        print(objtype, "\t", res_str)
    else:
        mean_res_str = res_str
print("mean", "\t", mean_res_str)

# plot recall vs. number of proposals
plt.figure()
legends = []
for objtype, evalres in eval_results.items():
    plt.plot(np.arange(400)+1, evalres)
    legends.append(objtype)
plt.legend(legends)
plt.xlabel("#proposals")
plt.ylabel("AR")
plt.show()
