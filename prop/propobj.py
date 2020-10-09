#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:37:26 2020

@author: ydzhao
"""

class Proposal(object):
    def __init__(self, no, boxarray, points):
        self.center = boxarray[0:3]  # center of bounding box
        self.extent = boxarray[3:6]     # extent of bounding box
        self.qrot = boxarray[6:10]    # rotation of bounding box
        self.score = boxarray[10]        # scale stability score
        self.points = points
    

