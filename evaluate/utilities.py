#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:45:21 2020

@author: ydzhao
"""


import numpy as np
from scipy.spatial import ConvexHull


## Some routines facilliating evaluations
def prepare_ious(bboxes_ground_truth, bboxes_proposal, which='indexmatching'):
    '''
    Compute the iou of each (groundtruth, proposal) belonging to the cartesian
    product of two lists of Seglib3dObject objecs, namely the 'bboxes_ground_truth'
    and 'bboxes_proposal'.

    Arguments:
        bboxes_ground_truth --- list of groundtruths (type: Seglib3dObject)
        bboxes_proposal --- list of proposals(type: Seglib3dObject)
        which --- accepted are 'indexmatching', '3d', '2d_image' and '2d_gp'

    '''
    ious = {(gt_idx, prop_idx):0. for gt_idx in range(len(bboxes_ground_truth))\
            for prop_idx in range(len(bboxes_proposal))}
    for gt_idx, gt in enumerate(bboxes_ground_truth):
        for prop_idx, prop in enumerate(bboxes_proposal):
            iou = gt.iou(prop, ioutype=which)
            ious[(gt_idx, prop_idx)] = iou
    return ious

def find_best_matches(bboxes_ground_truth, bboxes_proposal, ious):
    '''
    find the best match among proposals given the prepared ious.
    Arguments:
        bboxes_ground_truth --- list of groundtruths (type: Seglib3dObject)
        bboxes_proposal --- list of proposals(type: bboxes_proposal)
        ious ---  iou of each (groundtruth, proposal) belonging to the cartesian
                product of two lists of Seglib3dObject objecs, namely the 'bboxes_ground_truth'
                and 'bboxes_proposal'.
    '''
    best_matches = {gt_idx:[0., None] for gt_idx in range(len(bboxes_ground_truth))}
    for gt_idx, gt in enumerate(bboxes_ground_truth):
        bestiou = 0.; best_matched_prop = None;
        for prop_idx in range(len(bboxes_proposal)):
            iou = ious[(gt_idx, prop_idx)]
            if iou > bestiou:
                bestiou = iou
                best_matched_prop = prop_idx
        if best_matched_prop is not None :
            best_matches[gt_idx] = [bestiou, best_matched_prop]
    return best_matches

def find_matches(bboxes_ground_truth, bboxes_proposal, ious):
    '''
    find the best match among proposals given the prepared ious.
    Arguments:
        bboxes_ground_truth --- list of groundtruths (type: Seglib3dObject)
        bboxes_proposal --- list of proposals(type: bboxes_proposal)
        ious ---  iou of each (groundtruth, proposal) belonging to the cartesian
                product of two lists of Seglib3dObject objecs, namely the 'bboxes_ground_truth'
                and 'bboxes_proposal'.
    '''
    matches = {gt_idx:[] for gt_idx in range(len(bboxes_ground_truth))}
    for gt_idx, gt in enumerate(bboxes_ground_truth):
        for prop_idx in range(len(bboxes_proposal)):
            iou = ious[(gt_idx, prop_idx)]
            if iou > 0.:
                matches[gt_idx].append(prop_idx)
    return matches


def get_recall_history_recorder(K, objtypes):
    '''
    return a recall history recorder
    '''
    recall_history = {objtype : np.zeros(K) for objtype in objtypes }
    recall_history['in_all'] = np.zeros(K)
    return recall_history

def generate_frame_recall_history(best_matches, bboxes_ground_truth, IOU_THRESHOLD, objfilter, K=500):
    if IOU_THRESHOLD < 0:
        # Average Recall
        iou_threholds = tuple((x/100. for x in range(50, 100, 5)))
    else:
        iou_threholds = (IOU_THRESHOLD, )

    frame_recall_history = get_recall_history_recorder(K, objfilter.objtypes)
    for iou_thres in iou_threholds:
        for gt_idx, bm in best_matches.items():
            if bm[0] >= iou_thres:
                objtype = objfilter.getobjtype(bboxes_ground_truth[gt_idx])
                bm_prop_idx = bm[1]
                frame_recall_history[objtype][bm_prop_idx:] += 1
                frame_recall_history['in_all'][bm_prop_idx:] += 1
    for objtype, rh in frame_recall_history.items():
        rh /= len(iou_threholds)

    return frame_recall_history

def get_groundtruth_counter(objfilter):
    '''
    return a ground truth counter
    '''
    num_gts = {objtype: 0  for objtype in objfilter.objtypes}
    num_gts['in_all'] = 0
    return num_gts

def count_frame_num_gts(bboxes_ground_truth, objfilter):
    '''
    count number of groundtruths per class
    '''
    num_gts = get_groundtruth_counter(objfilter)
    for gt in bboxes_ground_truth:
        num_gts[objfilter.getobjtype(gt)] += 1
        num_gts['in_all'] += 1
    return num_gts

def compute_recall_rate(recall_history, num_gts, atk=0):
    '''
    compute recall rate
    Argumets:
        frame_recall_history --- recall history recorder
        frame_num_gts --- number of groundtruths per class
    '''

    recall = {}
    for objtype, rh in recall_history.items():
        n_recall = rh[atk-1]
        n_gt = num_gts[objtype]
        recall[objtype] = (n_recall, n_gt, n_recall/n_gt if n_gt > 0 else 0.)
    return recall



def boxoverlap(a,b,criterion="union"):
    """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
    """
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    w = x2-x1
    h = y2-y1

    if w<=0. or h<=0.:
        return 0.
    inter = w*h
    aarea = (a.x2-a.x1) * (a.y2-a.y1)
    barea = (b.x2-b.x1) * (b.y2-b.y1)
    # intersection over union overlap
    if criterion.lower()=="union":
        o = inter / float(aarea+barea-inter)
    elif criterion.lower()=="a":
        o = float(inter) / float(aarea)
    else:
        raise TypeError("Unkown type for criterion")
    return o

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0]
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

   outputList = subjectPolygon
   cp1 = clipPolygon[-1]

   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]

      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)



def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def box3d_iou(corners1, corners2, criterion='union'):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)]
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)

    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    if criterion.lower() == 'union':
	    iou = inter_vol / (vol1 + vol2 - inter_vol)
	    iou_2d = inter_area/(area1+area2-inter_area)
    elif criterion.lower() == 'a':
        iou = inter_vol / vol1
        iou_2d = inter_area / area1
    else:
        raise TypeError("Unkown type for criterion")
    return iou, iou_2d

def get_average_recall(ngt:dict, recallhistory:dict, **kwargs):
    min_percent = kwargs.get('minpercent', 0.0)
    curvelen = kwargs.get('curvelen', 400)

    metrics = {}
    # compute (average) recall
    for objtype, n in ngt.items():
        if objtype == 'in_all': continue # number of all groundtruths
        if n < 1 : continue # no ground truth
        if n/ngt['in_all'] < min_percent: continue # if #groundtruths is too small

        rh = recallhistory[objtype]
        rhsamples = rh/n     # recall
        metrics[objtype] = rhsamples[:curvelen]

    # mean average recall
    mAR = np.zeros(curvelen)
    nclasses = 0

    for objtype, rhsamples in metrics.items():
        mAR += rhsamples
        nclasses += 1
    mAR /= nclasses

    metrics["mean"] = mAR[:curvelen]
    return metrics



