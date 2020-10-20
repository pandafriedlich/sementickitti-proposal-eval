#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:18:32 2020

@author: ydzhao
"""
# import pyquaternion as pq
import numpy as np
from .utilities import box3d_iou, boxoverlap
#import pyquaternion as pq

class BoundingBox2D(object):
    def __init__(self, lcorner, rcorner, imsize):
        self.x1, self.y1 = lcorner
        self.x2, self.y2 = rcorner
        self.imsize = imsize
        if imsize is not None:
            # hard-limit
            # 0 <= x1, x2  <= imwidth-1, 0 <= y1, y2 <= imheight-1
            self.x1 = max(self.x1, 0);self.x1 = min(self.x1, imsize[0]-1)
            self.x2 = max(self.x2, 0);self.x2 = min(self.x2, imsize[0]-1)
            self.y1 = max(self.y1, 0);self.y1 = min(self.y1, imsize[1]-1)
            self.y2 = max(self.y2, 0);self.y2 = min(self.y2, imsize[1]-1)
        self.lcorner = (self.x1, self.y1)
        self.rcorner = (self.x2, self.y2)
    
    def bbox2diou(self, another):
        return boxoverlap(self, another)
    
    
class BoundingBox3D:
    def __init__(self, center, extent, rotation):
        self.center = center
        self.extent = extent
        self.rotation = rotation
        self.vertices = BoundingBox3D._compute_vertices(self.center, self.extent, self.rotation)
    
    @staticmethod
    def _compute_vertices(center, extent, rotation):
        (Cx, Cy, Cz) = center
        (w, h, l) = extent
        vertices_x_off = np.array( [1/2, 1/2, -1/2, -1/2,  1/2,  1/2, -1/2, -1/2])*w
        vertices_y_off = np.array( [h/2,   h/2,  h/2,   h/2,   -h/2,   -h/2,   -h/2,   -h/2])
        vertices_z_off = np.array( [1/2, -1/2, -1/2, 1/2, 1/2, -1/2, -1/2, 1/2  ])*l
        vertices_off = np.vstack((vertices_x_off, vertices_y_off, vertices_z_off))
        Ry = rotation.rotation_matrix
        vertices_off =  np.dot(Ry, vertices_off)
        vertices = vertices_off.T + center
        return vertices
    
    def bbox3diou(self, other):
        return box3d_iou(self.vertices, other.vertices)
    


        


class EvaluateObject:
    def __init__(self, **kwargs):
        self.objtype = kwargs.get('objtype', 'unknown') # object type
        self.center3d = kwargs.get('center', None)      # center of 3D bounding box
        self.extent3d = kwargs.get('extent', None)      # extent of 3D bounding box
        self.qrot = kwargs.get('qrot', None)            # rotation along y-axis
        self.scalescore = kwargs.get('scalescore', None) # scale stability score
        self.objectness = kwargs.get('objectness', None)
        self.combined = kwargs.get('combined', None)
        
        self.points = kwargs.get('points', None)
        self.no = kwargs.get('no', -1)
        self.bbox2d = kwargs.get('bbox2d', None)
        self.bbox3d = kwargs.get('bbox3d', None)
        self.track_id = kwargs.get('track_id', None)
        self.frame_idx = kwargs.get('frame_idx', None)
        
        if (self.center3d is not None) and \
            (self.extent3d is not None) and (self.qrot is not None):
            self.bbox3d = BoundingBox3D(self.center3d, self.extent3d, self.qrot)
            
        
    def get_bbox2d(self, P2, pointcloud, imsize=None):
        assert self.points is not None,\
            "EvaluateObject doesn't have point indices! Cannot project to images"
        objcloud = pointcloud[self.points, :]
        pts_hom = np.column_stack((objcloud, np.ones(objcloud.shape[0])))
        image_coordinates = np.dot(P2, pts_hom.T)        
        normalizer =  image_coordinates[-1, :]
        image_coordinates = image_coordinates[0:2, :] / normalizer
        b2d_lcorner = \
            (int(np.min(self.image_coordinates[0,:])), int(np.min(self.image_coordinates[1,:])))
        b2d_rcorner = \
            (int( np.max(self.image_coordinates[0,:])), int(np.max(self.image_coordinates[1,:])))
        
        self.bbox2d = BoundingBox2D(b2d_lcorner, b2d_rcorner, imsize)
    
    def get_points_inbox(self, pointcloud):
        assert self.bbox3d is not None, "3D bounding box unavailable!"
        R = self.qrot.rotation_matrix
        extent = self.extent
        center = self.center        
        points_inlocal = np.dot(R.T, pointcloud.T).T - np.dot(R.T, center)
        points_is_in_bbox = (points_inlocal[:, 0] <= (extent[0]/2.)) \
            & ( points_inlocal[:, 1] <= (extent[1]/2.) ) \
            & ( points_inlocal[:, 2] <= (extent[2]/2.) ) \
            & ( points_inlocal[:, 0] >= (-extent[0]/2.) ) \
            & ( points_inlocal[:, 1] >= (-extent[1]/2.) ) \
            & ( points_inlocal[:, 2] >= (-extent[2]/2.) )   
        self.points = np.where(points_is_in_bbox)[0]
    
    
    def iou(self, another, ioutype:str='indexmatching'):
        iou = 0
        if ioutype == 'indexmatching':
            if (self.points is None) or (another.points is None):
                raise ValueError("Objects have no points!") # points not prepared
            intersect = np.intersect1d(self.points,
                                   another.points,
                                   assume_unique=True)
            I = intersect.size
            U = self.points.size + another.points.size - I
            if U > 0:
                iou = I / U
        # elif ioutype == '2d_image':
        #     pass
        # elif ioutype =='3d':
        #     pass
        else :
            raise ValueError("Unknown IoU type: "+str(ioutype))
        return iou
    
    
    # adapters
    @staticmethod 
    def create_from_proposal(boxarray, points, no=-1, **kwargs):
        return EvaluateObject(center=boxarray[0:3],
                              extent = boxarray[3:6],
                              qrot = None, #pq.Quaternion(boxarray[6:10]),
                              scalescore = boxarray[10],
                              objectness = boxarray[11],
                              points=points,
                              objtype = 'proposal',
                              no=no
                              )
    @staticmethod
    def create_from_trackinglabel(trackinglabel, pointindices=None,  no=-1, motslabel=None):
            frame_idx = trackinglabel['frame']
            track_id = trackinglabel['track_id']
            truncated = trackinglabel['truncated']
            occluded = trackinglabel['occluded']
            (BCx, BCy, BCz) = trackinglabel['loc_x':'loc_z'].tolist()
            (H, W, L) = trackinglabel['dimension_h':'dimension_l'].tolist()
            rot_y = trackinglabel['rotation_y']
            # qrot = pq.Quaternion(axis=[0, 1, 0], angle=rot_y)
            objtype = trackinglabel['type']
            bbox2d = trackinglabel['bbox_left':'bbox_bottom'].tolist()
            (x1, y1, x2, y2) = bbox2d
            bbox2d = BoundingBox2D((int(x1), int(y1)), (int(x2), int(y2)), None)
            
            # initialize bottom face center and extent
            extent = (W, H, L)
            
            # compute cuboid center
            center = (BCx, BCy - H * .5, BCz)
            
            return EvaluateObject(frame_idx=frame_idx,
                                  track_id=track_id,
                                  truncated=truncated,
                                  occluded=occluded,
                                  center=center,
                                  extent=extent,
                                  qrot=None, #qrot,
                                  bbox2d=bbox2d,
                                  objtype=objtype,
                                  no=no,
                                  points=pointindices
                                  )

    @staticmethod
    def create_from_points(pointindices, objtype, instance_id=None, no=-1):
        return EvaluateObject(points=pointindices,
                              objtype=objtype,
                              track_id=instance_id,
                              no=no)
    
    @staticmethod 
    def create_from_mots(motslabel, imsize=None, no=-1):        
        if motslabel.class_id in (1, 2): 
            # Car: 1, Pedestrian: 2
            objtype = 'Car' if motslabel.class_id == 1 else 'Pedestrian'
            track_id = (motslabel.track_id % 1000)
            
            # retrieve pixel coordinates
            indices = np.argwhere(motslabel.mask).T
            image_coordinates = np.dot(np.array([[0,1], [1,0]]), indices)
            
            b2d_lcorner = \
                (int(np.min(image_coordinates[0,:])), int(np.min(image_coordinates[1,:])))
            b2d_rcorner = \
                (int( np.max(image_coordinates[0,:])), int(np.max(image_coordinates[1,:])))
            bbox2d = BoundingBox2D(b2d_lcorner, b2d_rcorner, imsize)
            return EvaluateObject(objtype=objtype, 
                                  track_id=track_id,
                                  bbox2d=bbox2d,
                                  no=no
                                  )
            
        else :
            return None

