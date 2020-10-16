#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:35:02 2020

@author: ydzhao
"""

from .stolenpykitti.odometry import odometry
import os
import numpy as np
import json
from typing import List



class KittiOdometry(odometry):
    def __init__(self, base_path, sequence, **kwargs):
        super().__init__(base_path, sequence, **kwargs)
        self.sequence_dir = os.path.join(base_path, "sequences", sequence)
        
    def get_velo_labels(self, frame_idx, **kwargs):
        labelsdir = kwargs.get('labelsdir', 'labels')
        label_path = os.path.join(self.sequence_dir, labelsdir, "%06d.label"%frame_idx)
        label = np.fromfile(label_path, dtype=np.uint32)
        return label

class KittiOdometryDataLoader:
    def __init__(self,  **kwargs):        
        self.datasetname = 'KittiOdometry'
        self.fconfig = kwargs.get('fconfig', None)  # path to configuration file as argument
        self.config = kwargs.get('config', None)    # configuration dictionary as argument
        self.dataset_base_path = kwargs.get('dataset_base', None)
        self.sequences = kwargs.get('sequences', [])
        self.frame_step_size = kwargs.get('frame_step_size', -1)        
        
        
        # if path to config file is passed in, load configs
        if self.fconfig is not None:
            with open(self.fconfig, 'r') as fconfig:
                self.config = json.load(fconfig)
        
        # if there is a config, use config info
        if self.config is not None:
            self.sequences = self.config.get('sequences', self.sequences)
            self.frame_step_size = self.config.get('frame_step_size', self.frame_step_size)        
            self.dataset_base_path = self.config.get('dataset_base_path', self.frame_step_size)
        
        self.training_path = self.dataset_base_path
        
        
    def getdata(self, what:List[str], **kwargs):        
        # loading options
        remove_background = kwargs.get('remove_background', True)
        do_transform = kwargs.get('do_transform', True)
        labels_dir = kwargs.get('labelsdir', 'labels')
        background_labels = kwargs.get('background_labels', [0, 1, 44, 48, 49, 50, 51, 52, 60, 70, ])  
        # seq = kwargs.get('sequence', None)
        # frame = kwargs.get('frame_idx', None)
        get_struct = kwargs.get('get_struct', None)
        frame_step_size = self.frame_step_size     

 
        # sequences to load
        sequences = get_struct.keys() if get_struct is not None else self.sequences    

        for seq in sequences:
            # initialize a data loader
            seqloader = KittiOdometry(self.training_path, sequence=seq)
            nframes =   len(seqloader.velo_files)
            
            # retrieve calibration data of this sequence
            Tr_velo_cam_hom = seqloader.calib.T_cam2_velo            
            
            if get_struct is not None:
                if get_struct[seq][0] == 0:
                    # fixed step size
                    frames = range(get_struct[seq][1], get_struct[seq][2]+1, get_struct[seq][3])
                elif get_struct[seq][0] == 1:
                    # variable step size
                    frames = get_struct[seq][1]
                else:
                    # unknown
                    raise ValueError("Unknown structure prefix: %d, sequence: %s"%(get_struct[seq][0], seq))
            else:
                frames = range(0, nframes, frame_step_size)
            
            for frame_idx in frames:
                assert frame_idx < nframes, "Frame index greater than number of frames"
                retval = []
                for w in what:
                    if w == 'pointcloud':                 
                        # raw point cloud in velodyne coordinate
                        pts_invelo = seqloader.get_velo(frame_idx)                
                        # remove background points
                        if remove_background:
                            pts_label = seqloader.get_velo_labels(frame_idx, labelsdir=labels_dir)
                            pts_is_background = np.zeros(pts_label.shape, dtype=bool)
                            for bl in background_labels:
                                pts_is_background |= (pts_label == bl)
                            pts_invelo[pts_is_background, :] = np.nan
                        
                        # homogeneous coordinate
                        pts_invelo[:, -1] = 1.
                        
                        # transform point cloud from LiDAR frame to camera frame
                        if do_transform:
                            pts_incamera = np.transpose( np.dot(Tr_velo_cam_hom, pts_invelo.T) )
                            pts_incamera = pts_incamera[:, :-1] # the last column must be all 1's
                            retval.append( pts_incamera)
                        else:
                            retval.append ( pts_invelo[:, :-1])                
                    
                    elif w == 'semantic-labels':
                        pts_label = seqloader.get_velo_labels(frame_idx, labelsdir=labels_dir)
                        retval.append(pts_label)
                            
                    elif w == 'Tr-lidar-cam':
                        # Transform from lindar to cam, 4x4 matrix
                        retval.append( Tr_velo_cam_hom ) 
                    
                    elif w == 'sequence':
                        retval.append(seq)
                        
                    elif w == 'frame-index':
                        retval.append(frame_idx)
                        
                    elif w == 'nframes':
                        # number of frames
                        retval.append(nframes)
                    else:
                        raise ValueError("KittiTrackingDataLoader cannot get data: "+ what)
                    

                yield retval
