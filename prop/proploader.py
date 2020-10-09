#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:28:50 2020

@author: ydzhao
"""
import glob
from os.path import join as pjoin
import pickle
import numpy as np

class Proposal(object):
    def __init__(self, no, boxarray, points):
        self.center = boxarray[0:3]  # center of bounding box
        self.extent = boxarray[3:6]     # extent of bounding box
        self.qrot = boxarray[6:10]    # rotation of bounding box
        self.score = boxarray[10]        # scale stability score
        self.points = points
    



class ProposalLoader:
    def __init__(self,  resultpath,  **kwargs):
        self.resultpath = resultpath
        self.seqdir = glob.glob(pjoin(self.resultpath, '*/'))
        self.sequences = [x.split('/')[-2] for x in self.seqdir ]
        self.sequences.sort()
        
        self.fparams = glob.glob(pjoin(self.resultpath, 'params.json'))
        self.fruntime = glob.glob(pjoin(self.resultpath, 'runtime.txt'))
        
        self.prop_struct = self._proposal_structure()

    def _proposal_structure(self, verbose=True):
        prop_struct = {}
        for seqd in self.seqdir:
            seq = seqd.split('/')[-2]
            
            # list all frames
            fframes = glob.glob(pjoin(seqd, "*.proposal"))
            frames = [ int(x.split('/')[-1][:-9]) for x in fframes]
            frames.sort()
            frames = np.array(frames)
            
            # frame step size
            nframes = frames.size
            begin, end = np.min(frames), np.max(frames)
            if nframes < 1:
                assert False, "No proposal found in %s!"%seqd
            elif nframes == 1:
                seqstep = (1 << 63)
            else:
                seqstep = frames[1:nframes] - frames[0:(nframes-1)]
                stepchecker = (seqstep == seqstep[0])
                if  stepchecker.all():
                    # fixed step size
                    prop_struct[seq] = [0, begin, end, seqstep[0]]
                else:                    
                    # variable step size
                    prop_struct[seq] = [1, frames] 
        return prop_struct            
            
            
        
    def getdata(self, what : [str], **kwargs):
        # do_transform = kwargs.get('do_transform', False)
        sequence = kwargs.get('sequence', None)
        frame = kwargs.get('frame', None)
        
        sequencetogo = self.sequences if sequence is None else [sequence, ]
        
        for seq in sequencetogo:
            seqdir = pjoin(self.resultpath, seq)    
            seqframes = glob.glob(pjoin(seqdir, '*.proposal'))
            seqframes = [ int(x.split('/')[-1][:-9]) for x in seqframes]
            seqframes.sort()
            
            frametogo = seqframes if frame is None else [frame, ]
            for fr in frametogo:
                fp = pjoin(seqdir, '%06d.proposal'%fr)
                retval = []
                with open(fp, "rb") as pfile:
                    (psequence, pframe, Tr) = pickle.load(pfile)
                    proposals = pickle.load(pfile)
                    
                for w in what:
                    if w == 'sequence':
                        retval.append(psequence)
                    elif w == 'frame-index':
                        retval.append(pframe)
                    elif w == 'proposals':
                        retval.append(proposals)
                    elif w == 'Tr-lidar-cam':
                        retval.append(Tr)
                    else:
                        raise ValueError("ProposalLoader cannot get:"+w)
                yield retval
                


