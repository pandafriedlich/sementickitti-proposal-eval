#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:51:04 2020

@author: ydzhao
"""

import sys
import numpy as np
from multiprocessing import Pool
import yaml

from . import evaluateobj as eobj
from . import utilities as utils

labels_to_names = { 
  0 : "unlabeled",
  1 : "outlier",
  10: "car",
  11: "bicycle",
  13: "bus",
  15: "motorcycle",
  16: "on-rails",
  18: "truck",
  20: "other-vehicle",
  30: "person",
  31: "bicyclist",
  32: "motorcyclist",
  40: "road",
  44: "parking",
  48: "sidewalk",
  49: "other-ground",
  50: "building",
  51: "fence",
  52: "other-structure",
  60: "lane-marking",
  70: "vegetation",
  71: "trunk",
  72: "terrain",
  80: "pole",
  81: "traffic-sign",
  99: "other-object",
  252: "moving-car",
  253: "moving-bicyclist",
  254: "moving-person",
  255: "moving-motorcyclist",
  256: "moving-on-rails",
  257: "moving-bus",
  258: "moving-truck",
  259: "moving-other-vehicle"
}
object_classes = [labels_to_names[x] for x in (10, 11, 13, 15, 16, 18, 20, 30, 31, 32)]
class_map = {x :x for x in labels_to_names.keys()}
class_map[252] = 10
class_map[253] = 31
class_map[254] = 30
class_map[255] = 32
class_map[256] = 16
class_map[257] = 13
class_map[258] = 18
class_map[259] = 20

class DefaultObjectFilter:
    def __init__(self, objtypes=object_classes):
        self.objtypes = objtypes
    
    def filterfunc(self, obj): 
        return True
    
    def getobjtype(self, obj):
        return obj.objtype



class SemanticKittiEvaluator:
    def __init__(self, objfilter=DefaultObjectFilter(), **kwargs):
        
        self.objfilter = objfilter
        
        # some options
        self.iouthreshold = kwargs.get('iouthreshold', -1)
        self.recallhistorylen = kwargs.get('recallhistorylen', 1000)
        # self.scoretype = kwargs.get('scoretype', 'scalestability')
        self.ioutype = 'indexmatching'
        
    
    def evaluate_single_frame(self, data):
        propinfo, frameinfo = data
        (sequence, frame_idx, labels) = frameinfo # frame of a dataset
        (props, sequencep, frame_idxp) = propinfo # proposals of that frame
        assert sequence==sequencep and frame_idx==frame_idxp, \
            "dataset loader and proposal loader output different frames!(%s-%d, %s-%d)"%(sequence, frame_idx, sequencep, frame_idxp)
        # print("Sequence: ", sequence, "Frame:", frame_idx)
        # load labels
        semantic_labels = labels & 0xffff
        instance_id = labels >> 16
        instance_ids = list(set(instance_id))
            
        # convert labels to evaluation objects
        groundtruths = []
        for j, instid in enumerate(instance_ids):
            if instid == 0: continue
            pts = (instance_id == instid) 
            # if self.positivez : pts &= (pc[:, -1] > 0)
            if pts.any():
                objtype = labels_to_names[class_map[semantic_labels[pts][0]] ]
                pointindices = np.where(pts)[0]
                gt = eobj.EvaluateObject.create_from_points(pointindices, objtype=objtype,
                                                            instance_id=instid, no=j)
                groundtruths.append(gt)
            
        
        # convert proposals to evaluation objects
        proposals = [None for p in props]
        for pi, p in enumerate(props):
            boxarray, points = p
            proposals[pi] = eobj.EvaluateObject.create_from_proposal(boxarray, points, no=pi)
        
        
        
        # compare proposals and groundtruths, use segment-based IoU
        frame_num_gts = utils.count_frame_num_gts(groundtruths, self.objfilter) # count number if gts per class
        ious_for_eval = utils.prepare_ious(groundtruths, proposals, which=self.ioutype)
        best_matches = utils.find_best_matches(groundtruths, proposals, ious_for_eval)
        frame_recall_history = utils.generate_frame_recall_history(best_matches,
                                                                   groundtruths,
                                                                   self.iouthreshold,
                                                                   self.objfilter,
                                                                   K=self.recallhistorylen)
        frame_recall_rate = utils.compute_recall_rate(frame_recall_history, frame_num_gts)
        

        # frame evaluation results
        frame_eval_results = {
            "seq" : sequence,
            "frame_idx" : frame_idx,
            "frame_recall_history" : frame_recall_history,
            "frame_num_gts" : frame_num_gts,
            "frame_recall_rate" : frame_recall_rate
            }
        return frame_eval_results
        

        


    def evaluate(self, dataloader, proploader, ncores=8, logfile=None):
        # dataloader
        proposalsgen = proploader.getdata(["proposals", "sequence", "frame-index"])
        datagen = dataloader.getdata(["sequence", "frame-index", "semantic-labels"],
                                          remove_background=False,
                                          get_struct=proploader.prop_struct,
                                          labelsdir='labels')

        
        input_data = ( (p, d) for p, d in zip(proposalsgen, datagen) )

        # parallelism
        if ncores > 1:
            with Pool(ncores) as p:
                evalres = p.map(self.evaluate_single_frame, input_data)
        else:
            evalres = list(map(self.evaluate_single_frame, input_data))

        # initalize containers
        overall_recall_history = utils.get_recall_history_recorder(self.recallhistorylen, self.objfilter.objtypes)
        overall_num_gts = utils.get_groundtruth_counter( objfilter=self.objfilter)
        
        # sort evaluation results according to sequence name and frame index for safty
        evalres.sort(key = lambda er: int(er["seq"]+"%06d"%er["frame_idx"]))
        
        # assembly evaluation results of all frames
        prev_seq = -1; seq_recall_history = None; seq_num_gts = None
        flog = open(logfile, "a") if logfile else None
        
        def write_log(logstr:str, logfobj=flog):
            if logfobj:
                logfobj.write(logstr)
            
        
        nevalres = len(evalres)
        for er_idx, er in enumerate(evalres):
            seq = er["seq"]
            if (seq != prev_seq and prev_seq  != -1 ):
                # end of sequence
                (seq_n_recall, seq_n_gt, seq_rrate) = \
                        utils.compute_recall_rate(seq_recall_history, seq_num_gts)['in_all']
                
                write_log(">>> Sequence end %d/%d (%.2f%%) <<<\n"%(seq_n_recall, seq_n_gt, seq_rrate*100))

                # add sequence result to overall
                for objtype, num_gt in seq_num_gts.items():
                    overall_num_gts[objtype] += num_gt
                    overall_recall_history[objtype] += seq_recall_history[objtype]
            if seq != prev_seq:
                # new sequence
                seq_recall_history = utils.get_recall_history_recorder(self.recallhistorylen , self.objfilter.objtypes)
                seq_num_gts = utils.get_groundtruth_counter(objfilter=self.objfilter)
                write_log(" >>> Sequence %s <<<\n"%seq)

            prev_seq = seq

            # get frame evaluation results
            frame_num_gts = er["frame_num_gts"]
            frame_recall_rate = er["frame_recall_rate"]
            frame_recall_history = er["frame_recall_history"]

            # write frame recall to log file
            write_log("*%-14s-%d\n"%('frame', er["frame_idx"]))
            for objtype, recall_info in frame_recall_rate.items():
                write_log("\t%-14s%-7d%-7d%-7.2f%%\n"%(objtype,
                                                 recall_info[0],
                                                 recall_info[1],
                                                 recall_info[2]*100))


            # sequence recall
            for objtype, num_gt in frame_num_gts.items():
                seq_num_gts[objtype] += num_gt
                seq_recall_history[objtype] += frame_recall_history[objtype]

            # last frame
            if er_idx == nevalres - 1:
                # end of sequence
                (seq_n_recall, seq_n_gt, seq_rrate) = \
                        utils.compute_recall_rate(seq_recall_history, seq_num_gts)['in_all']
                write_log(">>> Sequence end %d/%d (%.2f%%) <<<\n"%(seq_n_recall, seq_n_gt, seq_rrate*100))

                # add sequence result to overall
                for objtype, num_gt in seq_num_gts.items():
                    overall_num_gts[objtype] += num_gt
                    overall_recall_history[objtype] += seq_recall_history[objtype]
        # write overall recall to log
        overall_n_recall, overall_n_gt, overall_rrate = \
            utils.compute_recall_rate(overall_recall_history, overall_num_gts)['in_all']
        write_log("*** All complete%d/%d (%.2f%%) ***\n"%(overall_n_recall, overall_n_gt, overall_rrate*100))
        
        # close stream
        if logfile: flog.close()
        
        
        write_log("End of evaluation.")
        
        return (overall_num_gts, overall_recall_history)

