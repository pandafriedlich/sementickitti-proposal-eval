#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:22:35 2020

@author: ydzhao
"""
import prop.proploader as p
import dataloader.kittiodometry as dko
import evaluate.semanticeval as se
import evaluate.utilities as eu


def evaluate_proposals(proposalpath, datasetpath, nthreads=8, **kwargs):
    pl = p.ProposalLoader(proposalpath)
    ko = dko.KittiOdometryDataLoader(dataset_base=datasetpath)
    split = kwargs.get('split', 'test')
    curvelen = kwargs.get('nproposals', 400)

    if split == 'train':
        desired_sequences =  ['%02d'%(x) for x in range(0, 10+1) ]
    elif split == 'test':
        desired_sequences =  ['%02d'%(x) for x in range(11, 21+1) ]
    else:
        raise ValueError("Unknown split: %s"%(split))


    proposal_structure = pl.prop_struct
    sequences = list(proposal_structure.keys())
    assert sequences == desired_sequences, \
        "Missing sequences, detected: %s"%(sequences)


    # evaluation
    evltr = se.SemanticKittiEvaluator()
    ngt, recallhistory = evltr.evaluate(ko, pl, ncores=nthreads)
    metrics = eu.get_average_recall(ngt, recallhistory, curvelen=curvelen)



    return metrics
