#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nlgeval import compute_individual_metrics
from tqdm import tqdm

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

import numpy as np


# In[ ]:


def _strip(s):
    return s.strip()

def new_compute_metrics(hyp_list, ref_list):
#     with open(hypothesis, 'r') as f:
#         hyp_list = f.readlines()
#     ref_list = []
#     for iidx, reference in enumerate(references):
#         with open(reference, 'r') as f:
#             ref_list.append(f.readlines())
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
        if isinstance(scorer, Meteor):
            scorer.close()
    del scorers

    return ret_scores


# In[ ]:


def new_compute_metrics_single(hyp_list, ref_list, scorers, score_dict=None):

    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    for scorer, method in scorers:
        score = scorer.calc_score(hyps[0], refs[0])
        if score_dict is not None:
            score_dict[method].append(score)
    return score
#     return ret_scores


# In[ ]:


if __name__ == "__main__":
    file_ref = open('/dccstor/gaot1/MultiHopReason/comprehension_tasks/public_code/OpenNMT-py/data/valid-tgt.all.lower.txt', 'r')
#     file_res = open('/dccstor/gaot1/MultiHopReason/comprehension_tasks/public_code/OpenNMT-py/narrative_qa_lower_logs/all_pred_step_22000.txt',
#                     'r')
    file_res = open('/dccstor/gaot1/MultiHopReason/comprehension_tasks/public_code/OpenNMT-py/narrative_qa_lower_logs/all_pred_step_16000.txt',
                    'r')

    references = file_ref.readlines()
    results = file_res.readlines()

    print(len(references))
    print(len(results))

    hyp_list = []
    ref_list = [[], []]

    for i in tqdm(range(len(references) // 2)):
        ref1 = references[i * 2].strip().replace(' .', '').lower()
        ref2 = references[i * 2 + 1].strip().replace(' .', '').lower()
        res = results[i * 2].strip().replace(' .', '').lower()

        hyp_list.append(res)
        ref_list[0].append(ref1)
        ref_list[1].append(ref2)

    print('number of predictions: {}'.format(len(hyp_list)))
        
    total_metrics_dict = new_compute_metrics(hyp_list, ref_list)    

    print(total_metrics_dict)
    
    test_single = True
    if test_single:
        score_dict = {}
        scorers = [
            (Rouge(), "ROUGE_L"),
        ]
        
        for scorer, method in scorers:
            if isinstance(method, list):
                for m in method:
                    score_dict[m] = []
            else:
                score_dict[method] = []
        
        for i in tqdm(range(len(references) // 2)):
            hyp_list = []
            ref_list = [[], []]
            
            ref1 = references[i * 2].strip().replace(' .', '').lower()
            ref2 = references[i * 2 + 1].strip().replace(' .', '').lower()
            res = results[i * 2].strip().replace(' .', '').lower()

            hyp_list.append(res)
            ref_list[0].append(ref1)
            ref_list[1].append(ref2)
            
            new_compute_metrics_single(hyp_list, ref_list, scorers, score_dict)
            
#         print(score_dict)            
        
        for scorer, method in scorers:
            print('method: %0.6f'%(np.mean(np.array(score_dict[method]))))

#         total_metrics_dict = new_compute_metrics(hyp_list, ref_list)    
        


# In[ ]:


target = ['jim s trial takes place in macedonia baptist church', 'macedonia baptist church']
res1 = ['macedonia baptist church']
res2 = ['we \'ll have the trial in de baptist church']

