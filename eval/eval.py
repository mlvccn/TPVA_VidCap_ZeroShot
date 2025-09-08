import torch
import torch.nn as nn
from collections import OrderedDict
from eval.pycocoevalcap.bleu.bleu import Bleu
from eval.pycocoevalcap.cider.cider import Cider
from eval.pycocoevalcap.meteor.meteor import Meteor
from eval.pycocoevalcap.rouge.rouge import Rouge
from eval.pycocoevalcap.spice.spice import Spice
from eval.clip_score.clipscore import ClipScore
import json

# def eval(model,eval_dataloader,tokenizer,torch_dtype):
#     batched_pred_seq = []
#     batched_ground_seq = []
#     model.eval()
#     model.to(torch_dtype)
#     with torch.no_grad():
#         for cap,vid_feat in eval_dataloader:
#             vid_feat = vid_feat.to(torch_dtype)
#             pred = model(vid_feat)
#             batched_pred_seq += pred
#             batched_ground_seq += cap
            
#             f = open('out/ans.txt',mode='a')
#             for ans in pred:
#                 f.write(ans + '\n')
#         f.write('-----------------------------------\n')
#         f.close()
#     model.train()
#     return language_eval(batched_pred_seq,batched_ground_seq)

def language_eval(predictions, references, args):
    # assert len(sample_seqs) == len(groundtruth_seqs)

    # references, predictions = OrderedDict(), OrderedDict()
    # for i in range(len(groundtruth_seqs)):
    #     references[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]
    # for i in range(len(sample_seqs)):
    #     predictions[i] = [sample_seqs[i].lower()]

    # predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    # references = {i: references[i] for i in range(len(groundtruth_seqs))}
    

    assert len(predictions.keys()) == len(references.keys())
    
    # json.dump(predictions, open('/home/wangtao/video_caption/project_v3.0/eval/vatex_clips_candidates.json', "w"))
    # json.dump(references, open('/home/wangtao/video_caption/project_v3.0/eval/vatex_clips_reference.json', "w"))

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)

    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)

    avg_clip_score, avg_refclip_score = ClipScore(args).compute_score(references, predictions)
    print('avg_clip_score == ', avg_clip_score)
    print('avg_refclip_score == ',  avg_refclip_score)

    # avg_spice_score, spice_score = Spice().compute_score(references, predictions)
    # print('avg_spice_score == ', avg_spice_score)

    return {'BLEU_4': avg_bleu_score[3], 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 
            'ROUGE': avg_rouge_score, 'CLIPScore': avg_clip_score, 'RefCLIPScore': avg_refclip_score}

def text_only_language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs)
    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i].lower()]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i].lower()]

    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}
    
    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)

    # avg_spice_score, spice_score = Spice().compute_score(references, predictions)
    # print('avg_spice_score == ', avg_spice_score)

    return {'BLEU_4': avg_bleu_score[3], 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 'ROUGE': avg_rouge_score}