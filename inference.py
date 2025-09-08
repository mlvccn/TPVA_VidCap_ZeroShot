import torch
import os
import sys
import clip
import json
from icecream import ic
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# from peft import PeftConfig, get_peft_model, PeftModel
from utils.inference_setting import get_args
from dataset.inference.vatex.vatex_verb_prompt_test import vatex_dataset_test,vatex_test_collate_fn
from dataset.inference.msrvtt.msrvtt_verb_prompt_test import msrvtt_dataset_test,msrvtt_test_collate_fn
from dataset.inference.activitynet.activitynet_verb_prompt_test import activitynet_dataset_test,activitynet_test_collate_fn
from model.model_relu import LinearProjectedLM
from eval.eval import language_eval
from collections import OrderedDict

@torch.no_grad()
def inference(text_features, model, clip_model, tokenizer, dataloader, args, writer, i):
    model.eval()
    clip_model.eval()
    bos = '<s>'
    batched_pred = []
    batched_groundtruth = []
    bos_token_embeds = model.word_embedding(
        tokenizer(bos, return_tensors='pt')['input_ids'].to(args.device))
    
    references, predictions = OrderedDict(), OrderedDict()
    for name, caption, prefix_verb, vid_feat in tqdm(dataloader):
    # for name, caption, prefix_verb, vid_feat in dataloader:
        # 所有语句一次性生成
        vid_feat = vid_feat.mean(dim=1).to(args.device).float()  # (10, 768)
        vid_feat /= vid_feat.norm(dim=-1, keepdim=True)
        sim = vid_feat@text_features.T.float()
        sim = (sim*100).softmax(dim=1)  # (10, 550000)
        zero = torch.zeros(vid_feat.size()).to(args.device)
        for k in range(args.topk_sents):
            _, max_id = torch.max(sim, dim=1)
            text_features_list = []
            for index, id in enumerate(max_id):
                text_features_list.append(text_features[id].unsqueeze(0))
                sim[index][id] = 0
            text_features_tensor = torch.cat(
                text_features_list, dim=0).to(args.device)
            zero += text_features_tensor

        zero /= zero.norm(dim=-1, keepdim=True)

        clip_encoded_verb = clip_model.encode_text(
            clip.tokenize(prefix_verb).to(args.device)).float()
        clip_encoded_verb /= clip_encoded_verb.norm(
            dim=-1, keepdim=True)
        verb_embeds = model.project(
            clip_encoded_verb).unsqueeze(1)

        vid_embeds = model.project(zero).unsqueeze(1)
        batch_size = vid_embeds.size(0)
        if args.decoer_model == 'Llama':
            input_embeds = torch.cat(
                (verb_embeds,vid_embeds, bos_token_embeds.expand((batch_size, -1, -1))), dim=1)
        elif args.decoer_model == 'GPT2':
            input_embeds = torch.cat(
                    (verb_embeds,vid_embeds), dim=1)
            
        generated_caption = model.generate(input_embeds)
        generated_str = tokenizer.batch_decode(
            generated_caption, skip_special_tokens=True)
        for idx in range(len(generated_str)):
            str_list = []
            string = generated_str[idx]
            str_list.append(string)
            predictions[name[idx]] = str_list
            references[name[idx]] = caption[idx]

    print('epoch:', i)
    scores = language_eval(predictions, references, args)
    for key in scores.keys():
        writer.add_scalar('test/'+key, float(scores[key]), 0)
    
    json.dump(predictions, open(args.inf_path, "w"))
    
    torch.cuda.empty_cache()

def main():
    args = get_args('msrvtt')
    clip_model, _ = clip.load("ViT-L/14", device=args.device, jit=False)
    for param in clip_model.parameters():
        param.requires_grad = False
    
    writer = SummaryWriter(log_dir=args.log_dir+"/"+args.project_checkpoint.split('/')[-1][:-4])

    text_features = torch.load(
        args.text_feature_path, map_location=torch.device(args.device))
    model = LinearProjectedLM(args)
    model = model.to(args.device)

    for param in model.parameters():
        param.requires_grad = False

    tokenizer = model.get_tokenizer(args)
    if args.dataset == 'msrvtt':
        test_dataset = msrvtt_dataset_test(args)
        test_dataloader = DataLoader(test_dataset, args.val_batch_size, args.shuffle,
                                    collate_fn=msrvtt_test_collate_fn, num_workers=args.num_workers)
    elif args.dataset == 'vatex':
        test_dataset = vatex_dataset_test(args)
        test_dataloader = DataLoader(test_dataset, args.val_batch_size, args.shuffle,
                                    collate_fn=vatex_test_collate_fn, num_workers=args.num_workers)
    
    elif args.dataset == 'activitynet':
        test_dataset = activitynet_dataset_test(args)
        test_dataloader = DataLoader(test_dataset, args.val_batch_size, args.shuffle,
                                    collate_fn=activitynet_test_collate_fn, num_workers=args.num_workers)
    
    check_point_path = args.project_checkpoint
    model.project.load_state_dict(torch.load(check_point_path, map_location=torch.device(args.device)))
    inference(text_features, model, clip_model, tokenizer,
            test_dataloader, args, writer, 1)

if __name__ == '__main__':
    main()
