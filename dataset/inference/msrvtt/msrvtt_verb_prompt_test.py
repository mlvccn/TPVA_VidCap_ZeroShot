import torch
import json
import torch.utils.data
import glob
import os
import numpy as np
from icecream import ic

def msrvtt_test_collate_fn(batch):
    name, batched_caption,verb_prompt,vid_feat = zip(*batch)
    name = list(name)
    batched_caption = list(batched_caption)
    verb_prompt = list(verb_prompt)
    vid_feat = torch.cat([torch.tensor(item,dtype=torch.float32).unsqueeze(0)for item in vid_feat],dim=0)
    return name,batched_caption,verb_prompt,vid_feat

def annotation_process(caption_path):
    """
    return a list of tuple -> (caption,name) / train_video_id_list / val_video_id_list
    """
    with open(caption_path,mode='r') as p:
        json_file = json.load(p)
        videos = json_file['videos']
        sentences = json_file['sentences']
    name_list = []
    name2cap = {}
    for video_info in videos:
        if video_info['split'] == 'test':
            name_list.append(video_info['video_id'])
    for sentence in sentences:
        caption = sentence['caption']
        name = sentence['video_id']
        if name in name2cap.keys():
            name2cap[name].append(caption)
        else:
            name2cap[name] = [caption]
    return name2cap,name_list

class msrvtt_dataset_test(torch.utils.data.Dataset):
    def __init__(self,args) -> None:
        super().__init__()
        self.caption_length = args.caption_seq_len
        self.video_length = args.video_seq_len
        
        self.video_file_path = args.video_file_path
        self.caption_file_path = args.caption_file_path
        self.verb_file_path = args.verb_file_path
        
        with open(self.verb_file_path,mode='r') as f:
            self.name2verb = json.load(f)
        self.name2cap,self.name_list = annotation_process(self.caption_file_path)
        self.name2path = {}
        videoes_path_list = glob.glob(self.video_file_path + '/*.npy')
        for video_path in videoes_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            if vid_name in self.name_list:
                self.name2path[vid_name] = video_path
        
        
    def __getitem__(self, index):
        name = self.name_list[index]
        batched_cap = self.name2cap[name]
        verb = self.name2verb[name]
        verb_prompt = "There is an action \"" + verb +"\" in the video."
        # verb_prompt = self.prompt.format(','.join(list(verb)))
        vid_feat = np.load(self.name2path[name])
        return name,batched_cap,verb_prompt,vid_feat
    
    def __len__(self,):
        return len(self.name_list)