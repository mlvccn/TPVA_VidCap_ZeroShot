import torch
import json
import torch.utils.data
import glob
import numpy as np
from icecream import ic

def vatex_test_collate_fn(batch):
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
    name2cap = {}
    name_list = []
    with open(caption_path, mode='r') as p:
        json_file = json.load(p)
    for item in json_file:
        captions = item['enCap']
        name = item['videoID']
        name_list.append(name)
        name2cap[name] = captions
    return name2cap,name_list

class vatex_dataset_test(torch.utils.data.Dataset):
    def __init__(self,args) -> None:
        super().__init__()
        self.caption_length = args.caption_seq_len
        self.video_length = args.video_seq_len
        
        self.video_file_path = args.video_file_path
        self.caption_file_path = args.caption_file_path
        self.verb_file_path = args.verb_file_path
        
        with open(self.verb_file_path,mode='r') as f:
            self.name2verb = json.load(f)
        self.name2cap, name_list_1 = annotation_process(self.caption_file_path)
        self.name2path = {}
        name_list = []
        videoes_path_list = glob.glob(self.video_file_path + '/*.npy')
        for video_path in videoes_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            if vid_name in name_list_1:
                name_list.append(vid_name)
                self.name2path[vid_name] = video_path
        self.name_list = name_list
        
    def __getitem__(self, index):
        name = self.name_list[index]
        batched_cap = self.name2cap[name]
        verb = self.name2verb[name]
        verb_prompt = "There is an action \"" + verb +"\" in the video."
        vid_feat = np.load(self.name2path[name])
        return name, batched_cap,verb_prompt,vid_feat
    
    def __len__(self,):
        return len(self.name_list)