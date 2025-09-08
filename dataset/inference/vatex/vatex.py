import torch
import json
import torch.utils.data
import glob
import numpy as np
from icecream import ic


def vatex_train_collate_fn(batch):
    caption, vid_feat = zip(*batch)
    caption = list(caption)
    vid_feat = torch.cat(
        [torch.tensor(item, dtype=torch.float32).unsqueeze(0)for item in vid_feat], dim=0)
    return caption, vid_feat


def vatex_val_collate_fn(batch):
    caption, vid_feat = zip(*batch)
    caption = list(caption)
    vid_feat = torch.cat(
        [torch.tensor(item, dtype=torch.float32).unsqueeze(0)for item in vid_feat], dim=0)
    return caption, vid_feat


def annotation_process(video_path):
    """
    retrun a list of tuple (caption,name)
    """
    out = []
    with open(video_path, mode='r') as p:
        json_file = json.load(p)
    for item in json_file:
        name = item['videoID']
        captions = item['enCap']
        for caption in captions:
            out.append((caption, name))
    return out


class vatex_dataset_train(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.cap2path = []
        self.caption_length = args.caption_seq_len
        self.video_length = args.video_seq_len
        self.video_file_path = args.video_file_path
        self.caption_file_path = args.caption_file_path

        name2path = {}
        videoes_path_list = glob.glob(
            self.video_file_path + '/*.npy')[:args.split_numbers]
        for video_path in videoes_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            name2path[vid_name] = video_path
        caption_out = annotation_process(self.caption_file_path)
        for cap_name in caption_out:
            cap, name = cap_name
            if name in name2path:
                self.cap2path.append((cap, name2path[name]))

    def __getitem__(self, index):
        cur_cap2path = self.cap2path[index]
        cap, path = cur_cap2path
        vid_feat = np.load(path)
        return cap, vid_feat

    def __len__(self,):
        return len(self.cap2path)


class vatex_dataset_val(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.caption_length = args.caption_seq_len
        self.video_length = args.video_seq_len
        self.video_file_path = args.video_file_path
        self.caption_file_path = args.caption_file_path

        self.name2path = {}
        self.name2cap = {}
        self.namelist = []
        videoes_path_list = glob.glob(
            self.video_file_path + '/*.npy')[args.split_numbers:]
        for video_path in videoes_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            self.name2path[vid_name] = video_path
        caption_out = annotation_process(self.caption_file_path)  # cap,name
        for cap_name in caption_out:
            cap, name = cap_name
            if name in self.name2path.keys():
                if name in self.name2cap.keys():
                    self.name2cap[name].append(cap)
                else:
                    self.name2cap[name] = []
                    self.name2cap[name].append(cap)

        self.namelist = list(self.name2cap.keys())

    def __getitem__(self, index):
        cur_name = self.namelist[index]
        path = self.name2path[cur_name]
        vid_feat = np.load(path)
        cap = self.name2cap[cur_name]
        return cap, vid_feat

    def __len__(self,):
        return len(self.namelist)
