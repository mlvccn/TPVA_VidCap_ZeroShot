import torch
import json
import torch.utils.data
import glob
import numpy as np
from icecream import ic


def msrvtt_train_collate_fn(batch):
    caption, vid_feat = zip(*batch)
    caption = list(caption)
    vid_feat = torch.cat(
        [torch.tensor(item, dtype=torch.float32).unsqueeze(0)for item in vid_feat], dim=0)
    return caption, vid_feat


def msrvtt_val_collate_fn(batch):
    caption, vid_feat = zip(*batch)
    caption = list(caption)
    vid_feat = torch.cat(
        [torch.tensor(item, dtype=torch.float32).unsqueeze(0)for item in vid_feat], dim=0)
    return caption, vid_feat

def msrvtt_test_collate_fn(batch):
    caption, vid_feat = zip(*batch)
    caption = list(caption)
    vid_feat = torch.cat(
        [torch.tensor(item, dtype=torch.float32).unsqueeze(0)for item in vid_feat], dim=0)
    return caption, vid_feat


def annotation_process(caption_path):
    """
    return a list of tuple -> (caption,name) / train_video_id_list / val_video_id_list / test_video_id_list
    """
    with open(caption_path, mode='r') as p:
        json_file = json.load(p)
        videos = json_file['videos']
        sentences = json_file['sentences']
    train_video_list = []
    val_video_list = []
    test_video_list = []
    cap_name = []
    for video_info in videos:
        if video_info['split'] == 'train':
            train_video_list.append(video_info['video_id'])
        elif video_info['split'] == 'validate':
            val_video_list.append(video_info['video_id'])
        elif video_info['split'] == 'test':
            test_video_list.append(video_info['video_id'])
    for sentence in sentences:
        caption = sentence['caption']
        name = sentence['video_id']
        cap_name.append((caption, name))
    return cap_name, train_video_list, val_video_list, test_video_list

class msrvtt_dataset_train(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.cap2path = []
        self.caption_length = args.caption_seq_len
        self.video_length = args.video_seq_len
        self.video_file_path = args.video_file_path
        self.caption_file_path = args.caption_file_path

        caption_out, train_video_list, _, _ = annotation_process(
            self.caption_file_path)
        name2path = {}
        videoes_path_list = glob.glob(self.video_file_path + '/*.npy')
        for video_path in videoes_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            if vid_name in train_video_list:
                name2path[vid_name] = video_path
        for cap_name in caption_out:
            cap, name = cap_name
            if name in name2path.keys():
                self.cap2path.append((cap, name2path[name]))

    def __getitem__(self, index):
        cur_cap2path = self.cap2path[index]
        cap, path = cur_cap2path
        vid_feat = np.load(path)
        return cap, vid_feat

    def __len__(self,):
        return len(self.cap2path)


class msrvtt_dataset_val(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.caption_length = args.caption_seq_len
        self.video_length = args.video_seq_len
        self.video_file_path = args.video_file_path
        self.caption_file_path = args.caption_file_path

        caption_out, _, val_video_list, _ = annotation_process(
            self.caption_file_path)  # cap,name
        self.name2path = {}
        self.name2cap = {}
        self.namelist = []
        videoes_path_list = glob.glob(self.video_file_path + '/*.npy')
        for video_path in videoes_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            if vid_name in val_video_list:
                self.name2path[vid_name] = video_path
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


class msrvtt_dataset_test(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.caption_length = args.caption_seq_len
        self.video_length = args.video_seq_len
        self.video_file_path = args.video_file_path
        self.caption_file_path = args.caption_file_path

        caption_out, _, _, test_video_list = annotation_process(
            self.caption_file_path)  # cap, name
        self.name2path = {}
        self.name2cap = {}
        self.namelist = []
        videoes_path_list = glob.glob(self.video_file_path + '/*.npy')
        for video_path in videoes_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            if vid_name in test_video_list:
                self.name2path[vid_name] = video_path
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
