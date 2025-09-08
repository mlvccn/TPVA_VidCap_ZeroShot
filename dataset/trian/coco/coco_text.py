import torch
import json
import torch.utils.data
import numpy as np
from icecream import ic


def coco_text_only_collate_fn(batch):
    return batch


def annotation_process(file_path):
    total_caption = []
    with open(file_path, mode='r') as f:
        annotations = json.load(f)['annotations']
    for annotation in annotations:
        text = annotation['caption']
        total_caption.append(text)
    return total_caption


class coco_text_only_dataset_train(torch.utils.data.Dataset):
    def __init__(self, args,) -> None:
        super().__init__()
        self.caption_file_path_train = args.caption_file_path_train
        self.caption_file_path_val = args.caption_file_path_val
        self.caption = annotation_process(
            self.caption_file_path_train) + annotation_process(self.caption_file_path_val)
        self.caption = self.caption[:-args.split]

    def __getitem__(self, index):
        return self.caption[index]

    def __len__(self,):
        return len(self.caption)


class coco_text_only_dataset_val(torch.utils.data.Dataset):
    def __init__(self, args,) -> None:
        super().__init__()
        self.caption_file_path_train = args.caption_file_path_train
        self.caption_file_path_val = args.caption_file_path_val
        self.caption = annotation_process(
            self.caption_file_path_train) + annotation_process(self.caption_file_path_val)
        self.caption = self.caption[-args.split:]

    def __getitem__(self, index):
        return self.caption[index]

    def __len__(self,):
        return len(self.caption)
