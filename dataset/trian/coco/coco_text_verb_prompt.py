import torch
import json
import torch.utils.data
import numpy as np
import nltk
import pickle
import os
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from icecream import ic


def coco_text_only_collate_fn(batch):
    caption, verb = zip(*batch)
    return list(caption), list(verb)


def annotation_process(file_path, prompt: str):
    lemmatizer = WordNetLemmatizer()

    caption_verb = []
    total_caption = []
    with open(file_path, mode='r') as f:
        annotations = json.load(f)['annotations']

    for annotation in annotations:
        text = annotation['caption']
        total_caption.append(text)

    candidate_tag = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    removing_word = ["is", "are", "was", "were",
                     "ha", "s", "S", f"'s", "being", "be"]
    verb_template = "There is an action \""
    verbs_template = "There are some actions \""

    for caption in tqdm(total_caption):
        cur_caption_verb = []
        pos_tags = nltk.pos_tag(nltk.word_tokenize(caption))
        for word, tag in pos_tags:
            if tag in candidate_tag:
                verb = lemmatizer.lemmatize(word.lower().strip())
                if verb not in removing_word:
                    cur_caption_verb.append(verb)

        if (len(cur_caption_verb) == 0):
            sentence = "There is no action in the video"
        elif (len(cur_caption_verb) == 1):
            sentence = verb_template + \
                str(cur_caption_verb[0]) + "\" in the video."
        else:
            sentence = verbs_template
            for i, verb in enumerate(cur_caption_verb):
                sentence += str(verb)
                if i < len(cur_caption_verb) - 1:
                    sentence += ", "
            sentence += "\" in the video."
        # sentence = prompt.format(','.join(cur_caption_verb))
        caption_verb.append((caption, sentence))
    return caption_verb


class coco_text_only_dataset_train(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.caption_file_path_train = args.caption_file_path_train
        self.caption_file_path_val = args.caption_file_path_val
        self.caption_verb_path = args.caption_verb_path
        self.prompt = args.prompt

        # if not os.path.exists(self.caption_verb_path):
        caption_verb_train = annotation_process(self.caption_file_path_train, self.prompt)
        caption_verb_val = annotation_process(self.caption_file_path_val, self.prompt)
        caption_verb_all = caption_verb_train + caption_verb_val

        #     torch.save(caption_verb_all, self.caption_verb_path)
        # else:
            # self.caption_verb = json.load(open(self.caption_verb_path,mode='r'))
        self.caption_verb = caption_verb_all

        self.caption_verb = self.caption_verb[:-args.split]
        self.caption_verb = self.caption_verb[:int(len(self.caption_verb) * args.percent)]

    def __getitem__(self, index):
        return self.caption_verb[index]

    def __len__(self,):
        return len(self.caption_verb)


class coco_text_only_dataset_val(torch.utils.data.Dataset):
    def __init__(self, args,) -> None:
        super().__init__()
        self.caption_file_path_train = args.caption_file_path_train
        self.caption_file_path_val = args.caption_file_path_val
        self.caption_verb_path = args.caption_verb_path

        self.caption_verb = json.load(open(self.caption_verb_path,mode='r'))

        self.caption_verb = self.caption_verb[-args.split:]

    def __getitem__(self, index):
        return self.caption_verb[index]

    def __len__(self,):
        return len(self.caption_verb)
