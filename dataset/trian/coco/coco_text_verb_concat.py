import torch
import json
import torch.utils.data
import numpy as np
import nltk
import pickle
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from icecream import ic

def coco_text_only_collate_fn(batch):
    caption,verb = zip(*batch)
    return list(caption),list(verb)

def annotation_process(file_path):
    lemmatizer = WordNetLemmatizer()
    
    caption_verb = []
    total_caption = []
    with open(file_path, mode='r') as f:
        annotations = json.load(f)['annotations']
        
    for annotation in annotations:
        text = annotation['caption']
        total_caption.append(text)
    
    candidate_tag = ["VB","VBD","VBG","VBN","VBP","VBZ"]
    removing_word = ["is","are","was","were","ha","s","S",f"'s","being","be"]
    for caption in tqdm(total_caption):
        cur_caption_verb = []
        pos_tags = nltk.pos_tag(nltk.word_tokenize(caption))
        for word,tag in pos_tags:
            if tag in candidate_tag:
                verb = lemmatizer.lemmatize(word.lower().strip())
                if verb not in removing_word:
                    cur_caption_verb.append(verb)
        sentence = ""
        for i,verb in enumerate(cur_caption_verb):
            sentence += str(verb)
            if i < len(cur_caption_verb) - 1:
                sentence += ", "
        caption_verb.append((caption,sentence))
    return caption_verb


class coco_text_only_dataset_train(torch.utils.data.Dataset):
    def __init__(self, args,) -> None:
        super().__init__()
        self.caption_file_path_train = args.caption_file_path_train
        self.caption_file_path_val = args.caption_file_path_val
        self.caption_verb_path_train = args.caption_verb_path_train
        self.caption_verb_path_val = args.caption_verb_path_val
        
        # caption_verb_train = annotation_process(self.caption_file_path_train)
        # caption_verb_val = annotation_process(self.caption_file_path_val)
        
        # torch.save(caption_verb_train,self.caption_verb_path_train)
        # torch.save(caption_verb_val, self.caption_verb_path_val)
        
        caption_verb_train = torch.load(self.caption_verb_path_train)
        caption_verb_val = torch.load(self.caption_verb_path_val)
        
        self.caption_verb = caption_verb_train + caption_verb_val
        
        self.caption_verb = self.caption_verb[:-args.split]

    def __getitem__(self, index):
        return self.caption_verb[index]

    def __len__(self,):
        return len(self.caption_verb)


class coco_text_only_dataset_val(torch.utils.data.Dataset):
    def __init__(self, args,) -> None:
        super().__init__()
        self.caption_file_path_train = args.caption_file_path_train
        self.caption_file_path_val = args.caption_file_path_val
        self.caption_verb_path_train = args.caption_verb_path_train
        self.caption_verb_path_val = args.caption_verb_path_val

        caption_verb_train = torch.load(self.caption_verb_path_train)
        caption_verb_val = torch.load(self.caption_verb_path_val)
        
        self.caption_verb = caption_verb_train + caption_verb_val
        
        self.caption_verb = self.caption_verb[-args.split:]

    def __getitem__(self, index):
        return self.caption_verb[index]

    def __len__(self,):
        return len(self.caption_verb)
