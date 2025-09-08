import torch
import json
import torch.utils.data
import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from icecream import ic

def msrvtt_text_only_collate_fn(batch):
    caption, verb = zip(*batch)
    return list(caption), list(verb)

def annotation_process(file_path):
    lemmatizer = WordNetLemmatizer()
    caption_verb = []

    candidate_tag = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    removing_word = ["is", "are", "was", "were",
                     "ha", "s", "S", f"'s", "being", "be"]
    verb_template = "There is an action \""
    verbs_template = "There are some actions \""
    
    with open(file_path, mode='r') as p:
        json_file = json.load(p)
        videos = json_file['videos']
        sentences = json_file['sentences']
        
    train_video_list = []
    for video_info in videos:
        if video_info['split'] == 'train':
            train_video_list.append(video_info['video_id'])
        elif video_info['split'] == 'validate':
            train_video_list.append(video_info['video_id'])
    
    total_caption = []
    for sentence in sentences:
        caption = sentence['caption']
        name = sentence['video_id']
        if name in train_video_list:
            total_caption.append(caption)
    
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
        caption_verb.append((caption, sentence))
    return caption_verb

class msrvtt_text_only_dataset_train(torch.utils.data.Dataset):
    def __init__(self, args,) -> None:
        super().__init__()
        self.caption_file_path = args.caption_file_path
        self.caption_verb_path = args.caption_verb_path
        
        # caption_verb = annotation_process(self.caption_file_path)
        # torch.save(caption_verb, self.caption_verb_path)

        self.caption_verb = json.load(open(self.caption_verb_path,mode='r'))
        self.caption_verb = self.caption_verb[:-args.split]

    def __getitem__(self, index):
        return self.caption_verb[index]

    def __len__(self,):
        return len(self.caption_verb)


class msrvtt_text_only_dataset_val(torch.utils.data.Dataset):
    def __init__(self, args,) -> None:
        super().__init__()
        self.caption_file_path = args.caption_file_path
        self.caption_verb_path = args.caption_verb_path

        self.caption_verb = json.load(open(self.caption_verb_path,mode='r'))
        self.caption_verb = self.caption_verb[-args.split:]

    def __getitem__(self, index):
        return self.caption_verb[index]

    def __len__(self,):
        return len(self.caption_verb)
