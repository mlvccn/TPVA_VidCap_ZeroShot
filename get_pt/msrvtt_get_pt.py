import os
import sys
import torch
import json
import clip
import random
from tqdm import tqdm

device = 'cuda:3'

clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
with open('data/sentences/trian/msrvtt/msrvtt_train.json', 'r') as f:
    data = json.load(f)

tokenizer = clip.tokenize
data = random.sample(data, 130000)
text_features = []
captions = []
batch_size = 1000
clip_model.eval()
for i in tqdm(range(0, len(data[:])//batch_size)):
    texts = data[i*batch_size:(i+1)*batch_size]
    with torch.no_grad():
        texts_token = tokenizer(texts).to(device)
        text_feature = clip_model.encode_text(texts_token)
        text_features.append(text_feature)
        captions.extend(texts)

text_features = torch.cat(text_features, dim=0)
text_features /= text_features.norm(dim=-1, keepdim=True).float()
torch.save(text_features, "msrvtt_text_features.pt")
