import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from icecream import ic
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils.pretraining_settings import get_args
from dataset.trian.coco.coco_text_verb_prompt import coco_text_only_dataset_train, coco_text_only_dataset_val, coco_text_only_collate_fn
from dataset.trian.activitynet.activitynet_text_verb_prompt import activitynet_text_only_dataset_train, activitynet_text_only_dataset_val, activitynet_text_only_collate_fn
from dataset.trian.msrvtt.msrvtt_text_verb_prompt import msrvtt_text_only_dataset_train, msrvtt_text_only_dataset_val, msrvtt_text_only_collate_fn
from dataset.trian.vatex.vatex_text_verb_prompt import vatex_text_only_dataset_train, vatex_text_only_dataset_val, vatex_text_only_collate_fn
from model.model_relu import LinearProjectedLM

def pretraining(model, clip_model, optimizer, tokenizer, loss_fn, train_dataloader, val_dataloader, args, writer):
    total_step = 0
    clip_model.eval()

    for epoch in tqdm(range(args.epoch)):
        model.train()
        for caption,verb_sentence in tqdm(train_dataloader, position=1):
            total_step += 1
            tokenized_caption = tokenizer(caption,
                                          padding=True,
                                          truncation=args.tokenizer_truncation,
                                          max_length=args.caption_seq_len,
                                          return_tensors=args.tokenizer_return_tensors).to(args.device)
            token_ids = tokenized_caption['input_ids']  # (64, 20)
            batch_size = token_ids.size(0)  # 64
            tokens_embeds = model.word_embedding(token_ids)  # (64, 20, 1280)
            
            clip_encoded_text = clip_model.encode_text(
                clip.tokenize(caption, truncate=True).to(args.device)).float()  # N*H (64, 768)
            clip_encoded_text /= clip_encoded_text.norm(dim=-1, keepdim=True) # (64, 768)
            clip_encoded_text = model.project(
                clip_encoded_text).unsqueeze(1)  # (64, 1, 1280)
            
            clip_encoded_verb = clip_model.encode_text(
                clip.tokenize(verb_sentence).to(args.device)).float() #(64, 768)
            clip_encoded_verb /= clip_encoded_verb.norm(dim=-1, keepdim=True)
            clip_encoded_verb = model.project(
                clip_encoded_verb).unsqueeze(1) # (64, 1, 1280)

            if args.decoer_model == 'Llama':
                actual_input_embeds = torch.cat(
                    (clip_encoded_verb, clip_encoded_text, tokens_embeds[:, :-1]), dim=1)  # (64, 20, 1280)
                logits = model(actual_input_embeds).logits  # (64, 20, 32000)
                loss = loss_fn(logits[:, 2:].reshape(-1, args.vocab_size), token_ids[:, 1:].reshape(-1),)
            elif args.decoer_model == 'GPT2':
                actual_input_embeds = torch.cat(
                    (clip_encoded_verb, clip_encoded_text, tokens_embeds), dim=1) # (64, 22, 1280)
                logits = model(actual_input_embeds).logits[:, 1: -1] # (64, 20, 32000)
                loss = loss_fn(logits.reshape(-1, args.vocab_size), token_ids.reshape(-1),)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', float(loss), total_step)

        if int(epoch) % 1 == 0:
            folder_path = os.path.join(str(args.save_path),str(args.prompt),)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, str(args.dataset)+"_"+str(int(epoch)+1)+'.pth')
            torch.save(model.project.state_dict(
            ), file_path)

def main():
    args = get_args('coco')
    model = LinearProjectedLM(args)
    model = model.to(args.device)

    # 创建SummaryWriter对象
    writer = SummaryWriter(log_dir=args.log_dir)

    clip_model, _ = clip.load("ViT-L/14", device=args.device, jit=False)
    for param in clip_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,)
    tokenizer = model.get_tokenizer(args)
    loss_fn = nn.CrossEntropyLoss()
    if args.dataset == 'coco':
        train_dataset = coco_text_only_dataset_train(args)
        val_dataset = coco_text_only_dataset_val(args)
        train_dataloader = DataLoader(train_dataset, args.train_batch_size, args.shuffle,
                                      collate_fn=coco_text_only_collate_fn, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, args.val_batch_size, args.shuffle,
                                    collate_fn=coco_text_only_collate_fn, num_workers=args.num_workers)
    elif args.dataset == 'vatex':
        train_dataset = vatex_text_only_dataset_train(args)
        val_dataset = vatex_text_only_dataset_val(args)
        train_dataloader = DataLoader(train_dataset, args.train_batch_size, args.shuffle,
                                      collate_fn=vatex_text_only_collate_fn, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, args.val_batch_size, args.shuffle,
                                    collate_fn=vatex_text_only_collate_fn, num_workers=args.num_workers)
    elif args.dataset == 'activitynet':
        train_dataset = activitynet_text_only_dataset_train(args)
        val_dataset = activitynet_text_only_dataset_val(args)
        train_dataloader = DataLoader(train_dataset, args.train_batch_size, args.shuffle,
                                      collate_fn=activitynet_text_only_collate_fn, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, args.val_batch_size, args.shuffle,
                                    collate_fn=activitynet_text_only_collate_fn, num_workers=args.num_workers)
    elif args.dataset == 'msrvtt':
        train_dataset = msrvtt_text_only_dataset_train(args)
        val_dataset = msrvtt_text_only_dataset_val(args)
        train_dataloader = DataLoader(train_dataset, args.train_batch_size, args.shuffle,
                                      collate_fn=msrvtt_text_only_collate_fn, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, args.val_batch_size, args.shuffle,
                                    collate_fn=msrvtt_text_only_collate_fn, num_workers=args.num_workers)
    else:
        raise "error dataset"
    
    pretraining(model, clip_model, optimizer, tokenizer, loss_fn,
                train_dataloader, val_dataloader, args, writer)


if __name__ == '__main__':
    main()
