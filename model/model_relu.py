import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from typing import Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import pickle

class LinearProjectedLM(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.decoer_model == 'Llama':
            self.language_model = LlamaForCausalLM.from_pretrained(
                args.language_model_weights).to(args.device)
            self.word_embedding = self.language_model.model.embed_tokens
            self.tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
        
        if args.decoer_model == 'GPT2':
            # with open('/home/wangtao/video_caption/project/decoder_config.pkl', 'rb') as f:
            #     config = pickle.load(f)
            # self.language_model = GPT2LMHeadModel(config)
            self.language_model = GPT2LMHeadModel.from_pretrained(
                args.language_model_weights).to(args.device)
            self.word_embedding = self.language_model.transformer.wte
            self.tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)

        # self.project = nn.Linear(args.clip_text_out_size, args.lm_input_size)
        self.project = MLP((args.clip_text_out_size, args.lm_input_size))
        self.caption_seq_len = args.caption_seq_len
        self.args = args

    def forward(self, input):
        output = self.language_model(inputs_embeds=input, return_dict=True)
        return output

    def generate(self, input_embed):
        generated_text = self.language_model.generate(
            inputs_embeds=input_embed, max_new_tokens=self.args.caption_seq_len, pad_token_id=self.tokenizer.eos_token_id, num_beams=5)
        # generated_text = self.language_model.generate(
        #     inputs_embeds=input_embed, max_new_tokens=self.args.caption_seq_len, pad_token_id=self.tokenizer.eos_token_id)
        return generated_text

    def get_tokenizer(self, args):
        """
        Note: using unk token as pad token
        """
        if args.decoer_model == 'Llama':
            # tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.add_bos_token = args.add_bos_token
            self.tokenizer.add_eos_token = args.add_eos_token
        elif args.decoer_model == 'GPT2':
            # tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.cls_token = self.tokenizer.eos_token
            self.tokenizer.sep_token = self.tokenizer.eos_token
        return self.tokenizer

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(sizes[0],sizes[1], bias=bias)
        self.fc2 = nn.Linear(sizes[1], sizes[1], bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x