import argparse
import torch


def get_args(dataset):
    parser = argparse.ArgumentParser()

    """
    Dataset setting
    """
    if dataset == 'coco':
        parser.add_argument('--dataset', type=str, default='coco')
        parser.add_argument('--caption_file_path_train', type=str,
                            default='data/sentences/trian/coco/coco_train.json')
        parser.add_argument('--caption_file_path_val', type=str,
                            default='data/sentences/trian/coco/coco_val.json')
        parser.add_argument('--caption_verb_path',type=str,default="data/sentences/trian/coco/coco_verb_prompt.json")
        parser.add_argument('--split', type=int, default=3000)
    elif dataset == 'vatex':
        parser.add_argument('--dataset', type=str, default='vatex')
        parser.add_argument('--caption_file_path_train', type=str,
                            default='data/sentences/trian/vatex/vatex_train.json')
        parser.add_argument('--caption_file_path_val', type=str,
                            default='data/sentences/trian/vatex/vatex_val.json')
        parser.add_argument('--caption_verb_path',type=str,default="data/sentences/trian/vatex/vatex_verb_prompt.json")
        parser.add_argument('--split', type=int, default=3000)
    elif dataset == 'activitynet':
        parser.add_argument('--dataset',type=str,default='activitynet')
        parser.add_argument('--caption_file_path',type=str,default="data/sentences/trian/activitynet/activitynet_train.json")
        parser.add_argument('--caption_verb_path',type=str,default="data/sentences/trian/activitynet/activitynet_verb_prompt.json")
        parser.add_argument('--split',type=int,default=3000)
    elif dataset == 'msrvtt':
        parser.add_argument('--dataset', type=str, default='msrvtt')
        parser.add_argument('--caption_file_path', type=str,
                            default='data/sentences/trian/msrvtt/msrvtt_train.json')
        parser.add_argument('--caption_verb_path', type=str,
                            default='data/sentences/trian/msrvtt/msrvtt_verb_prompt.json')
        parser.add_argument('--split', type=int, default=3000)
    else:
        raise "error dataset"
    
    parser.add_argument('--prompt',type=str,default='{}')

    parser.add_argument('--percent', type=float,default=1.0)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=True)

    """
    Traning settings
    """
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--fabric_precision', type=str, default='bf16-mixed')
    parser.add_argument('--torch_dtype', default=torch.float)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--decoer_model', type=str, default='GPT2')

    """
    Val/Test settings
    """
    parser.add_argument('--val_batch_size', type=int, default=128)

    """
    caption settings
    """
    parser.add_argument('--caption_seq_len', type=int, default=20)

    """
    CLIP settings
    """
    parser.add_argument('--clip_text_out_size', type=int,
                        default=768, help='The output dimension of CLIP')
    """
    language model settings
    """
    parser.add_argument('--language_model_weights', type=str,
                        default='gpt2-large')
    parser.add_argument('--lm_input_size', type=int,
                        default=1280, help='The input size of LM')
    parser.add_argument('--language_model_dtype', default=torch.bfloat16)

    """
    tokenizer settings
    """
    parser.add_argument('--tokenizer_path', type=str,
                        default='gpt2-large')
    parser.add_argument('--tokenizer_padding', type=str, default='max_length')
    parser.add_argument('--tokenizer_truncation', type=bool, default=True)
    parser.add_argument('--tokenizer_return_tensors', type=str, default='pt')
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--add_bos_token', type=bool, default=True)
    parser.add_argument('--add_eos_token', type=bool, default=True)

    """
    save settings
    """
    parser.add_argument('--save_path', type=str,
                        default='checkpoint/coco')
    parser.add_argument('--ans_path', type=str,
                        default='out/train_test.txt')

    """
    SummaryWriter settings
    """
    parser.add_argument('--log_dir', type=str,
                        default='runs')

    args = parser.parse_args()
    return args
