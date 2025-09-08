import argparse
import torch


def get_args(dataset) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    """
    Dataset setting
    """
    if dataset == 'msrvtt':
        parser.add_argument('--dataset', type=str, default='msrvtt')
        parser.add_argument('--video_file_path', type=str,
                            default='data/features/msrvtt/clip-vitL14')
        parser.add_argument('--caption_file_path', type=str,
                            default='data/sentences/inference/msrvtt/msr-vtt_test.json')
        parser.add_argument('--verb_file_path',type=str,default='data/sentences/inference/msrvtt/msr-vtt_test_verb.json')
        """
        ClipScore settings
        """
        parser.add_argument('--image_features_dir', type=str,
                            default='data/features/msrvtt/clip-vitB32')
    
    elif dataset == 'activitynet':
        parser.add_argument('--dataset', type=str, default='activitynet')
        parser.add_argument('--video_file_path', type=str,
                            default='data/features/activitynet/clip-vitL14')
        parser.add_argument('--caption_file_path', type=str,
                            default='data/sentences/inference/activitynet/activitynet_test.json')
        parser.add_argument('--verb_file_path',type=str,default='data/sentences/inference/activitynet/activitynet_test_verb.json')
        parser.add_argument('--image_features_dir', type=str,
                                default='data/features/activitynet/clip-vitB32')
    
    elif dataset == 'vatex':
        parser.add_argument('--dataset', type=str, default='vatex')
        parser.add_argument('--video_file_path', type=str,
                            default='data/features/vatex/clip-vitL14')
        parser.add_argument('--caption_file_path', type=str,
                            default='data/sentences/inference/vatex/vatex_test.json')
        parser.add_argument('--verb_file_path',type=str,default='data/sentences/inference/vatex/vatex_test_verb.json')
        parser.add_argument('--image_features_dir', type=str,
                                default='data/features/vatex/clip-vitB32')
    else:
        raise "error dataset"

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=False)

    """
    Traning settings
    """
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fabric_precision', type=str, default='bf16-mixed')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--torch_dtype', default=torch.float)
    parser.add_argument('--decoer_model', type=str, default='GPT2')

    """
    Val/Test settings
    """
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)

    """
    caption settings
    """
    parser.add_argument('--caption_seq_len', type=int, default=15)
    parser.add_argument('--topk_words', type=int, default=200)
    parser.add_argument('--topk_sents', type=int, default=20)
    parser.add_argument('--lambda_lm', type=float, default=2.0)
    parser.add_argument('--lambda_vid_sim', type=float, default=2.0)
    parser.add_argument('--lambda_text_sim', type=float, default=2.0)

    """
    Video settings
    """
    parser.add_argument('--video_seq_len', type=int, default=30)
    """
    Checkpoint setting
    """

    parser.add_argument('--project_checkpoint', type=str,
                        default="checkpoint/work2/msrvtt_20.pth")
    parser.add_argument('--text_feature_path',type=str,
                        default="pt/msrvtt_text_features.pt")
    """
    CLIP settings
    """
    parser.add_argument('--clip_text_out_size', type=int,
                        default=768, help='The output dimension of CLIP')
    """
    language model settings
    """
    parser.add_argument('--language_model_config', type=str,
                        default='gpt2-large')
    parser.add_argument('--language_model_weights', type=str,
                        default='gpt2-large')
    parser.add_argument('--lm_input_size', type=int,
                        default=1280, help='The input size of LM')
    parser.add_argument('--language_model_dtype',default=torch.bfloat16)

    """
    tokenizer settings
    """
    parser.add_argument('--tokenizer_path', type=str,
                        default='gpt2')
    parser.add_argument('--tokenizer_padding', type=str, default='max_length')
    parser.add_argument('--tokenizer_truncation', type=bool, default=True)
    parser.add_argument('--tokenizer_return_tensors', type=str, default='pt')
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--add_bos_token', type=bool, default=False)
    parser.add_argument('--add_eos_token', type=bool, default=False)

    """
    SummaryWriter settings
    """
    parser.add_argument('--inf_path', type=str,
                        default='output/inf_test.json')
    
    """
    SummaryWriter settings
    """
    parser.add_argument('--log_dir', type=str,
                        default='runs')
    
    args = parser.parse_args()
    return args
