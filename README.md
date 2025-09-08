## Temporal Prompt Guided Visual-Text-Object Alignment for Zero-Shot Video Captioning

This code is the official PyTorch implementation of the paper: Temporal Prompt Guided Visual-Text-Object Alignment for Zero-Shot Video Captioning

## 🔍 Introduction

Temporal Prompt guided Visual-text-object Alignment **(TPVA)**. It consists of the temporal prompt guidance module and the visual-text-object alignment module. The former employs the pre-trained action recognition model to yield the action class as the key word of the temporal prompt, which guides the LLM to generate the text phrase containing the verb identifying action. The latter implements both visual-text alignment and text-object alignment by computing their similarity scores, respectively, which allows the model to generate the words better revealing the video semantics.

<div align="center">
<img alt="Logo" src="figures/motivation.png" width="100%"/>
</div>

## 🚀 Quickstart

1. Requirements

We use conda to control Python virtual environment. Install the dependencies with the following command:

```bash
conda create -n tpva python=3.10 # if using conda to control virtual environment
conda activate tpva
pip install -r requirements.txt

pip install ftfy regex tqdm # to install openai-clip
pip install git+https://github.com/openai/CLIP.git
```

1. Data preparation

Download the dataset from the official URL provided in the paper. For [MSR-VTT](https://www.kaggle.com/datasets/vishnutheepb/msrvtt?resource=download), [VATEX](https://eric-xw.github.io/vatex-website/about.html) and [ActivityNet](http://activity-net.org/download.html). Place the downloaded data under the folder `./data/videos`.

We provide processed annotations and extracted video features for each dataset, which can be found [Here](https://pan.baidu.com/s/1rxuB5yijWak_01zUsZdd-g?pwd=fkbx).

Please uncompress the download files and place them in `./data/`

2. Download pycocoevaltools (Optional)

You can downlaod pycocoevaltools from the same link we provided. Please place it under `./eval/`

3. Training

- To see the model structure of TPVA, [click here](./model/model_relu.py).
- The configurations are [here](./utils). You can change settings here.

```bash
python pretraining.py
```

4. Evaluation

Before evaluation, you need to extract text features of the specific dataset. You can find codes [Here](./get_pt/).

```bash
python inference.py
```

## 📊 Results

### In-domain results on MSR-VTT and VATEX

| Models      | Setting    | Venue     | MSR-VTT B@4 | MSR-VTT M | MSR-VTT C | MSR-VTT CL-S^Ref | MSR-VTT CL-S | VATEX B@4 | VATEX M | VATEX C | VATEX CL-S^Ref | VATEX CL-S |
|-------------|------------|-----------|-------------|-----------|-----------|------------------|--------------|-----------|---------|---------|----------------|------------|
| ORG-TRL     | Supervised | CVPR’20   | 43.6        | 28.8      | 50.9      | -                | -            | 32.1      | 28.8    | 50.9    | -              | -          |
| HRNAT       | Supervised | TIP’21    | 42.1        | 28.0      | 48.2      | -                | -            | 32.5      | 22.3    | 50.7    | -              | -          |
| MAN         | Supervised | TMM’23    | 41.3        | 28.0      | 49.8      | -                | -            | 32.7      | 22.4    | 48.9    | -              | -          |
| CMGNet      | Supervised | CVIU’24   | 43.6        | 29.1      | 54.6      | -                | -            | -         | -       | -       | -              | -          |
| ASGMet      | Supervised | CVIU’25   | 43.4        | 29.6      | 52.6      | -                | -            | -         | -       | -       | -              | -          |
| KG-VCN      | Supervised | PR’25     | 45.0        | 28.7      | 51.9      | -                | -            | 33.3      | 22.9    | 53.3    | -              | -          |
| CLIPRe      | Zero-shot  | arXiv’22  | 10.2        | 18.8      | 19.9      | 83.5             | 85.2         | 11.1      | 17.0    | 27.1    | 83.5           | 87.7       |
| DeCap       | Zero-shot  | ICLR’23   | 23.1        | 23.6      | 34.8      | 82.3             | 77.0         | 21.3      | 20.7    | 43.1    | 83.4           | 82.4       |
| DeCap-VD    | Zero-shot  | ICLR’23   | 5.9         | 16.3      | 10.2      | 76.1             | 69.7         | 7.4       | 12.9    | 13.8    | 73.2           | 73.3       |
| DeCap-NND   | Zero-shot  | ICLR’23   | 13.1        | 20.2      | 24.4      | 80.5             | 77.1         | 14.8      | 18.1    | 32.4    | 80.9           | 81.1       |
| Knight      | Zero-shot  | IJCAI’23  | 25.4        | 28.0      | 31.9      | 81.3             | 78.1         | 19.0      | 20.3    | 27.7    | 81.2           | 78.1       |
| IFCap       | Zero-shot  | EMNLP’24  | 27.1        | 25.9      | 38.9      | 81.6             | 77.9         | 17.1      | 18.1    | 32.5    | 80.8           | 79.2       |
| Ours        | Zero-shot  | -         | 24.4        | 25.0      | 37.2      | 83.8             | 77.8         | 22.4      | 22.2    | 44.6    | 81.9           | 81.6       |

### In-domain results on ActivityNet

| Models     | Setting    | Venue     | B@4 | M   | C   | CL-S^Ref | CL-S |
|------------|------------|-----------|-----|-----|-----|----------|------|
| PDVC       | Supervised | CVPR’21   | 11.8| 15.9| 27.3| -        | -    |
| Reasoner   | Supervised | CVPR’22   | 12.5| 16.4| 30.0| -        | -    |
| VLTinT     | Supervised | AAAI’23   | 14.5| 17.9| 31.1| -        | -    |
| CLIPRe     | Zero-shot  | arXiv’22  | 1.4 | 8.2 | 15.2| 83.0     | 87.1 |
| DeCap      | Zero-shot  | ICLR’23   | 2.3 | 9.4 | 20.6| 76.7     | 79.7 |
| DeCap-VD   | Zero-shot  | ICLR’23   | 1.1 | 6.6 | 10.2| 68.2     | 71.2 |
| DeCap-NND  | Zero-shot  | ICLR’23   | 1.9 | 8.3 | 15.5| 74.5     | 77.5 |
| Knight     | Zero-shot  | IJCAI’24  | 3.7 |13.1 | 11.8| 67.9     | 66.1 |
| IFCap      | Zero-shot  | EMNLP’24  | 3.9 |14.4 | 20.3| 71.0     | 77.3 |
| Ours       | Zero-shot  | -         | 4.1 |14.6 | 24.5| 77.2     | 78.8 |


## 📄 Citation

If you find this repo useful, please cite the following paper.

```bibtex
@article{Temporal Prompt Guided Visual-Text-Object Alignment for Zero-Shot Video Captioning,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  pages     = {},
  year      = {2025}
}
```

## 📄 License

This project is licensed under the Apache License. See [LICENSE](./LICENSE) for more details.