# KERC Competition Repo of CNUSclab Team

## Introduction

The 4th Korean Emotion Recognition International Challenge ([KERC22](https://sites.google.com/view/kerc2022)) aims to solve the prolem of Socio-behavior in Korean Conversation. The data is collected from transcript of Korean drama. Each utterance in the conversation is labeled as "dysphoria", "euphoria", or "neutral".

I (CNU_Sclab Team) paticipated in the competition and achieved 4th prize. I use pre-trained language model (particularly, electra) to analyze the conversation context and the speakers' memory. The detailed solution is described in my paper [MAnalyzing Context and Speaker Memory using Pretrained Language Model for Emotion Recognition in Korean Conversation task](https://drive.google.com/file/d/1c7N5KcWahmLxxS42UXhjY6CIjdcjh26o/view). The paper is in proceeding of The 10th International Conference on Big Data Applications and Services ([BIGDAS22](http://kbigdata.or.kr/bigdas2022/)).

I used language model pre-trained on Korean language task from [Kim](hhttps://github.com/kiyoungkim1/LMkor). The idea is inspired the CoMPM model of [Lee](https://github.com/rungjoo/CoMPM).

# Model Architecture

![Architecture of our model](https://raw.githubusercontent.com/khanhnd185/my-blog/my-pages/_posts/images/kerc22/model.png)

## Training
------
```
python train.py --pretrained <kobert|electr>
```

## Generate submission
------
```
python submit.py --model <model_name>.bin --pretrained <kobert|electr> --output <file_name>.csv --input <private_test_data|public_test_data>.tsv
```

# Results

The best performances on validation set of the HUME-VB dataset are listed below:

| Model | F1-score |
| --- | --- |
| Baseline | 65.67 | 
| Emotionflow | 66.88 | 
| Ours (Kobert) | 75.33 | 
| Ours (Elektra) | 76.50 | 
