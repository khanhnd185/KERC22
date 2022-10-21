# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from dataset import KERC22
from model import CoMPM

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
from utils import tokenizer_info, make_batch


def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss(weight=torch.tensor([1.63, 3.73, 8.67]).cuda())
    loss_val = loss(pred_outs, labels)
    return loss_val


## finetune RoBETa-large
def main():
    """Dataset Loading"""
    model_type = args.pretrained
    freeze = args.freeze
    initial = args.initial
    attention = args.att

    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'

    tokenizer, special_token = tokenizer_info[model_type]
    train_dataset = KERC22(tokenizer, './dataset/KERC/train_data.tsv', label_file_name='./dataset/KERC/train_labels.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=make_batch)
    train_sample_num = len(train_dataloader)

    """logging and path"""
    save_path = os.path.join('KERC_models', model_type, initial, freeze_type, attention)

    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    model = CoMPM(model_type, 3, special_token, tokenizer.cls_token_id, tokenizer.pad_token_id, len(tokenizer), attention=attention)
    model = model.cuda()
    model.train()

    """Training Setting"""
    training_epochs = args.epoch
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset) * training_epochs
    num_warmup_steps = len(train_dataset)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    optimizer = torch.optim.AdamW(model.train_params, lr=lr)  # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    """Input & Label Setting"""
    for epoch in range(training_epochs):
        model.train()
        for i_batch, data in enumerate(tqdm(train_dataloader)):
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break

            """Prediction"""
            tokens, speakers, skips, labels = data
            labels = labels.cuda()

            pred_logits = model(tokens, speakers, skips)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, labels)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        """Dev & Test evaluation"""
        model.eval()
        dev_pred_list, dev_label_list = _CalACC(model, train_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list,
                                                                         average='micro')

        _SaveModel(model, save_path, "{}.bin".format(epoch))

        logger.info('Epoch: {}'.format(epoch))
        logger.info('Train ## precision: {}, recall: {}, fscore: {}'.format(dev_pre, dev_rec, dev_fbeta))
        logger.info('')

def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []

    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            tokens, speakers, skips, labels = data
            labels = labels.cuda()

            pred_logits = model(tokens, speakers, skips)

            """Calculation"""
            pred_label = pred_logits.argmax(1).item()
            true_label = labels.item()

            pred_list += pred_label
            label_list += true_label
    return pred_list, label_list


def _SaveModel(model, path, name='model.bin'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, name))


if __name__ == '__main__':
    torch.cuda.empty_cache()

    """Parameters"""
    parser = argparse.ArgumentParser(description="Emotion Classifier")
    parser.add_argument("--epoch", type=int, help='training epohcs', default=30)  # 12 for iemocap
    parser.add_argument("--norm", type=int, help="max_grad_norm", default=10)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-6)  # 1e-5
    parser.add_argument("--pretrained", help='kobert albert funnel electr',
                        default='kobert')
    parser.add_argument("--initial", help='pretrained or scratch', default='pretrained')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument("--att", help='attention mechanism', default='none')

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()

    main()
