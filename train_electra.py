# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn

from transformers import RobertaTokenizer
from ERC_dataset import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader, KERC_loader
from model import ERC_model
# from ERCcombined import ERC_model

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support

from utils import make_batch_roberta, make_batch_bert, make_batch_gpt, make_batch_electra


def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val


## finetune RoBETa-large
def main():
    """Dataset Loading"""
    batch_size = args.batch
    dataset = args.dataset
    dataclass = args.cls
    sample = args.sample
    model_type = args.pretrained
    freeze = args.freeze
    initial = args.initial

    dataType = 'multi'
    if dataset == 'MELD':
        if args.dyadic:
            dataType = 'dyadic'
        else:
            dataType = 'multi'
        data_path = './dataset/MELD/' + dataType + '/'
        DATA_loader = MELD_loader
    elif dataset == 'EMORY':
        data_path = './dataset/EMORY/'
        DATA_loader = Emory_loader
    elif dataset == 'iemocap':
        data_path = './dataset/iemocap/'
        DATA_loader = IEMOCAP_loader
    elif dataset == 'dailydialog':
        data_path = './dataset/dailydialog/'
        DATA_loader = DD_loader
    elif dataset == 'KERC':
        data_path = './dataset/KERC/'
        DATA_loader = KERC_loader

    if 'roberta' in model_type:
        make_batch = make_batch_roberta
    elif model_type == 'bert-large-uncased':
        make_batch = make_batch_bert
    elif model_type == 'electra-kor-base':
        make_batch = make_batch_electra
    else:
        make_batch = make_batch_gpt

    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'

    train_path = data_path + dataset + '_train.txt'

    train_dataset = DATA_loader(train_path, dataclass)
    if sample < 1.0:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                      collate_fn=make_batch)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                      collate_fn=make_batch)
    train_sample_num = int(len(train_dataloader) * sample)

    """logging and path"""
    save_path = os.path.join(dataset + '_models', model_type, initial, freeze_type, dataclass, str(sample))

    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    """Model Loading"""
    if 'gpt2' in model_type:
        last = True
    else:
        last = False

    print('DataClass: ', dataclass, '!!!')  # emotion
    clsNum = len(train_dataset.labelList)
    model = ERC_model(model_type, clsNum, last, freeze, initial)
    model = model.cuda()
    model.train()

    """Training Setting"""
    training_epochs = args.epoch
    save_term = int(training_epochs / 5)
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset) * training_epochs
    num_warmup_steps = len(train_dataset)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    optimizer = torch.optim.AdamW(model.train_params, lr=lr)  # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    """Input & Label Setting"""
    best_dev_fscore, best_test_fscore = 0, 0
    best_dev_fscore_macro, best_dev_fscore_micro, best_test_fscore_macro, best_test_fscore_micro = 0, 0, 0, 0
    best_epoch = 0
    for epoch in range(training_epochs):
        model.train()
        for i_batch, data in enumerate(tqdm(train_dataloader)):
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break

            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()

            pred_logits = model(batch_input_tokens, batch_speaker_tokens)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_labels)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        """Dev & Test evaluation"""
        model.eval()
        dev_acc, dev_pred_list, dev_label_list = _CalACC(model, train_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list,
                                                                         average='micro')

        """Best Score & Model Save"""
        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta

            best_epoch = epoch
            _SaveModel(model, save_path)

        logger.info('Epoch: {}'.format(epoch))

        logger.info(
            'Train ## accuracy: {}, precision: {}, recall: {}, fscore: {}'.format(dev_acc, dev_pre,
                                                                                        dev_rec, dev_fbeta))
        logger.info('')

def _CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []

    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()

            pred_logits = model(batch_input_tokens, batch_speaker_tokens)  # (1, clsNum)

            """Calculation"""
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_labels.item()

            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct / len(dataloader)
    return acc, pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))


if __name__ == '__main__':
    torch.cuda.empty_cache()

    """Parameters"""
    parser = argparse.ArgumentParser(description="Emotion Classifier")
    parser.add_argument("--batch", type=int, help="batch_size", default=1)

    parser.add_argument("--epoch", type=int, help='training epohcs', default=10)  # 12 for iemocap
    parser.add_argument("--norm", type=int, help="max_grad_norm", default=10)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-6)  # 1e-5
    parser.add_argument("--sample", type=float, help="sampling trainign dataset", default=1.0)  #

    parser.add_argument("--dataset", help='MELD or EMORY or iemocap or dailydialog', default='KERC')

    parser.add_argument("--pretrained", help='roberta-large or bert-large-uncased or gpt2 or gpt2-large or gpt2-medium',
                        default='electra-kor-base')
    parser.add_argument("--initial", help='pretrained or scratch', default='pretrained')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument("--cls", help='emotion or sentiment', default='emotion')

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()

    main()
