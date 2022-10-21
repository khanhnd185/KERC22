from tqdm import tqdm
import os
import torch
from dataset import KERC22
from model import CoMPM
from torch.utils.data import DataLoader
import argparse, logging
from utils import tokenizer_info, make_batch

def main():
    """Dataset Loading"""
    model_type = args.pretrained
    freeze = args.freeze
    initial = args.initial
    name = args.model
    input = args.input
    output = args.output
    attention = args.att

    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'

    tokenizer, special_token = tokenizer_info[model_type]
    test_dataset = KERC22(tokenizer, './dataset/KERC/train_data.tsv', label_file_name='./dataset/KERC/train_labels.csv')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=make_batch)

    """logging and path"""
    save_path = os.path.join('KERC_models', model_type, initial, freeze_type, attention)
    modelfile = os.path.join(save_path, name)

    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)

    print('Load model: ', modelfile, '!!!')  # emotion
    model = CoMPM(model_type, 3, special_token, tokenizer.cls_token_id, tokenizer.pad_token_id, len(tokenizer), attention=attention)
    model.load_state_dict(torch.load(modelfile))
    model = model.cuda()


    """Dev & Test evaluation"""
    model.eval()
    dev_id_list, dev_pred_list = _gen(model, dataloader)
    generate_output(output, dev_id_list, dev_pred_list)


def generate_output(filename, id_list, pred_list):
    emoList = sorted({"dysphoria", "euphoria", "neutral"})

    with open(filename, 'w') as f:
        f.write("Id,Predicted\n")

        for id, pred in zip(id_list, pred_list):
            f.write("{},{}\n".format(int(id), emoList[pred]))

def _gen(model, dataloader):
    model.eval()
    pred_list = []
    id_list = []

    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_input_tokens , batch_speaker_tokens, id = data
            batch_input_tokens = batch_input_tokens.cuda()

            pred_logits = model(batch_input_tokens, batch_speaker_tokens)  # (1, clsNum)

            """Calculation"""
            pred_label = pred_logits.argmax(1).item()

            pred_list.append(pred_label)
            id_list.append(id)
    return id_list, pred_list

if __name__ == '__main__':
    torch.cuda.empty_cache()

    """Parameters"""
    parser = argparse.ArgumentParser(description="Emotion Classifier")

    parser.add_argument("--epoch", type=int, help='training epohcs', default=10)  # 12 for iemocap
    parser.add_argument("--norm", type=int, help="max_grad_norm", default=10)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-6)  # 1e-5
    parser.add_argument("--pretrained", help='kobert albert funnel electr', default='kobert')
    parser.add_argument("--initial", help='pretrained or scratch', default='pretrained')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument("--model", help='Model', default='3.bin')
    parser.add_argument("--input", help='Input file', default='public_test_data.tsv')
    parser.add_argument("--output", help='Submission name', default='submission.csv')
    parser.add_argument("--att", help='attention mechanism', default='none')

    args = parser.parse_args()

    streamHandler = logging.StreamHandler()

    main()
