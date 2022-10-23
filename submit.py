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
    tokenizer, special_token = tokenizer_info[args.pretrained]
    test_dataset = KERC22(tokenizer, './dataset/KERC/' + args.input)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_batch)

    """logging and path"""
    modelfile = os.path.join('KERC_models', args.pretrained, args.att, args.model)

    print('Load model: ', modelfile, '!!!')  # emotion
    model = CoMPM(args.pretrained, 3, special_token, tokenizer.cls_token_id, tokenizer.pad_token_id, len(tokenizer), attention=args.att)
    model.load_state_dict(torch.load(modelfile))
    model = model.cuda()


    """Dev & Test evaluation"""
    model.eval()
    dev_id_list, dev_pred_list = _gen(model, dataloader)
    generate_output(args.output, dev_id_list, dev_pred_list)


def generate_output(filename, id_list, pred_list):
    emoList = sorted({"dysphoria", "euphoria", "neutral"})

    with open(filename, 'w') as f:
        f.write("Id,Predicted\n")

        for id, pred in zip(id_list, pred_list):
            f.write("{},{}\n".format(int(id), emoList[pred]))

def _gen(model, dataloader):
    model.eval()
    pred_list = []
    label_list = []

    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            tokens, speakers, skips, labels = data
            labels = labels.cuda()

            pred_logits = model(tokens, speakers, skips)

            """Calculation"""
            pred_label = pred_logits.argmax(1).tolist()
            true_label = labels.tolist()

            pred_list += pred_label
            label_list += true_label
    return label_list, pred_list

if __name__ == '__main__':
    torch.cuda.empty_cache()

    """Parameters"""
    parser = argparse.ArgumentParser(description="Emotion Classifier")
    parser.add_argument("--pretrained", help='kobert albert funnel electr', default='kobert')
    parser.add_argument("--model", help='Model', default='3.bin')
    parser.add_argument("--input", help='Input file', default='public_test_data.tsv')
    parser.add_argument("--output", help='Submission name', default='submission.csv')
    parser.add_argument("--att", help='attention mechanism', default='none')

    args = parser.parse_args()

    streamHandler = logging.StreamHandler()

    main()
