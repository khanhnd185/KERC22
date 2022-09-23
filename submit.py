from tqdm import tqdm
import os
import torch
from ERC_dataset import KERCTest_loader
from model import ERC_model
from torch.utils.data import DataLoader
import argparse, logging
from utils import make_test_batch_electra

def main():
    """Dataset Loading"""
    dataset = args.dataset
    dataclass = args.cls
    model_type = args.pretrained
    freeze = args.freeze
    initial = args.initial
    input = args.input
    output = args.output

    data_path = './dataset/KERC/'
    DATA_loader = KERCTest_loader
    make_batch = make_test_batch_electra

    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'

    test_path = data_path + dataset + '_publictest.txt'
    test_path = data_path + 'KERC_publictest_narrator.txt'

    test_dataset = DATA_loader(test_path, dataclass)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                    collate_fn=make_batch)

    """logging and path"""
    save_path = os.path.join(dataset + '_models', model_type, initial, freeze_type, dataclass, "1.0")
    modelfile = os.path.join(save_path, input)

    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)

    """Model Loading"""
    if 'gpt2' in model_type:
        last = True
    else:
        last = False

    print('Load model: ', modelfile, '!!!')  # emotion
    model = ERC_model(model_type, 3, last, freeze, initial)
    model.load_state_dict(torch.load(modelfile))
    model = model.cuda()


    """Dev & Test evaluation"""
    model.eval()
    dev_id_list, dev_pred_list = _gen(model, dataloader)
    generate_output(output, dev_id_list, dev_pred_list)


def generate_output(filename, id_list, pred_list):
    emoSet = {"euphoria", "dysphoria", "neutral"}
    emoList = sorted(emoSet)

    with open(filename, 'w') as f:
        f.write("Id,Predicted\n")

        for id, pred in zip(id_list, pred_list):
            f.write("{},{}\n".format(int(id[0]), emoList[pred]))

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
    parser.add_argument("--dataset", help='MELD or EMORY or iemocap or dailydialog', default='KERC')
    parser.add_argument("--pretrained", help='roberta-large or bert-large-uncased or gpt2 or gpt2-large or gpt2-medium',
                        default='electra-kor-base')
    parser.add_argument("--initial", help='pretrained or scratch', default='pretrained')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument("--cls", help='emotion or sentiment', default='emotion')
    parser.add_argument("--input", help='Model', default='model_origin.bin')
    parser.add_argument("--output", help='Submission name', default='submission.csv')

    args = parser.parse_args()

    streamHandler = logging.StreamHandler()

    main()
