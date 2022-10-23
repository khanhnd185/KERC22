import torch
from transformers import ElectraTokenizerFast
from transformers import BertTokenizerFast
from transformers import FunnelTokenizerFast

MAX_NUM_SPEAKERS = 12

tokenizer_kobert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
tokenizer_albert = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")
tokenizer_funnel = FunnelTokenizerFast.from_pretrained("kykim/funnel-kor-base")
tokenizer_electr = ElectraTokenizerFast.from_pretrained("kykim/electra-kor-base")

condition_token = ['<s{}>'.format(i+1) for i in range(MAX_NUM_SPEAKERS)]
special_tokens = {'additional_special_tokens': condition_token}

special_token_kobert = len(tokenizer_kobert)
special_token_albert = len(tokenizer_albert)
special_token_funnel = len(tokenizer_funnel)
special_token_electr = len(tokenizer_electr)

tokenizer_kobert.add_special_tokens(special_tokens)
tokenizer_albert.add_special_tokens(special_tokens)
tokenizer_funnel.add_special_tokens(special_tokens)
tokenizer_electr.add_special_tokens(special_tokens)

tokenizer_info = {
    'kobert': [tokenizer_kobert, special_token_kobert],
    'albert': [tokenizer_albert, special_token_albert],
    'funnel': [tokenizer_funnel, special_token_funnel],
    'electr': [tokenizer_electr, special_token_electr],
}

def make_batch(sessions):
    tokens, speakers, skips, ret = sessions[0] # fixed batch_size=1
    ret = torch.tensor(ret)

    return tokens, speakers, skips, ret
