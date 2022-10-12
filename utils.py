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

tokenizer_kobert.add_special_tokens(special_tokens)
tokenizer_albert.add_special_tokens(special_tokens)
tokenizer_funnel.add_special_tokens(special_tokens)
tokenizer_electr.add_special_tokens(special_tokens)

MAX_EMBEDS_KOBERT = len(tokenizer_kobert)
MAX_EMBEDS_ALBERT = len(tokenizer_albert)
MAX_EMBEDS_FUNNEL = len(tokenizer_funnel)
MAX_EMBEDS_ELECTR = len(tokenizer_electr)

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)

    return [tokenizer.cls_token_id] + ids

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)

    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]

        pad_ids.append(ids+add_ids)

    return torch.tensor(pad_ids)

def make_batch_kobert(sessions):
    batch_input, batch_labels, batch_speaker_tokens = [], [], []
    for data in sessions:
        context_speaker, context, emotion = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []

        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "

            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, tokenizer_kobert))

        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, tokenizer_kobert))
        batch_labels.append(emotion)        
        batch_speaker_tokens.append(padding(speaker_utt_list, tokenizer_kobert))

    batch_input_tokens = padding(batch_input, tokenizer_kobert)
    batch_labels = torch.tensor(batch_labels)    

    return batch_input_tokens, batch_speaker_tokens, batch_labels

def make_batch_albert(sessions):
    batch_input, batch_labels, batch_speaker_tokens = [], [], []
    for data in sessions:
        context_speaker, context, emotion = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []

        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "

            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, tokenizer_albert, max_length=255)) # Albert's max length is 256

        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, tokenizer_albert, max_length=255)) # Albert's max length is 256 
        batch_labels.append(emotion)        
        batch_speaker_tokens.append(padding(speaker_utt_list, tokenizer_albert))

    batch_input_tokens = padding(batch_input, tokenizer_albert)
    batch_labels = torch.tensor(batch_labels)    

    return batch_input_tokens, batch_speaker_tokens, batch_labels

def make_batch_funnel(sessions):
    batch_input, batch_labels, batch_speaker_tokens = [], [], []
    for data in sessions:
        context_speaker, context, emotion = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []

        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "

            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, tokenizer_funnel))

        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, tokenizer_funnel))
        batch_labels.append(emotion)        
        batch_speaker_tokens.append(padding(speaker_utt_list, tokenizer_funnel))

    batch_input_tokens = padding(batch_input, tokenizer_funnel)
    batch_labels = torch.tensor(batch_labels)    

    return batch_input_tokens, batch_speaker_tokens, batch_labels

def make_batch_electr(sessions):
    batch_input, batch_labels, batch_speaker_tokens = [], [], []
    for data in sessions:
        context_speaker, context, emotion = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []

        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "

            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, tokenizer_electr))

        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, tokenizer_electr))
        batch_labels.append(emotion)        
        batch_speaker_tokens.append(padding(speaker_utt_list, tokenizer_electr))

    batch_input_tokens = padding(batch_input, tokenizer_electr)
    batch_labels = torch.tensor(batch_labels)    

    return batch_input_tokens, batch_speaker_tokens, batch_labels

tokenizer_info = {
    'kobert': [MAX_EMBEDS_KOBERT, make_batch_kobert],
    'albert': [MAX_EMBEDS_ALBERT, make_batch_albert],
    'funnel': [MAX_EMBEDS_FUNNEL, make_batch_funnel],
    'electr': [MAX_EMBEDS_ELECTR, make_batch_electr]
}
