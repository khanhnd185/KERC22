import torch
from transformers import ElectraTokenizerFast

MAX_NUM_SPEAKERS = 12

electra_tokenizer = ElectraTokenizerFast.from_pretrained("kykim/electra-kor-base")
condition_token = ['<s{}>'.format(i+1) for i in range(MAX_NUM_SPEAKERS)]
special_tokens = {'additional_special_tokens': condition_token}
electra_tokenizer.add_special_tokens(special_tokens)

MAX_NUM_EMBEDDINGS = len(electra_tokenizer)

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

def make_batch_electra(sessions):
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
                speaker_utt_list.append(encode_right_truncated(utt, electra_tokenizer))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, electra_tokenizer))
        batch_labels.append(emotion)        
        batch_speaker_tokens.append(padding(speaker_utt_list, electra_tokenizer))

    batch_input_tokens = padding(batch_input, electra_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_speaker_tokens, batch_labels

def make_test_batch_electra(sessions):
    batch_input, batch_speaker_tokens, id_list = [], [], []
    for session in sessions:
        context_speaker, context, id = session
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, electra_tokenizer))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, electra_tokenizer))       
        id_list.append(id[-1])
        batch_speaker_tokens.append(padding(speaker_utt_list, electra_tokenizer))

    batch_input_tokens = padding(batch_input, electra_tokenizer)
    
    return batch_input_tokens, batch_speaker_tokens, id_list
