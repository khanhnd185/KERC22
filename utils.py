import torch
from transformers import ElectraTokenizerFast

electra_tokenizer = ElectraTokenizerFast.from_pretrained("kykim/electra-kor-base")

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
    for session in sessions:
        data = session[0]
        label_list = session[1]
        
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
        
        label_ind = label_list.index(emotion) # 지금 여기서 오류가 뜨는거 같은데
        batch_labels.append(label_ind)        
        
        batch_speaker_tokens.append(padding(speaker_utt_list, electra_tokenizer))

    batch_input_tokens = padding(batch_input, electra_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels, batch_speaker_tokens

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
