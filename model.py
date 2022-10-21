import torch
import torch.nn as nn

from transformers import BertModel, AlbertModel, FunnelModel, ElectraModel

from attention import *

def padding(ids_list, pad_token_id):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)

    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [pad_token_id for _ in range(pad_len)]

        pad_ids.append(ids+add_ids)

    return torch.tensor(pad_ids)

class CoMPM(nn.Module):
    def __init__(self, model_type, clsNum, special_token, cls_token, pad_token, num_emb, attention='none'):
        super(CoMPM, self).__init__()
        self.gpu = True
        self.attention = attention
        self.special_token = special_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        
        if model_type == 'kobert':
            self.context_model = BertModel.from_pretrained("kykim/bert-kor-base")
            self.speaker_model = BertModel.from_pretrained("kykim/bert-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)
        elif model_type == 'albert':
            self.context_model = AlbertModel.from_pretrained("kykim/albert-kor-base")
            self.speaker_model = AlbertModel.from_pretrained("kykim/albert-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)
        elif model_type == 'funnel':
            self.context_model = FunnelModel.from_pretrained("kykim/funnel-kor-base")
            self.speaker_model = FunnelModel.from_pretrained("kykim/funnel-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)
        else: # electr
            self.context_model = ElectraModel.from_pretrained("kykim/electra-kor-base")
            self.speaker_model = ElectraModel.from_pretrained("kykim/electra-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)

        self.hiddenDim = self.context_model.config.hidden_size
        
        zero = torch.empty(2, 1, self.hiddenDim).cuda()
        self.h0 = torch.zeros_like(zero) # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
        self.conversationGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
            
        """score"""
        # self.SC = nn.Linear(self.hiddenDim, self.hiddenDim)
        self.W = nn.Linear(self.hiddenDim, clsNum)
        if attention == 'dot':
            self.attend = DotProductAttention()
        elif attention == 'cross':
            self.attend = CrossAttention(768, 768)
        elif attention == 'add':
            self.attend = AdditiveAttention(768, 0)

        """parameters"""
        self.train_params = list(self.context_model.parameters())+list(self.speakerGRU.parameters())+list(self.W.parameters()) # +list(self.SC.parameters())
        self.train_params += list(self.speaker_model.parameters())+list(self.conversationGRU.parameters())
    
    def forward(self, tokens, speakers, skips):
        outputs = []
        context = []
        speaker_tokens_list = [[] for _ in range(max(speakers) + 1)]

        for token, speaker, skip in zip(tokens, speakers, skips):
            truncated = token[-511:]
            speaker_tokens_list[speaker].append([self.cls_token] + truncated)
            context = [self.special_token + speaker] + token

            if skip:
                continue

            truncated = context[-511:]
            batch_input_tokens = torch.tensor([self.cls_token] + truncated).unsqueeze(0)
            batch_speaker_tokens = [padding(speaker_tokens_list[speaker], self.pad_token)]

            output = self._forward(batch_input_tokens, batch_speaker_tokens)
            outputs.append(output)
            
        outputs = torch.cat(outputs, 0)
        final_output, _ = self.conversationGRU(outputs)
        context_logit = self.W(final_output)
        
        return context_logit

    def _forward(self, batch_input_tokens, batch_speaker_tokens):
        """
            batch_input_tokens: (batch, len)
            batch_speaker_tokens: [(speaker_utt_num, len), ..., ]
        """
        batch_input_tokens = batch_input_tokens.cuda()

        batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,0,:] # (batch, 1024)
        
        batch_speaker_output = []
        for speaker_tokens in batch_speaker_tokens:
            if speaker_tokens.shape[0] == 0:
                speaker_track_vector = torch.zeros(1, self.hiddenDim).cuda()
            else:
                speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,0,:] # (speaker_utt_num, 1024)
                speaker_output = speaker_output.unsqueeze(1) # (speaker_utt_num, 1, 1024)
                speaker_GRU_output, _ = self.speakerGRU(speaker_output, self.h0) # (speaker_utt_num, 1, 1024) <- (seq_len, batch, output_size)
                speaker_track_vector = speaker_GRU_output[-1,:,:] # (1, 1024)
            batch_speaker_output.append(speaker_track_vector)
        batch_speaker_output = torch.cat(batch_speaker_output, 0) # (batch, 1024)
                   
        # final_output = batch_context_output + batch_speaker_output
        # final_output = batch_context_output + self.SC(batch_speaker_output)        
        if self.attention == 'none':
            final_output = batch_context_output + batch_speaker_output
        else:
            q = batch_speaker_output.unsqueeze(1)
            k = batch_context_output.unsqueeze(1)
            v = batch_context_output.unsqueeze(1)
            final_output = self.attend(q, k, v)
            final_output = final_output.squeeze(1)

        return final_output