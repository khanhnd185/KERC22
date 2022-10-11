import torch
import torch.nn as nn

from transformers import BertModel, AlbertModel, FunnelModel, ElectraModel

from attention import *

class CoMPM(nn.Module):
    def __init__(self, model_type, clsNum, last, freeze, initial, num_emb, attention='none'):
        super(CoMPM, self).__init__()
        self.gpu = True
        self.last = last
        self.attention = attention
        
        if model_type == 'kobert':
            self.context_model = BertModel.from_pretrained("kykim/bert-kor-base")
            self.speaker_model = BertModel.from_pretrained("kykim/bert-kor-base")
            self.descrip_model = BertModel.from_pretrained("kykim/bert-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)
            self.descrip_model.resize_token_embeddings(num_emb)
        elif model_type == 'albert':
            self.context_model = AlbertModel.from_pretrained("kykim/albert-kor-base")
            self.speaker_model = AlbertModel.from_pretrained("kykim/albert-kor-base")
            self.descrip_model = AlbertModel.from_pretrained("kykim/albert-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)
            self.descrip_model.resize_token_embeddings(num_emb)
        elif model_type == 'funnel':
            self.context_model = FunnelModel.from_pretrained("kykim/funnel-kor-base")
            self.speaker_model = FunnelModel.from_pretrained("kykim/funnel-kor-base")
            self.descrip_model = FunnelModel.from_pretrained("kykim/funnel-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)
            self.descrip_model.resize_token_embeddings(num_emb)
        else: # electr
            self.context_model = ElectraModel.from_pretrained("kykim/electra-kor-base")
            self.speaker_model = ElectraModel.from_pretrained("kykim/electra-kor-base")
            self.descrip_model = ElectraModel.from_pretrained("kykim/electra-kor-base")
            self.context_model.resize_token_embeddings(num_emb)
            self.speaker_model.resize_token_embeddings(num_emb)
            self.descrip_model.resize_token_embeddings(num_emb)

        self.hiddenDim = self.context_model.config.hidden_size
        
        zero = torch.empty(2, 1, self.hiddenDim).cuda()
        self.h0 = torch.zeros_like(zero) # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
            
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
        if not freeze:
            self.train_params += list(self.speaker_model.parameters())

    def forward(self, batch_input_tokens, batch_descrip_tokens, batch_speaker_tokens):
        """
            batch_input_tokens: (batch, len)
            batch_speaker_tokens: [(speaker_utt_num, len), ..., ]
        """
        if self.last:
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,-1,:] # (batch, 1024)
        else:
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,0,:] # (batch, 1024)

        if batch_descrip_tokens != None:
            if self.last:
                batch_descrip_output = self.descrip_model(batch_descrip_tokens.cuda()).last_hidden_state[:,-1,:] # (batch, 1024)
            else:
                batch_descrip_output = self.descrip_model(batch_descrip_tokens.cuda()).last_hidden_state[:,0,:] # (batch, 1024)
        else:
            batch_descrip_output = torch.zeros(1, self.hiddenDim).cuda()
        
        batch_speaker_output = []
        for speaker_tokens in batch_speaker_tokens:
            if speaker_tokens.shape[0] == 0:
                speaker_track_vector = torch.zeros(1, self.hiddenDim).cuda()
            else:
                if self.last:
                    speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,-1,:] # (speaker_utt_num, 1024)
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
            final_output = batch_context_output + batch_speaker_output + batch_descrip_output
        else:
            q = batch_speaker_output.unsqueeze(1)
            k = batch_context_output.unsqueeze(1)
            v = batch_context_output.unsqueeze(1)
            final_output = self.attend(q, k, v)
            final_output = final_output.squeeze(1)
        context_logit = self.W(final_output) # (batch, clsNum)
        
        return context_logit