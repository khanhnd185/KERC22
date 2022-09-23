import math
import torch
import torch.nn as nn

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout=0, num_heads=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads  # To be covered later
        self.softmax = nn.Softmax(dim=-1)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.softmax(scores)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = self.attend(scores)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

class CrossAttention(nn.Module):
    """ cross attention Module"""
    def __init__(self, in_channels, hidden_size):
        super(CrossAttention, self).__init__()
        self.linear_q = nn.Linear(in_channels, hidden_size)
        self.linear_k = nn.Linear(in_channels, hidden_size)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = hidden_size ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / hidden_size))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / hidden_size))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, q, k, v):
        query = self.linear_q(q)
        key = self.linear_k(k)
        value = self.linear_v(v)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out
