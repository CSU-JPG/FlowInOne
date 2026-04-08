"""
Transformer-based varitional encoder model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Adaptor(nn.Module):
    def __init__(self, input_dim, tar_dim):
        super(Adaptor, self).__init__()

        if tar_dim == 32768:
            output_channel = 8
        elif tar_dim == 16384:
            output_channel = 4
        else:
            raise NotImplementedError("only support 512px, 256px does not need this")

        self.tar_dim = tar_dim
        
        self.fc1 = nn.Linear(input_dim, 4096)
        self.ln_fc1 = nn.LayerNorm(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.ln_fc2 = nn.LayerNorm(4096)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.ln_conv1 = nn.LayerNorm([32, 64, 64])
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.ln_conv2 = nn.LayerNorm([64, 64, 64])
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = torch.relu(self.ln_fc1(self.fc1(x)))
        x = torch.relu(self.ln_fc2(self.fc2(x)))
        
        x = x.view(-1, 1, 64, 64)
        
        x = torch.relu(self.ln_conv1(self.conv1(x)))
        x = torch.relu(self.ln_conv2(self.conv2(x)))

        x = self.conv3(x)
        x = x.view(-1, self.tar_dim)
        
        return x


class TransEncoder(nn.Module):
    def __init__(self, d_model, N, num_token, head_num, d_ff, latten_size, down_sample_block=3, dropout=0.1, last_norm=True):
        super(TransEncoder, self).__init__()
        self.N = N
        
        self.layers = clones(EncoderLayer(MultiHeadAttentioin(d_model, head_num, dropout=dropout),
                                          FeedForward(d_model, d_ff, dropout=dropout),
                                          LayerNorm(d_model),
                                          LayerNorm(d_model)), N)
        
        self.reduction_layers = nn.ModuleList()
        for _ in range(down_sample_block):
            self.reduction_layers.append(
                EncoderReductionLayer(MultiHeadAttentioin(d_model, head_num, dropout=dropout),
                                  FeedForward(d_model, d_ff, dropout=dropout),
                                  nn.Linear(d_model, d_model // 2),
                                  LayerNorm(d_model),
                                  LayerNorm(d_model)))
            d_model = d_model // 2

        if latten_size == 8192 or latten_size == 4096:
            self.arc = 0
            self.linear = nn.Linear(d_model*num_token, latten_size)
            self.norm = LayerNorm(latten_size) if last_norm else None
        else:
            self.arc = 1
            self.adaptor = Adaptor(d_model*num_token, latten_size)


    def forward(self, x, mask):
        mask = mask.unsqueeze(1)

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)

        for i, layer in enumerate(self.reduction_layers):
            x = layer(x, mask)

        if self.arc == 0:
            x = self.linear(x.view(x.shape[0],-1))
            x = self.norm(x) if self.norm else x
        else:
            x = self.adaptor(x.view(x.shape[0],-1))

        return x


class EncoderLayer(nn.Module):
    def __init__(self, attn, feed_forward, norm1, norm2, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.norm1, self.norm2 = norm1, norm2

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # multihead attn & norm
        a = self.attn(x, x, x, mask)
        t = self.norm1(x + self.dropout1(a))

        # feed forward & norm
        z = self.feed_forward(t)  # linear(dropout(act(linear(x)))))
        y = self.norm2(t + self.dropout2(z))

        return y


class EncoderReductionLayer(nn.Module):
    def __init__(self, attn, feed_forward, reduction, norm1, norm2, dropout=0.1):
        super(EncoderReductionLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.reduction = reduction
        self.norm1, self.norm2 = norm1, norm2

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # multihead attn & norm
        a = self.attn(x, x, x, mask)
        t = self.norm1(x + self.dropout1(a))

        # feed forward & norm
        z = self.feed_forward(t)  # linear(dropout(act(linear(x)))))
        y = self.norm2(t + self.dropout2(z))

        # reduction
        # y = self.reduction(y).view(x.shape[0], -1, x.shape[-1])
        y = self.reduction(y)

        return y


class MultiHeadAttentioin(nn.Module):
    def __init__(self, d_model, head_num, dropout=0.1, d_v=None):
        super(MultiHeadAttentioin, self).__init__()
        assert d_model % head_num == 0, "d_model must be divisible by head_num"

        self.d_model = d_model
        self.head_num = head_num
        self.d_k = d_model // head_num
        self.d_v = self.d_k if d_v is None else d_v

        # d_model = d_k * head_num
        self.W_Q = nn.Linear(d_model, head_num * self.d_k)
        self.W_K = nn.Linear(d_model, head_num * self.d_k)
        self.W_V = nn.Linear(d_model, head_num * self.d_v)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dp_attn(self, query, key, value, mask=None):
        assert self.d_k == query.shape[-1]

        # scores: [batch_size, head_num, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # if torch.isinf(scores).any():
        #     # to avoid leaking
        #     scores = torch.where(scores == float('-inf'), torch.tensor(-65504.0), scores)
        #     scores = torch.where(scores == float('inf'), torch.tensor(65504.0), scores)

        if mask is not None:
            assert mask.ndim == 3, "Mask shape {} doesn't seem right...".format(mask.shape)
            mask = mask.unsqueeze(1)
            try:
                if scores.dtype == torch.float32:
                    scores = scores.masked_fill(mask == 0, -1e9)
                else:
                    scores = scores.masked_fill(mask == 0, -1e4)
            except RuntimeError:
                print("- scores device: {}".format(scores.device))
                print("- mask device: {}".format(mask.device))

        # attn: [batch_size, head_num, seq_len, seq_len]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]

        query = self.W_Q(q).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        key = self.W_K(k).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        value = self.W_V(v).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)

        heads, attn = self.scaled_dp_attn(query, key, value, mask)
        heads = heads.transpose(1, 2).contiguous().view(batch_size, -1,
                                                        self.head_num * self.d_k)
        assert heads.shape[-1] == self.d_model and heads.shape[0] == batch_size

        y = self.W_O(heads)

        assert y.shape == q.shape
        return y


class LayerNorm(nn.Module):
    def __init__(self, layer_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(layer_size))
        self.b = nn.Parameter(torch.zeros(layer_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.g * x + self.b


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, act='relu', d_output=None):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        d_output = d_model if d_output is None else d_output

        self.ffn_1 = nn.Linear(d_model, d_ff)
        self.ffn_2 = nn.Linear(d_ff, d_output)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ffn_2(self.dropout(self.act(self.ffn_1(x))))
        return y