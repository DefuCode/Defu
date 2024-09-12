# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.autograd as autograd
import logging

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, dim, args):
        super(Attention,self).__init__()
        self.args =args
        self.linear = nn.Linear(dim*2, dim)
        self.mask = None
        self.fc = nn.Linear(self.args.filter_size * dim, dim)
        #self.adaptavgpool = torch.nn.AdaptiveAvgPool1d(1)

    def set_mask(self,mask):
        self.mask = torch.ByteTensor(mask).unsqueeze(2).cuda()

    def forward(self, key, query):
        batch_size = key.size(0)
        hidden_size = key.size(2)
        input_size = query.size(1)
        # (batch, key_len, dim) * (batch, query_len, dim) -> (batch, key_len, query_len)
        attn = torch.bmm(key, query.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill(self.mask, -float('inf'))
        attn = F.softmax(attn, dim=1)

        # (batch, key_len, query_len) * (batch, query_len, dim) -> (batch, key_len, dim)
        energy = torch.bmm(attn, query)
        # concat -> (batch, outlen, 2*dim)
        combind = torch.cat((energy,key), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear(combind.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        output = output.reshape(batch_size, -1)
        output = self.fc(output)
        return output, attn


class AttentionFuse(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear = nn.Linear(self.args.filter_size * config.hidden_size, config.hidden_size)

        self.attention = Attention(config.hidden_size, args)


    def forward(self, features, **kwargs):
        # ------------- cnn -------------------------
        x = torch.unsqueeze(features, dim=1)  # [B, L*D] -> [B, 1, D*L]
        x = x.reshape(x.shape[0], -1, 768)  # [B, L, D]
        outputs, _ = self.attention(x, x)                                 # ->[B, 128]
        # features = self.linear_mlp(features)       # [B, L*D] -> [B, D]
        features = self.linear(features)  # [B, 3*768] -> [B, 128]
        x = torch.cat((outputs, features), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- cnn ----------------------

        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x


class DistanceClassifier(nn.Module):
    def __init__(self, config, input_size, hidden_size):
        super(DistanceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input1, input2):
        # Concatenate two input embeddings
        x = torch.cat((input1, input2), dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1)  # 3->5
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.attention_fuse = AttentionFuse(config, self.args)
        self.distance_cal = DistanceClassifier(config, 2 * config.hidden_size, 2 * self.args.d_size)  # [2*768,2*128]

    # 计算代码表示
    def enc(self, seq_embeds):
        batch_size = seq_embeds.shape[0]
        # token_len = seq_ids.shape[-1]
        # # 计算路径表示E                                                    # [batch, path_num, ids_len_of_path/token_len]
        # seq_inputs = seq_ids.reshape(-1, token_len)  # [4, 3, 400] -> [4*3, 400]
        # seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]  # [4*3, 400] -> [4*3, 400, 768]
        # seq_embeds = seq_embeds[:, 0, :]  # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)  # [4*3, 768] -> [4, 3*768]
        # outputs_seq = self.dropout(outputs_seq)

        # 计算代码表示Z
        return self.attention_fuse(outputs_seq)

    def forward(self, anchor, positive, negative=None):
        if negative is not None:
            an_logits = self.enc(anchor)        # [B, 768]
            po_logits = self.enc(positive)
            ne_logits = self.enc(negative)
            ap_dis = self.distance_cal(an_logits, po_logits)
            an_dis = self.distance_cal(an_logits, ne_logits)

            return ap_dis, an_dis
        else:
            an_logits = self.enc(anchor)
            co_logits = self.enc(positive)
            ac_dis = self.distance_cal(an_logits, co_logits)
            return ac_dis
