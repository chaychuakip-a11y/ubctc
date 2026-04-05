import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ..functions import xavier
from ..c import *
from torch.jit import Final
from ..data import clip_mask, cnn2rnn
class MultiHeadAttention(nn.Module):

    num_head: Final[int]
    attention_dim: Final[int]
    window_width: Final[int]

    def __init__(self, num_head, input_dim, attention_dim, output_dim, window_width=31, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_head = num_head
        self.attention_dim = attention_dim
        self.window_width = window_width

        self.input_trans = nn.Linear(input_dim, attention_dim * 3)
        self.fc = nn.Linear(attention_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        xavier(self.input_trans.weight.data)
        nn.init.constant_(self.input_trans.bias.data, 0)
        xavier(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0)


    def forward(self, input_:torch.Tensor, meta:Dict[str, torch.Tensor]) -> torch.Tensor:
        
        n, c, h, w = input_.shape
        input_ = input_.reshape( (n, c, w) ).permute( (0, 2, 1) ) # (b, t, d)

        temp = self.input_trans(input_) # (b, t, d)
        # now actual is (b, num_head, head_dim, t), but num_head is add to head_dim, so we have (b, num_head*head_dim, t). curse caffe
        temp = temp.permute(0, 2, 1) 
    
        heads = temp.chunk(self.num_head, dim=1)

        out_holder = []

        for i in range(self.num_head):
            one_head = heads[i]
            q, k, v = one_head.chunk(3, dim=1) # [q, k, v] all is (b, head_dim, t)

            attn = torch.bmm(q.permute(0, 2, 1), k) # b t d @ b d t -> b t t
            attn = attn * (self.num_head / self.attention_dim)
            mask = clip_mask(meta["mask"], attn.size(2), 1)
            attn = mha_mask(attn, mask, self.window_width, float(-1e10), float(0))
            attn = F.softmax(attn, dim=2)
            attn = mha_mask(attn, mask, self.window_width, float(0), float(0))
            attn = self.dropout(attn)
            output = torch.bmm(v, attn.permute(0, 2, 1)) # b d t @ b t t -> b d t | (1, 64, 329)

            out_holder.append(output)
     
        output = torch.cat(out_holder, dim=1) # (1, 512, 329)

        output = output.permute(0, 2, 1)
        output = self.fc(output)

        b, t, d = output.shape
        output = output.permute( (0, 2, 1) ).reshape( (b, d, 1, t) )

        return output


