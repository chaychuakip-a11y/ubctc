# by pcli2 2019 Dec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import time
import math
from asr.layers import *
from asr.data import clip_mask, cnn2rnn, rnn2cnn
from typing import List, Tuple, Dict
from asr.functions import xavier
from asr.train import Trainer

class Dnn(nn.Module):
    def __init__(self):
        super(Dnn, self).__init__()
        self.encoder = Encoder()
        #self.decoder = Decoder()
        self.classification = nn.Conv2d(224, 9003, 1, 1, 0)
        xavier(self.classification.weight)
        nn.init.zeros_(self.classification.bias.data)
        self.loss = CeLoss()
        self.accuracy = ACC()

    def forward(self, x: torch.Tensor, meta:Dict[str, torch.Tensor]):
        att_label = clip_mask(meta["att_label"], self.encoder.concat_fr.nmod, 0)
        enc_output = self.encoder(x, meta)
        #dec_output = self.decoder(enc_output, meta)
        
        b,d,f,t = enc_output.shape
        enc_output = enc_output.permute((2, 1, 3, 0))
        enc_output = enc_output.reshape((1, d, 1, t*b))

        logit = self.classification(enc_output)
        logit = logit.squeeze().permute((1, 0))

        if self.training:
            loss = self.loss(logit, att_label)
            return loss
        else:
            acc = self.accuracy(logit, att_label)
            return acc
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.concat_fr = ConcatFrLayer(4)
        self.lorder  = 5
        self.rorder  = 5
        self.dnn1 = nn.Conv1d(75, 2048, self.lorder+self.rorder+1, stride=1, padding=0, groups=75)
        self.dnn2 = nn.Linear(2048, 2048)
        self.dnn3 = nn.Linear(2048, 2048)
        self.dnn4 = nn.Linear(2048, 2048)
        self.dnn5 = nn.Linear(2048, 2048)
        self.dnn6 = nn.Linear(2048, 2048)
        # xavier(self.conv1.weight)
    
    def forward(self, x: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.concat_fr(x)
        x = cnn2rnn(x)
        lstm_mask = clip_mask(meta["rnn_mask"], self.concat_fr.nmod, 0)
        x = self.ub1(x, lstm_mask)
        x = self.ub2(x, lstm_mask)
        x = self.ub3(x, lstm_mask)
        x = rnn2cnn(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        #x = cnn2rnn(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.mlp_attention = MochaAttention(num_class=14831, input_dim=512, embedding_dim=512, mlp_at_dim=512, lstm_cell_num=2048, lstm_out_dim=512,
                                            act_type="sigmoid", mocha_act_type="sigmoid", mocha_w=8, mocha_valid=True, mocha_noise_var=1.0,
                                            clip_threshold=1.0)
        self.dec_lstm = LSTMP(1024, 2048, 512)

    def forward(self, x: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        att_mask = meta["att_mask"]
        dec_lstm_mask = att_mask.unsqueeze(2)
        att_feature = self.mlp_attention(x, meta)
        x = self.dec_lstm(att_feature, dec_lstm_mask)

        t, b, d = x.shape
        x = x.permute((2, 0, 1))
        x = x.reshape((1, d, 1, t*b))
        return x


class trainer(Trainer):
    def forward(self, index, data_element, model):
        data, meta = data_element
        data = data.cuda()
        for key in meta:
            meta[key] = meta[key].cuda()
        celoss = model(data, meta)
        return celoss
