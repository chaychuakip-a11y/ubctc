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

class Ubctc(nn.Module):
    def __init__(self):
        super(Ubctc, self).__init__()
        self.encoder = Encoder()
        #self.decoder = Decoder()
        self.classification = nn.Conv2d(256, 9004, 1, 1, 0)
        xavier(self.classification.weight)
        nn.init.zeros_(self.classification.bias.data)
        self.loss = CTCLoss(9003)
        self.accuracy = AccCtc()

    def forward(self, x: torch.Tensor, meta:Dict[str, torch.Tensor]):
        # att_label = clip_mask(meta["att_label"], self.encoder.concat_fr.nmod, 0)
        meta["rnn_mask"] = clip_mask(meta["rnn_mask"], self.encoder.concat_fr.nmod, 0)
        ctc_label = meta["att_label"].permute(1,0).clone() # att_label:(t,b)
        lstm_mask = meta["rnn_mask"]
        enc_output = self.encoder(x, meta)
        # print('att_label', meta["att_label"].shape)
        # print('ctc_label', ctc_label.shape)
        
        b,d,f,t = enc_output.shape
        enc_output = enc_output.permute((2, 1, 3, 0))
        enc_output = enc_output.reshape((1, d, 1, t*b))

        logit = self.classification(enc_output); # print('logit0', logit.shape)

        if self.training:
            d= logit.shape[1]
            logit = logit.reshape((1,d,-1,b)).squeeze(0).permute((1,2,0)); # print('logit3', logit.shape)
            # print('lstm_mask', lstm_mask.shape)
            input_lengths = lstm_mask.squeeze(2).permute(1,0).sum(-1)
            # print('input_lengths', input_lengths.shape, input_lengths)
            target_lengths = (ctc_label>=0).sum(1)
            # print('target_lengths', target_lengths.shape, target_lengths)

            log_probs = F.log_softmax(logit, dim=-1)
            # print('log_probs', log_probs.shape)
            #### assert shape: log_probs:(t,b,d), ctc_label:(b,t), input_lengths:(b), target_lengths:(b)
            loss = self.loss(log_probs,ctc_label,input_lengths.long(),target_lengths.long())
            # print('loss:', loss)
            return loss, loss, loss
        else:
            logit = logit.squeeze().permute((1, 0))
            acc = self.accuracy(logit, ctc_label)
            return acc, logit
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.concat_fr = ConcatFrLayer(4)
        self.ub1 = LSTMP(160, 1024, 256)
        self.ub2 = LSTMP(256, 1024, 256)
        self.ub3 = UBLSTMP(256, 512, 150, 512, 150, 4, 6)

        self.conv1 = nn.Conv2d(300, 1024, 1, 1, 0)
        xavier(self.conv1.weight)
        nn.init.constant_(self.conv1.bias.data, 0)
        self.relu1 = nn.LeakyReLU(negative_slope=0.0)
        self.conv2 = nn.Conv2d(1024, 256, 1, 1, 0)
        xavier(self.conv2.weight)
        nn.init.constant_(self.conv2.bias.data, 0)
    
    def forward(self, x: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.concat_fr(x)
        x = cnn2rnn(x)
        #lstm_mask = clip_mask(meta["rnn_mask"], self.concat_fr.nmod, 0)
        lstm_mask = meta["rnn_mask"]
        # print(x.shape, lstm_mask.shape) #torch.Size([41, 50, 160]) torch.Size([41, 50, 1])
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
