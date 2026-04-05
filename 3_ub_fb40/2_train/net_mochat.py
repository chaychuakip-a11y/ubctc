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
from conformer import ConformerBlock
import random
import torch.nn.init as init
from densenet_xt import DenseNet
# from warprnnt_pytorch import RNNTLoss
from ctc_fa import CTCForcedAligner
import numpy as np 
from attention import MochaAttention
# from sdt import *

n_class=14839
n_ce_class=15003
n_lid_class=7

def swish(x):
    return x*torch.sigmoid(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CE_Layer(nn.Module):
    def __init__(self, ce_class = n_ce_class):
        super(CE_Layer, self).__init__()

        self.ce_lstm = LSTMP(512, 1024, 512)
        # self.ce_fsmn = FSMN(512, 2048, 512,2, 2, 1, 1, 0)
        self.ce_conv = nn.Conv2d(512, ce_class, 1, 1, 0)

        xavier(self.ce_conv.weight)
        nn.init.zeros_(self.ce_conv.bias.data)

    def forward(self,enc_output,meta):

        ce_x = enc_output.clone() #t/4,b,d

        ce_label = meta["label_ce"]   # b,1,1,t
        # print(ce_label.shape)
        # print(ce_x.shape)
        # exit()
        ce_lstm_mask = meta["rnn_mask"]
        ce_x = self.ce_lstm(ce_x, ce_lstm_mask)

        # mem=None
        # ce_x = ce_x.permute(1,2,0) # B,D,T
        # ce_x,_=self.ce_fsmn(ce_x,mem)       
        # ce_x = ce_x.permute(2,0,1)
        t, b, d = ce_x.shape
        ce_x = ce_x.permute((2, 0, 1))
        ce_x = ce_x.reshape((1, d, 1, t*b))

        ce_logit = self.ce_conv(ce_x)

        ce_label = clip_mask(ce_label, enc_output.size(0), 3)# b,1,1,t/4
        # print(ce_label.shape)
        # print(ce_x.shape)
        # exit()
        ce_logit = ce_logit.squeeze().permute(1,0) #t/4,b
        ce_label = ce_label.reshape(b,t).permute(1,0).flatten()

        return ce_logit,ce_label

class LID_Layer(nn.Module):
    def __init__(self, input_feamap = 512, lid_class = n_lid_class):
        super(LID_Layer, self).__init__()

        self.dnn_lan = nn.Conv2d(input_feamap, 256, 1, 1, 0)
        xavier(self.dnn_lan.weight)
        nn.init.zeros_(self.dnn_lan.bias.data)
        self.classify_lan = nn.Conv2d(256, lid_class, 1, 1, 0)
        xavier(self.classify_lan.weight)
        nn.init.zeros_(self.classify_lan.bias.data)
        

    def forward(self,data,meta):

        lan_label = meta["label_lid"]   # b,1,1,t
        lid_mask = meta["lid_mask"]
        x_lan = self.dnn_lan(data)
        x_lan = self.classify_lan(x_lan)
        # print(x_lan.shape)
        
        b, d, _, t = x_lan.shape
        lid_mask = clip_mask(lid_mask, t, 3)
        lid_mask = lid_mask.reshape(b,d,t).permute((2,0,1)).reshape(t*b,d)
        # x_lan[lid_mask != 1] = -1e8
        # print(lid_mask[0,:,:,20])
        # print(x_lan[0,:,:,20])
        # print(data.shape)
        # print(x_lan.shape)
        lan_logit = x_lan.reshape(b,d,t).permute((2,0,1)).reshape(t*b,d)
        # prob = torch.nn.functional.softmax(lan_logit, dim=1)
        # print(prob.shape)
        # prob[lid_mask != 1] = 0
        # print(lan_label.shape)
        lan_label = clip_mask(lan_label, t, 3) #for label ce
        # lan_label = lan_label.squeeze().permute(1,0)
        lan_label = lan_label.reshape(b,t).permute(1,0)
        lan_label = lan_label.flatten()


        return lan_logit, lan_label
# class LID_Layer(nn.Module):
#     def __init__(self, lid_class = n_lid_class):
#         super(LID_Layer, self).__init__()

#         self.ce_lstm = LSTMP(512, 1024, 512)
#         self.ce_conv = nn.Conv2d(512, lid_class, 1, 1, 0)

#         xavier(self.ce_conv.weight)
#         nn.init.zeros_(self.ce_conv.bias.data)

#     def forward(self,enc_output,meta):

#         ce_x = enc_output.clone() #t/4,b,d

#         # ce_label = meta["label_ctc"]   # t,b
#         ce_label = meta["label_lid"]   # b,1,1,t
#         ce_label = ce_label.squeeze(1).squeeze(1).permute(1,0)

#         ce_lstm_mask = meta["rnn_mask"]
#         ce_x = self.ce_lstm(ce_x, ce_lstm_mask)

#         t, b, d = ce_x.shape
#         ce_x = ce_x.permute((2, 0, 1))
#         ce_x = ce_x.reshape((1, d, 1, t*b))

#         ce_logit = self.ce_conv(ce_x)

#         # print(meta["label_ce"].shape, meta["label_lid"].shape, enc_output.shape)  #torch.Size([20, 1, 1, 144]) torch.Size([36, 20, 512])
#         ce_label = clip_mask(ce_label, enc_output.size(0), 0) #for label_ctc
#         # ce_label = clip_mask(ce_label, enc_output.size(0), 3) #for label ce

#         ce_logit = ce_logit.squeeze().permute(1,0)
#         ce_label = ce_label.flatten()

#         return ce_logit,ce_label

class ConvLN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, height):
        super(ConvLN, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride=stride, padding=padding)
        self.ln=torch.nn.LayerNorm((output_size,height))
        
        nn.init.constant_(self.ln.weight.data,1)
        nn.init.constant_(self.ln.bias.data,0)
        nn.init.xavier_uniform_(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias.data)

    def forward(self, x):
        y=self.conv(x)
        y=self.ln(y.permute(0,3,1,2)).permute(0,2,3,1)
        y=swish(y)
        return y

class CTC_Loss(nn.Module):
    def __init__(self):
        super(CTC_Loss, self).__init__()

        self.ctc_lstm = LSTMP(512, 2048, 512)
        self.ctc_dnn  = nn.Linear(512, 14832)
        self.ctc_fa   = CTCForcedAligner(blank=14839)
        self.ctc_loss = torch.nn.CTCLoss(blank=14839, reduction='mean', zero_infinity=True)

    def forward(self,x,att_label,meta):

        tg_pos = None
        ctc_label = att_label.permute(1,0).clone() ## att_label:(t,b)

        ctc_lstm_mask = meta["rnn_mask"]
        x = self.ctc_lstm(x, ctc_lstm_mask)
        x = self.ctc_dnn(x) # CTC_loss IN->(t b c)

        ctc_lstm_mask = clip_mask(ctc_lstm_mask, x.size(0), 0)
        # print("ctc_lstm_mask",ctc_lstm_mask.size()) #(T,B,1)
        inputs_length = ctc_lstm_mask.squeeze(2).permute(1,0).sum(-1)
        # print("input_lengths",inputs_length.size())

        """ # cal CTC align pos """
        tg_pos = None
        # tg_pos = self.align(x,ctc_label,inputs_length,meta)

        x = F.log_softmax(x,dim=-1)
        
        target_lengths = (ctc_label>=0).sum(1)
        # print("target_lenth",target_lengths)
        
        ctc_loss = self.ctc_loss(x,ctc_label,inputs_length.long(),target_lengths.long())

        #print("ctc_loss",ctc_loss)

        return ctc_loss,tg_pos

    def align(self,x,ctc_label,inputs_length,meta):
        """ 
            input:  [x, input_lens, label] 
            inputs_length: (B,T)
            ctc_label:     (B,U)
            x:             (B,T,D)
            add_eos:       print out eos or not

        """
        """     EXAMPLE
            B,T,H=2,4,3
            x=torch.randn(B,T,H)
            xl=torch.from_numpy(np.array([3,4]))
            y=[[1,2],[2,2,1]]
            tg_pos=self.ctc_fa(x, xl, y,add_eos=False) 
        """

        align_label=[]
        for i in range(ctc_label.size(0)):
            tmp_lab = ctc_label[i]
            tmp_lab = tmp_lab[tmp_lab>=0]
            align_label.append(tmp_lab)
        # print(align_label)

        tg_pos = self.ctc_fa(x.permute(1,0,2),inputs_length.int(),align_label,add_eos=False) 
        tg_pos = tg_pos[:,:-1]
        # print('tg_pos:',tg_pos.size())
        att_mask = meta["att_mask"].permute(1,0)
        # print('att_mask:',att_mask.size())

        tg_pos[att_mask==0]=-1
        
        return tg_pos

class conformer(nn.Module):
    def __init__(self):
        super(conformer, self).__init__()
        self.encoder = Encoder(512)
        self.decoder = Decoder()

        self.ce_layer = CE_Layer()
        # self.lid_layer_dense1 = LID_Layer(1600, 7)
        # self.lid_layer_dense2 = LID_Layer(1040, 7)
        # self.lid_layer_conformer = LID_Layer(512, 7)
        # self.lid_layer = LID_Layer(3)
        
        self.classification = nn.Conv2d(512, n_class, 1, 1, 0)        
        xavier(self.classification.weight)
        nn.init.zeros_(self.classification.bias.data)

        self.lm_classification = nn.Conv2d(512, n_class, 1, 1, 0)
        xavier(self.lm_classification.weight)
        nn.init.zeros_(self.lm_classification.bias.data)

        self.ctc_loss = CTC_Loss()
        self.pos_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.loss = CeLoss()
        self.ce_loss = CeLoss(smooth=1.0)
        self.accuracy = ACC()
    def mask_data(self,data:torch.Tensor):
        b, _, _, w = data.shape
        for i in range(b):
            random.seed()
            index = random.randint(0, w)
            random.seed()
            width = random.randint(0, 100)
            end = min(index + width, w)
            data.data[i, :, :, index:end].fill_(0)
        return data
    def forward(self, x: torch.Tensor, meta:Dict[str, torch.Tensor]):
        att_label = meta["att_label"]   #[t,b]

        # lid_label = meta["label_lid"]
        # print(lid_label.shape)
        # print(x.shape)
        # x shape:B,1,40,T

        # #tfmask        
        if self.training:
            x = self.mask_data(x)
        # x shape:B,1,40,T
        # if self.training:
        #     n_filter=x.shape[2]
        #     if random.random()>0.5:
        #         for i in range(2):
        #             f=random.randint(0,n_filter//4)
        #             f0=random.randint(0,n_filter-f)
        #             x[:,:,f0:f0+f]=0
        
        #             t=random.randint(0,32)
        #             t0=random.randint(0,x.shape[-1]-t)
        #             x[:,:,:,t0:t0+t]=0

        enc_output = self.encoder(x, meta)

        T, B, D = enc_output.size()
        
        am,lm = self.decoder(enc_output, meta) #alps:U,B,T
        # logit = self.classification(dec_output)
        logit = torch.log_softmax(self.classification(am),1)+torch.log_softmax(self.lm_classification(lm),1)
        logit = logit.squeeze().permute((1, 0)) 
        # # logit_am = torch.log_softmax(self.classification(am),1)
        # logit_am = self.classification(am)
        # logit_am = logit_am.squeeze().permute((1, 0)) 
        # # logit_lm = torch.log_softmax(self.lm_classification(lm),1)
        # logit_lm = self.lm_classification(lm)
        # logit_lm = logit_lm.squeeze().permute((1, 0)) 
        ce_logit,ce_label = self.ce_layer(enc_output, meta)
        
        att_label_en = att_label.clone()
        att_label_en[att_label<6735] = -1
        att_label_en[att_label>14834] = -1

        if self.training:

            loss_ce = 0
             
            loss_ce = self.ce_loss(ce_logit, ce_label)
           # print("loss_ce",loss_ce)  
            
            ctc_loss, pos_loss, sdt_loss, loss = 0, 0, 0, 0
            """ CTC LOSS """
            
            # ctc_loss,tg_pos = self.ctc_loss(enc_output,att_label,meta)

            """ POS LOSS """
            # pos_loss = self.pos_loss(alp.permute(1,2,0),tg_pos.to(alp).long())
            # print("pos_loss",pos_loss)

            """ SDT LOSS """             
            # sdt_loss = SDT(self.decoder ,self.classification, self.lm_classification, enc_output, meta)
            # print("sdt_loss",sdt_loss)
            

            """ CE LOSS """
            loss = self.loss(logit, att_label)


            loss_final = loss+0.25*loss_ce
            ### aug en loss
            loss_en = torch.tensor([0])
            if torch.any(att_label_en>-1):
                loss_en = self.loss(logit, att_label_en)
                # loss_final = loss_final + 0.25*loss_en
                loss_final = 0.25*loss_final +loss_en

            # print("cross entropy loss",loss)
            # print(logit.shape, att_label.shape);exit()  #torch.Size([252, 14839]) torch.Size([6, 42])

            # lid_logit, lid_label = self.lid_layer(enc_output, meta) 
            # loss_lid = self.ce_loss(lid_logit, lid_label)
            # print("ed loss")
            # print(loss)
            # print("ce loss")
            # print(loss_ce)
            return loss_final, loss, loss_en
            # return loss + loss_lid
        else:
            acc = self.accuracy(logit, att_label)
            acc_ce = self.accuracy(ce_logit, ce_label)
            acc_en = self.accuracy(logit, att_label_en)
            return acc, acc_en, acc_ce
            #return {"acc": float(acc), "acc": float(acc),"acc_ce": float(acc_ce)}

class Encoder(nn.Module):
    def __init__(self,enc_dim):
        super(Encoder, self).__init__()
        
        self.densenet = DenseNet(growthRate=32, layers=(3,4,5), reduction=0.5,  bottleneck=1)
        for p in self.parameters():
            p.requires_grad = False

        num_head = 8
        attention_dim = 512
        window_width = 15
        dropout_rate=0.1
        self.dropout_rate=dropout_rate
        self.dropout=nn.Dropout(dropout_rate)
                
        n_filter=40
        # self.ln=torch.nn.LayerNorm((n_filter))
        kernel_size=3
        stride=1
        padding=1
        # self.conv1=ConvLN(1, 64, kernel_size, stride, padding, n_filter)
        # self.conv2=ConvLN(64, 64, kernel_size, stride, padding, n_filter)
        # self.conv3=ConvLN(64, 128, kernel_size, stride, padding, n_filter//2)
        # self.conv4=ConvLN(128, 128, kernel_size, stride, padding, n_filter//2)
        # self.pool = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)
        # #self.conv3=ConvLN(64, 128, kernel_size, stride, padding, n_filter//8)
        self.proj=torch.nn.Conv1d(104*10,enc_dim,1,1,0)
        net=[]
        for i in range(16):# 16 -> 12 by xiaobao 20220513
            #net.append(conformer_block(enc_dim, attention_dim, window_width, num_head, pe, dropout_rate))
            block = ConformerBlock(
                dim = enc_dim,
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                lorder = (window_width-1)//2,
                rorder =  (window_width-1)//2,
                attn_dropout = dropout_rate,
                ff_dropout = dropout_rate,
                conv_dropout = dropout_rate,
                causal=False
            )
            net.append(block)
        self.net = nn.Sequential(*net)
        self.enc_lstm = LSTMP(enc_dim, 2048, enc_dim)
        
        # nn.init.constant_(self.ln.weight.data,1)
        # nn.init.constant_(self.ln.bias.data,0)
        nn.init.xavier_uniform_(self.proj.weight.data)
        nn.init.zeros_(self.proj.bias.data)
        # self.enc_conv1 = nn.Linear(enc_dim,enc_dim)
        # self.enc_conv1 = torch.nn.Conv1d(enc_dim, enc_dim, 1, stride=1, padding=0)
        
        # self.enc_conv2 = torch.nn.Conv1d(2048, enc_dim, 1, stride=1, padding=0) 
    
    def forward(self, x: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # print("start")
        #rm layer norm by xiaobao 20220513
        # x=self.ln(x.permute(0,1,3,2)).permute(0,1,3,2)

        # with torch.cuda.amp.autocast():
        x = self.densenet(x)
        
        x=x.reshape(x.shape[0],104*10,x.shape[-1])#B*640*T
        x=self.proj(x)
        #x=F.dropout(x,self.dropout_rate)
        x=self.dropout(x)
        x = x.permute(0, 2, 1)

        conformer_mask = meta["rnn_mask"]
        conformer_mask = clip_mask(conformer_mask, x.size(1),0)
        x,_ = self.net([x,conformer_mask]) # B T D
        # x = self.enc_conv1(x)
        x = x.float()		
        x = x.permute(1, 0, 2)
        # x = self.enc_conv1(x.permute(0,2,1)).permute(2,0,1)
        # x = self.enc_conf2(x).squeeze(1).permute(2,0,1) # B,T,H

        # x = x.float()

        # x = x.permute(1, 0, 2) #B*T*H -> T*B*H
        # #x = x.permute(2, 0, 1) #B*H*T->T*B*H
        enc_lstm_mask = meta["rnn_mask"]
        x = self.enc_lstm(x, enc_lstm_mask)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.mlp_attention = MochaAttention(num_class=n_class, input_dim=512, embedding_dim=512, mlp_at_dim=512, lstm_cell_num=2048, lstm_out_dim=512,
                                           act_type="sigmoid", mocha_act_type="sigmoid", mocha_w=8, mocha_valid=True)
        # self.mlp_attention = MlpAttention(num_class=n_class, input_dim=512, embedding_dim=512, mlp_at_dim=512, lstm_cell_num=2048, lstm_out_dim=512)
        # self.am_dec_lstm = LSTMP(512, 2048, 512)
        # self.lm_dec_lstm = LSTMP(512, 2048, 512)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, x: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        att_mask = meta["att_mask"]
        dec_lstm_mask = att_mask.unsqueeze(2)
        am, lm = self.mlp_attention(x, meta) #alps:U,B,T
        am, lm = am.contiguous(), lm.contiguous()
        # am, lm, alp = self.mlp_attention(x, meta) #alps:U,B,T
        ## am = self.am_dec_lstm(am, dec_lstm_mask)
        ## lm = self.lm_dec_lstm(lm, dec_lstm_mask)

        # print(am.shape)  #torch.Size([5, 42, 512])
        t, b, d = am.shape
        am = am.permute((2, 0, 1))  #[t,b,d]->[d,t,b]
        am = am.reshape((1, d, 1, t*b))
        
        lm = lm.permute((2, 0, 1))
        lm = lm.reshape((1, d, 1, t*b))
        
        return am, lm



class MlpAttention(nn.Module):

    def __init__(self, num_class, input_dim, embedding_dim, mlp_at_dim, lstm_cell_num, lstm_out_dim, clip_threshold=1.0):
        super(MlpAttention, self).__init__()
        self.num_class = num_class
        self.input_dim = input_dim
        self.lstm_cell_num = lstm_cell_num
        self.lstm_out_dim = lstm_out_dim
        self.clip_threshold = float(clip_threshold)
        self.embedding = MaskEmbedding(num_embeddings=num_class, embedding_dim=embedding_dim)
        self.att_lstm = LSTMP(input_dim=embedding_dim, cell_num=lstm_cell_num, out_dim=lstm_out_dim, clip_threshold=self.clip_threshold)
        self.att_i2h = nn.Linear(in_features=input_dim, out_features=mlp_at_dim, bias=False)
        self.att_p2s = nn.Linear(in_features=lstm_out_dim, out_features=mlp_at_dim, bias=False)
        self.att_m = nn.Linear(in_features=input_dim, out_features=mlp_at_dim, bias=False)
        self.w_att_v = nn.Parameter(torch.Tensor(1, mlp_at_dim))

        self.location_conv = nn.Conv1d(1, 8, 21, 1, 10)
        self.location_layer = nn.Conv1d(8, 512, 1, 1, 0)

        init.uniform_(self.att_i2h.weight.data, -0.05, 0.05)
        init.uniform_(self.att_p2s.weight.data, -0.05, 0.05)
        init.uniform_(self.att_m.weight.data, -0.05, 0.05)
        init.zeros_(self.w_att_v.data)

    def forward(self, data: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        att_mask = meta["att_mask"]
        att_label = meta["att_label"]

        rnn_mask = meta["rnn_mask"]
        rnn_mask = clip_mask(rnn_mask, data.size(0), 0)
        rnn_mask = rnn_mask.squeeze(2)
        rnn_t, rnn_b = rnn_mask.shape
        att_t, att_b = att_mask.shape  # (29, 5)

        c_lstm = torch.zeros(att_b, self.lstm_cell_num, device=data.device)
        p_lstm = torch.zeros(att_b, self.lstm_out_dim, device=data.device)
        # context = torch.zeros(att_b, self.input_dim, device=data.device)  # (5, 512)
        sos = torch.ones((1, att_b), dtype=torch.float32, device=data.device) * (self.num_class - 2)

        att_label = torch.cat((sos, att_label[:-1]), dim=0)

        att_h = self.att_i2h(data)  # 57 5 512
        embed = self.embedding(att_label.long())  # att_label: (29, 5)  embed: (29, 5, 512)

        lstmp_holder = []
        context_holder = []
        alp_holder = []
        last_score_sum=torch.zeros(rnn_b, rnn_t, device=data.device) #(b,t)

        for i in range(att_t):  # 29

            # body = torch.cat((embed[i], context), dim=1)  # body (5, 1024)
            body = embed[i]
            c_lstm, p_lstm = self.att_lstm.step(body, c_lstm, p_lstm)  # c (5, 2048) p (5, 512)
            lstmp_holder.append(p_lstm)

            f=self.location_conv(last_score_sum.unsqueeze(1)) #out (b,d,t)
            f=self.location_layer(f)#5 512 57
            f=f.permute(2,0,1) #out (t,b,d)

            state = self.att_p2s(p_lstm)  # state (5, 512)

            if torch.rand(1)>=0.5:
                out= self.w_att_v * torch.tanh(state.unsqueeze(0) + att_h+f) 
            else: 
                out= self.w_att_v * torch.tanh(att_h+f)

            # out = self.w_att_v * torch.tanh(state.unsqueeze(0) + att_h + f)  # 57 5 512
            scalar = out.sum(2)  # 57 5
            scalar = scalar + (rnn_mask - 1) * 1e10
            alpha  = torch.sigmoid(scalar)

            memory=torch.cumsum(alpha.unsqueeze(-1)*data,dim=0)
            memory=F.pad(memory,pad=[0,0,0,0,1,0])[:-1]
            memory=self.att_m(memory)
            if torch.rand(1)>=0.5:
                out= self.w_att_v * torch.tanh(state + att_h+f+memory) 
            else: 
                out= self.w_att_v * torch.tanh(att_h+f+memory)
            scalar = out.sum(2)  # 57 5
            scalar = scalar + (rnn_mask - 1) * 1e10
            scalar = torch.sigmoid(scalar)

            last_score_sum = last_score_sum+scalar.permute(1,0)

            scalar = scalar.unsqueeze(dim=2)  # (57 5 1)
            context = data * scalar  # (57 5 512)  (57 5 1)
            context = context.sum(dim=0)  # (5, 512)
            context_holder.append(context)
            alp_holder.append(scalar.squeeze(2).permute(1,0))

        alps = torch.stack(alp_holder)
        lstmps = torch.stack(lstmp_holder)  # (29, 5, 512)
        contexts = torch.stack(context_holder)  # (29, 5, 512)
        att_mask = att_mask.unsqueeze(dim=2)  # (29, 5, 1)
        lstmps = lstmps * att_mask
        contexts = contexts * att_mask

        # att_fea = torch.cat((lstmps, contexts), dim=2)

        return contexts, lstmps ,alps

    def step(self, enc_out, cur_label, c_lstm, p_lstm, context, rnn_mask,last_score_sum):
        rnn_mask = clip_mask(rnn_mask, enc_out.size(0), 0)

        f=self.location_conv(last_score_sum.unsqueeze(1)) #out (b,d,t)
        f=self.location_layer(f)#5 512 57
        f=f.permute(2,0,1) #out (t,b,d)
        # print('--f',f.size())

        p_encout = self.att_i2h(enc_out)  # (104, 1, 512)
        embed = self.embedding(cur_label).squeeze(1)  # (10,512)

        # body = torch.cat((embed, context), dim=1)  # body (10, 1024)
        body = embed
        c_lstm, p_lstm = self.att_lstm.step(body, c_lstm, p_lstm)  # c (10, 2048) p (10, 512)

        state = self.att_p2s(p_lstm)  # state (10, 512)

        out = self.w_att_v * torch.tanh(state.unsqueeze(0) + p_encout + f)  # 104 10 512
        scalar = out.sum(2)  # 104, 10

        scalar = scalar + (rnn_mask - 1) * 1e10
        alpha = torch.sigmoid(scalar)

        memory=torch.cumsum(alpha.unsqueeze(-1)*enc_out,dim=0)
        memory=F.pad(memory,pad=[0,0,0,0,1,0])[:-1]
        memory=self.att_m(memory)
        out=self.w_att_v * torch.tanh(state + p_encout+f+memory)
        scalar = out.sum(2)  # 57 5
        scalar = scalar + (rnn_mask - 1) * 1e10
        scalar = torch.sigmoid(scalar)
        beta   = scalar
        # print("scalar",scalar.argmax(0))

        last_score_sum = last_score_sum+scalar.permute(1,0)

        scalar = scalar.unsqueeze(dim=2)  # (104 10 1)

        context = enc_out * scalar  # (104 1 512)  (104 10 1)
        context = context.sum(dim=0)  # (5, 512)
        # att_fea = torch.cat((p_lstm, context), dim=1)

        return context, c_lstm, p_lstm,last_score_sum,alpha,beta
class FSMN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lorder, rorder, dilation_rate, strides, dropout):
        #input_size ����ά��        #hidden_size ����ά��        #output_size ���ά��
        #lorder �����        #rorder �ҽ���        #strides ������δʹ��
        
        super(FSMN, self).__init__()
        self.lorder=lorder
        self.rorder=rorder
        self.mem = torch.nn.Conv1d(input_size, input_size, lorder+rorder+1, stride=1, padding=0, groups=input_size)
        self.conv1 = torch.nn.Conv1d(input_size, hidden_size, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(hidden_size, output_size, 1, stride=1, padding=0)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        
        xavier(self.mem.weight)
        nn.init.constant_(self.mem.bias.data, 0)
        xavier(self.conv1.weight)
        nn.init.constant_(self.conv1.bias.data, 0)
        xavier(self.conv2.weight)
        nn.init.constant_(self.conv2.bias.data, 0)

    def forward(self, features,last_mem):

        # features=F.pad(features,(self.lorder,self.rorder,0,0,0,0),"constant",value=0)
        # mem = self.mem(features)
        
        # if last_mem is not None:
        #     mem+=last_mem
        xs=self.relu(self.conv1(features))
        
        xs=self.conv2(xs)

        return xs, None
