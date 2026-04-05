import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.jit import Final
from torch.nn import Parameter
from .lstmp import LSTMP
from typing import List
from ..data import clip_mask, cnn2rnn
from typing import Dict, Tuple
# from ..functions import selfatt_enc_dec_mask, selfatt_dec_dec_mask, xavier
from ..functions import xavier
from ..c import *


class AddSOS(nn.Module):
    """

    remove eos in the end of label, and add sos at the start.

    usage:

        input: 
                [meta] which contains original selfatt_label
                [num_class]
        output:
                converted selfatt_label
        attributes:
                [decode] will not perform transformation when set to True

    by pcli2

    """
    decode: bool
    def __init__(self):
        super(AddSOS, self).__init__()
        self.decode = False

    def forward(self, meta: Dict[str, torch.Tensor], num_class: int) -> torch.Tensor:
        if self.decode:
            label = meta["att_label"]
            return label
        else:
            label = meta["att_label"]
            front = torch.zeros((1, label.shape[1]), device=label.device, dtype=label.dtype)
            front.fill_(num_class - 2)
            label = torch.cat((front, label), dim=0)
            label = label[0:-1, :]
            return label



class MaskEmbedding(nn.Module):
    """

    perform label embedding, this module is basicly the same as nn.Embedding, but will ignore -1 

    usage:
        
        parameters:
                [num_embeddings] the quantity of label, usually number of classes.
        input: 
                [input_] label to be embedded.
        output:
                embedding label.

    by pcli2.

    """

    def __init__(self, num_embeddings, embedding_dim):
        super(MaskEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        init.uniform_(self.embedding.weight.data, -0.05, 0.05)


    def forward(self, input_):
        mask, input_ = self.mask(input_)
        out = self.embedding(input_)
        mask = mask.unsqueeze(-1)
        out = out * mask.float()
        return out

    def mask(self, input_) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = input_.clone()
        input_[input_ < 0] = 0
        mask[mask >= 0] = 1
        mask[mask < 0] = 0
        return mask, input_


class MlpAttention(nn.Module):
    num_class: Final[int]
    input_dim: Final[int]
    lstm_cell_num: Final[int]
    lstm_out_dim: Final[int]

    def __init__(self, num_class, input_dim, embedding_dim, mlp_at_dim, lstm_cell_num, lstm_out_dim, clip_threshold=1.0):
        super(MlpAttention, self).__init__()
        self.num_class = num_class
        self.input_dim = input_dim
        self.lstm_cell_num = lstm_cell_num
        self.lstm_out_dim = lstm_out_dim
        self.clip_threshold = float(clip_threshold)
        self.embedding = MaskEmbedding(num_embeddings=num_class, embedding_dim=embedding_dim)
        self.att_lstm = LSTMP(input_dim=embedding_dim + input_dim, cell_num=lstm_cell_num, out_dim=lstm_out_dim, clip_threshold=self.clip_threshold)
        self.att_i2h = nn.Linear(in_features=input_dim, out_features=mlp_at_dim, bias=False)
        self.att_p2s = nn.Linear(in_features=lstm_out_dim, out_features=mlp_at_dim, bias=False)
        self.w_att_v = Parameter(torch.Tensor(1, mlp_at_dim))

        init.uniform_(self.att_i2h.weight.data, -0.05, 0.05)
        init.uniform_(self.att_p2s.weight.data, -0.05, 0.05)
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
        context = torch.zeros(att_b, self.input_dim, device=data.device)  # (5, 512)
        sos = torch.ones((1, att_b), dtype=torch.float32, device=data.device) * (self.num_class - 2)

        att_label = torch.cat((sos, att_label[:-1]), dim=0)

        att_h = self.att_i2h(data)  # 57 5 512
        embed = self.embedding(att_label.long())  # att_label: (29, 5)  embed: (29, 5, 512)

        lstmp_holder = []
        context_holder = []
        for i in range(att_t):  # 29

            body = torch.cat((embed[i], context), dim=1)  # body (5, 1024)
            c_lstm, p_lstm = self.att_lstm.step(body, c_lstm, p_lstm)  # c (5, 2048) p (5, 512)
            lstmp_holder.append(p_lstm)
            state = self.att_p2s(p_lstm)  # state (5, 512)
            out = self.w_att_v * torch.tanh(state.unsqueeze(0) + att_h)  # 57 5 512
            scalar = out.sum(2)  # 57 5
            scalar = scalar + (rnn_mask - 1) * 1e10
            scalar = torch.sigmoid(scalar)
            scalar = scalar.unsqueeze(dim=2)  # (57 5 1)
            context = data * scalar  # (57 5 512)  (57 5 1)
            context = context.sum(dim=0)  # (5, 512)
            context_holder.append(context)

        lstmps = torch.stack(lstmp_holder)  # (29, 5, 512)
        contexts = torch.stack(context_holder)  # (29, 5, 512)
        att_mask = att_mask.unsqueeze(dim=2)  # (29, 5, 1)
        lstmps = lstmps * att_mask
        contexts = contexts * att_mask

        att_fea = torch.cat((lstmps, contexts), dim=2)

        return att_fea

    def step(self, enc_out, cur_label, c_lstm, p_lstm, context, rnn_mask):
        rnn_mask = clip_mask(rnn_mask, enc_out.size(0), 0)
        rnn_t = rnn_mask.shape[0]

        p_encout = self.att_i2h(enc_out)  # (104, 1, 512)
        embed = self.embedding(cur_label).squeeze(1)  # (10,512)

        body = torch.cat((embed, context), dim=1)  # body (10, 1024)
        c_lstm, p_lstm = self.att_lstm.step(body, c_lstm, p_lstm)  # c (10, 2048) p (10, 512)

        state = self.att_p2s(p_lstm)  # state (10, 512)
        out = self.w_att_v * torch.tanh(state.unsqueeze(0) + p_encout)  # 104 10 512
        scalar = out.sum(2)  # 104, 10

        scalar = scalar + (rnn_mask - 1) * 1e10
        scalar = torch.sigmoid(scalar)
        scalar = scalar.unsqueeze(dim=2)  # (104 10 1)

        context = enc_out * scalar  # (104 1 512)  (104 10 1)
        context = context.sum(dim=0)  # (5, 512)
        att_fea = torch.cat((p_lstm, context), dim=1)

        return att_fea, c_lstm, p_lstm, context


class SelfAttention(nn.Module):
    num_head: Final[int]
    attention_dim: Final[int]
    has_encoder: Final[bool]
    decode: bool
    memory_initialized: bool

    def __init__(self, num_head, dim_q, dim_kv, attention_dim, output_dim, has_encoder, dropout=0):
        super(SelfAttention, self).__init__()

        self.num_head = num_head
        self.attention_dim = attention_dim
        self.has_encoder = has_encoder
        self.dropout = dropout

        self.q_trans = nn.Linear(dim_q, attention_dim)
        self.kv_trans = nn.Linear(dim_kv, attention_dim * 2)
        self.fc = nn.Linear(attention_dim, output_dim)
        self.register_buffer("memory", torch.zeros(1))
        xavier(self.q_trans.weight.data)
        init.constant_(self.q_trans.bias.data, 0)
        xavier(self.kv_trans.weight.data)
        init.constant_(self.kv_trans.bias.data, 0)
        xavier(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)
        self.dropout = nn.Dropout(float(dropout))
        self.decode = False
        self.memory_initialized = False


    def forward(self, q: torch.Tensor, kv: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.decode:
            return self.decode_forward(q, kv, meta)
        else:
            return self.train_forward(q, kv, meta)


    def train_forward(self, q: torch.Tensor, kv: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        n, dim_q, _, T_query = q.shape
        _, dim_kv, _, T_memory = kv.shape
        q = q.reshape((n, dim_q, T_query)).permute((0, 2, 1))  # (b, t, d)
        kv = kv.reshape((n, dim_kv, T_memory)).permute((0, 2, 1))

        q = self.q_trans(q)  # (b, t, d)
        kv = self.kv_trans(kv)
        # now actual is (b, num_head, head_dim, t), but num_head is add to head_dim, so we have (b, num_head*head_dim, t). curse caffe
        q = q.permute((0, 2, 1))
        kv = kv.permute((0, 2, 1))

        q_chunk = q.chunk(self.num_head, dim=1)
        kv_chunk = kv.chunk(self.num_head, dim=1)

        out_holder = []

        for i in range(self.num_head):
            q_one_head = q_chunk[i]
            k_one_head, v_one_head = kv_chunk[i].chunk(2, dim=1)
            attn = torch.bmm(q_one_head.permute(0, 2, 1), k_one_head)  # b t d @ b d t -> b t t
            attn = attn * (self.num_head / self.attention_dim)
            if self.has_encoder:
                attn = selfatt_enc_dec_mask(attn, meta, float(-1e10), float(0))
                attn = F.softmax(attn, dim=2)
                attn = selfatt_enc_dec_mask(attn, meta, float(0), float(0))
            if (not self.has_encoder) and (not self.decode):
                attn = selfatt_dec_dec_mask(attn, meta, float(-1e10), float(0))
                attn = F.softmax(attn, dim=2)
                attn = selfatt_dec_dec_mask(attn, meta, float(0), float(0))
            if (not self.has_encoder) and (self.decode):
                attn = F.softmax(attn, dim=2)
            attn = self.dropout(attn)
            output = torch.bmm(v_one_head, attn.permute(0, 2, 1))  # b d t @ b t t -> b d t | (1, 64, 329)
            out_holder.append(output)

        output = torch.cat(out_holder, dim=1)  # (1, 512, 329)

        output = output.permute(0, 2, 1)
        output = self.fc(output)

        b, t, d = output.shape
        output = output.permute((0, 2, 1)).reshape((b, d, 1, t))
        return output


    def decode_forward(self, q: torch.Tensor, kv: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.has_encoder:
            if not self.memory_initialized:
                self.memory = kv.clone()
                self.memory_initialized = True
            else:
                pathid = meta["pathid"].long()
                self.memory = self.memory[pathid, :, :, :]
                self.memory = torch.cat((self.memory, kv), dim=3)
            return self.train_forward(q, self.memory, meta)
        else:
            return self.train_forward(q, kv, meta)


        

class MochaAttention(nn.Module):
    act_type: Final[str]
    mocha_act_type: Final[str]
    mocha_w: Final[int]
    mocha_valid: Final[bool]
    num_class: Final[int]
    lstm_cell_num: Final[int]
    lstm_out_dim: Final[int]
    embedding_dim: Final[int]
    mlp_at_dim: Final[int]
    mocha_threshold: Final[float]
    clip_threshold: Final[float]

    def __init__(self, num_class, input_dim, embedding_dim, mlp_at_dim, lstm_cell_num, lstm_out_dim, 
                 act_type: str, mocha_act_type: str, mocha_w: int, mocha_valid: bool, mocha_noise_var: float=1.0,
                 clip_threshold: float=1.0):
        super(MochaAttention, self).__init__()

        self.att_lstm = LSTMP(embedding_dim + input_dim, lstm_cell_num, lstm_out_dim, clip_threshold=clip_threshold)
        self.embedding = MaskEmbedding(num_class, embedding_dim)
        self.weight_at_s = nn.Parameter(torch.zeros(mlp_at_dim, lstm_out_dim))
        self.weight_at_h = nn.Parameter(torch.zeros(mlp_at_dim, input_dim))
        self.weight_at_v = nn.Parameter(torch.zeros(mlp_at_dim))
        self.weight_chunk_at_s = nn.Parameter(torch.zeros(mlp_at_dim, lstm_out_dim))
        self.weight_chunk_at_h = nn.Parameter(torch.zeros(mlp_at_dim, input_dim))
        self.weight_chunk_at_v = nn.Parameter(torch.zeros(mlp_at_dim))

        



        self.embedding_dim = embedding_dim
        self.mlp_at_dim = mlp_at_dim
        self.lstm_cell_num = lstm_cell_num
        self.lstm_out_dim = lstm_out_dim
        self.act_type = act_type.lower()
        self.mocha_act_type = mocha_act_type.lower()
        self.mocha_w = mocha_w
        self.mocha_noise_var = mocha_noise_var
        self.num_class = num_class
        self.mocha_valid = mocha_valid
        self.clip_threshold = clip_threshold


        init.uniform_(self.weight_at_s, -0.05, 0.05)
        init.uniform_(self.weight_at_h, -0.05, 0.05)
        init.uniform_(self.weight_at_v, -0.05, 0.05)
        init.uniform_(self.weight_chunk_at_s, -0.05, 0.05)
        init.uniform_(self.weight_chunk_at_h, -0.05, 0.05)
        init.uniform_(self.weight_chunk_at_v, -0.05, 0.05)

        self.addsos = AddSOS()



    def forward(self, data: torch.Tensor, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        rnn_mask = meta["rnn_mask"]
        rnn_mask = clip_mask(rnn_mask, data.size(0), 0)
        rnn_mask = rnn_mask.squeeze(2)
        rnn_mask = rnn_mask.permute(1, 0)
        rnn_mask = rnn_mask.contiguous()
        att_mask = meta["att_mask"].contiguous()
        att_label = self.addsos(meta, self.num_class).contiguous()
        data = data.contiguous()
        symbol_T = att_label.size(0)
        D = data.size(2)
        S = data.size(1)
        T = data.size(0)

        # init noise
        at_noise = data.new_zeros(symbol_T, S, T)
        if self.training and self.mocha_noise_var > 0:
            at_noise.normal_(0, self.mocha_noise_var)

        # linear project
        at_h = F.linear(data, self.weight_at_h)
        if self.mocha_valid:
            chunk_at_h = F.linear(data, self.weight_chunk_at_h)

        c_t = data.new_zeros(S, self.lstm_cell_num)
        r_t = data.new_zeros(S, self.lstm_out_dim)
        context_t = data.new_zeros(S, D)
        at_alpha_t = data.new_zeros(S, T)
        at_alpha_t.data[:, 0].fill_(1)
        context = []
        lstm_r = []
        for t in range(symbol_T):           
            # embedding
            lab_t_1 = att_label[t, :]
            embedding_t = self.embedding(lab_t_1.long())
            lstm_x = torch.cat((embedding_t, context_t), dim=1)
            c_t, r_t = self.att_lstm.step(lstm_x, c_t, r_t)
            lstm_r.append(r_t)
            at_state_t = F.linear(r_t, self.weight_at_s)
            at_p_t = mocha_energy(at_state_t, at_h, self.weight_at_v, rnn_mask)

            if self.training and self.mocha_noise_var > 0:
                at_p_t = at_p_t + at_noise[t, :, :]
            if self.mocha_act_type == "sigmoid":
                at_p_t = torch.sigmoid(at_p_t)
            if self.mocha_act_type == "softmax":
                at_p_t = F.softmax(at_p_t, dim=1)
            

            if self.mocha_valid:
                at_cumprod_1mp_log_t = cumprod_1mp(at_p_t)
                at_cumsum_adp_t = cumsum_adp(at_alpha_t, at_cumprod_1mp_log_t)
                at_alpha_t = at_p_t * at_cumsum_adp_t
                chunk_at_state_t = F.linear(r_t, self.weight_chunk_at_s)
                chunk_at_energy_t = mocha_energy(chunk_at_state_t, chunk_at_h, self.weight_chunk_at_v, rnn_mask)
                if self.act_type == "sigmoid":
                    at_scalar_t = window_cumsum_alpha_sigmoid(at_alpha_t, chunk_at_energy_t, self.mocha_w)
                else:
                    at_scalar_t = window_cumsum_exp_alpha(at_alpha_t, chunk_at_energy_t, self.mocha_w)
            else:
                at_scalar_t = at_p_t
            context_t = mocha_context(data, at_scalar_t)
            context.append(context_t)

        lstm_r = torch.stack(lstm_r)
        context = torch.stack(context)
        mocha_attention = torch.cat((lstm_r, context), dim=2)
        att_mask = att_mask.reshape(symbol_T, S, 1)
        mocha_attention.data = mocha_attention.data * att_mask

        return mocha_attention


    @torch.no_grad()
    def step(self, enc_out: torch.Tensor, rnn_mask: torch.Tensor, cur_label: torch.Tensor, 
             c_t_1: torch.Tensor, r_t_1: torch.Tensor, context_vector_t_1: torch.Tensor, at_alpha_t_1: torch.Tensor, 
             mocha_mode: str="sum", th1: float=0.3, th2: float=0.5, mocha_norm_scale: float=10.0):
        rnn_mask = clip_mask(rnn_mask, enc_out.size(0), 0)
        rnn_mask = rnn_mask.squeeze(2)
        rnn_mask = rnn_mask.permute(1, 0)
        rnn_mask = rnn_mask.contiguous()
        enc_out = enc_out.contiguous()
        D = enc_out.size(2)
        S = enc_out.size(1)
        T = enc_out.size(0)

        # linear project
        at_h = F.linear(enc_out, self.weight_at_h)
        if self.mocha_valid:
            chunk_at_h = F.linear(enc_out, self.weight_chunk_at_h)   

        # embedding
        lab_t_1 = cur_label
        embedding_t = self.embedding(lab_t_1.long())
        lstm_x = torch.cat((embedding_t, context_vector_t_1), dim=1)
        c_t, r_t = self.att_lstm.step(lstm_x, c_t_1, r_t_1)
        at_state_t = F.linear(r_t, self.weight_at_s)
        at_p_t = mocha_energy(at_state_t, at_h, self.weight_at_v, rnn_mask)

        if self.mocha_act_type == "sigmoid":
            at_p_t = torch.sigmoid(at_p_t)
        if self.mocha_act_type == "softmax":
            at_p_t = F.softmax(at_p_t, dim=1)


        if self.mocha_valid:
            if mocha_mode == "hard" and th2 > 0 and th2 < 1:
                at_p_t[torch.ge(at_p_t, th2)] = 1
                at_p_t[torch.lt(at_p_t, th2)] = 0


            at_cumprod_1mp_log_t = cumprod_1mp(at_p_t)
            at_cumsum_adp_t = cumsum_adp(at_alpha_t_1, at_cumprod_1mp_log_t)
            at_alpha_t = at_p_t * at_cumsum_adp_t

            if mocha_mode == "sum" and th2 > 0 and th2 < 1:
                at_alpha_t, st_vector, et_vector = sum_hard_attention(at_alpha_t_1, at_alpha_t, th1, th2, mocha_norm_scale)
                
            chunk_at_state_t = F.linear(r_t, self.weight_chunk_at_s)
            chunk_at_energy_t = mocha_energy(chunk_at_state_t, chunk_at_h, self.weight_chunk_at_v, rnn_mask)
            if self.act_type == "sigmoid":
                at_scalar_t = window_cumsum_alpha_sigmoid(at_alpha_t, chunk_at_energy_t, self.mocha_w)
            else:
                at_scalar_t = window_cumsum_exp_alpha(at_alpha_t, chunk_at_energy_t, self.mocha_w)
        else:
            at_scalar_t = at_p_t
        context_t = mocha_context(enc_out, at_scalar_t)
        mocha_attention = torch.cat((r_t, context_t), dim=1)        

        return mocha_attention, c_t, r_t, at_alpha_t, context_t
        