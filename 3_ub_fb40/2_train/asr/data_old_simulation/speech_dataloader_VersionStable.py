import copy
import math
from sys import flags
from numpy.core.fromnumeric import squeeze
from numpy.core.numeric import ones
from torch._C import dtype
from .datum_pb2 import SimpleDatum, SpeechDatum
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import numpy as np
from typing import Sequence
from operator import index, itemgetter
from itertools import groupby, repeat, chain
import os
import sys
import scipy.io as sio
import delta
import wave
from library.utils import *

# # 【测试样例1】
# print(len(data_list))
# for i, item in enumerate(data_list):
#     write_audio(item, "./test/ori_data"+str(i)+".wav")
# # 【测试样例2】
# print(len(vocab_list))
# for i, item in enumerate(vocab_list):
#     for j, subitem in enumerate(item):
#         np.savetxt('./test/ori_data'+str(i)+"_"+str(j)+".txt", subitem, fmt="%d")
# print("Done")
# sys.exit()

# 全局变量
noneid_ivw0=3001
noneid_ivw1=9001
noneid_csk0=3003
noneid_csk1=15003

def func_calCmdBoundary(lmdb_cmdlist, train_e2e_winsize):
    cmd_len_mask = None
    if lmdb_cmdlist != None:
        file_lines = []
        with open(lmdb_cmdlist, 'r') as f:
            for line in f.readlines():
                if line.strip() != "":
                    file_lines.append(line.strip())
        cmd_len_mask = torch.zeros(len(file_lines), train_e2e_winsize//8)
        index = 0
        max_len = 0
        for line in file_lines:
            content = line.strip().split("\t")[0].split('-')
            length  = len(content)
            if length > max_len:
                max_len = length
            if len(content) == 1: #silnoisenone
                length = train_e2e_winsize//8
            ## 浮点: 
            cmd_len_mask[index][train_e2e_winsize//8-length:] = (train_e2e_winsize//8)/length
            ## 量化Q13: 8192, Q11: 2048
            # cmd_len_mask[index][train_e2e_winsize//8-length:] = math.floor( (train_e2e_winsize//8) /length * 8192 + 0.5) / 8192
            index   += 1
    return cmd_len_mask

class SpeechDatasetEval(Dataset):
    def __init__(self, lmdb_path, lmdb_key, lmdb_normfile, max_sent_frame=100000000, start_line=0, end_line=None, train_padhead_frame=0, train_padtail_frame=0, world_size=1, rank=0):
        self.lmdb_path = lmdb_path
        self.lmdb_key  = lmdb_key
        self.lmdb_env  = None
        self.train_padhead_frame = train_padhead_frame
        self.train_padtail_frame = train_padtail_frame
        self.max_sent_frame      = max_sent_frame
        self.data_keys = []
        self.data_lens = []
        self.data_id   = []
        for i,line in enumerate(open(lmdb_key)):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            if (i - rank) % world_size == 0:
                items      = line.strip().split()
                sent_frame = int(items[1])
                sent_id    = int(items[2])
                if sent_frame <= self.max_sent_frame:
                    self.data_keys.append(items[0])
                    self.data_lens.append(sent_frame)
                    self.data_id.append(sent_id)
        with open(lmdb_normfile, 'r') as nf:
            normfile_lines = nf.read().splitlines()
        self.mel_spec_mean = np.array([float(item) for item in normfile_lines[1:41]])
        self.mel_spec_std  = np.array([float(item) for item in normfile_lines[42:82]])

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        txn = self.lmdb_env.begin()
        with txn.cursor() as cursor:
            k = self.data_keys[index].encode('utf-8')
            cursor.set_key(k)
            datum = SpeechDatum()
            datum.ParseFromString(cursor.value())
            data     = np.fromstring(datum.anc.data, dtype=np.int16)
            label    = np.fromstring(datum.anc.state_data, dtype=np.int16).reshape(-1, 2)
            syllable = np.fromstring(datum.anc.syllable_data, dtype=np.int16).reshape(-1, 1)
            vocab    = np.fromstring(datum.anc.vocab_data, dtype=np.int16).reshape(-1, 1)
            end2end  = np.fromstring(datum.anc.e2e_data, dtype=np.int16).reshape(-1, 1)
        
        ## [提特征/feanorm] ##
        data = delta.build_fb40([data])
        data = (data - self.mel_spec_mean[None,:]) * self.mel_spec_std

        pad_head = self.train_padhead_frame
        pad_tail = self.train_padtail_frame
        if int(pad_head) > 0 or int(pad_tail) > 0:
            data_head = np.zeros((pad_head, 40), dtype="float32")
            data_tail = np.zeros((pad_tail, 40), dtype="float32")
            label_head = np.ones( (pad_head, 2), dtype="int16")*(-2)
            label_tail = np.ones( (pad_tail, 2), dtype="int16")*(-2)
            tmp_head = np.ones( (pad_head, 1), dtype="int16")*(-2)
            tmp_tail = np.ones( (pad_tail, 1), dtype="int16")*(-2)
            data = np.concatenate((data_head, data, data_tail), axis=0)
            label = np.concatenate((label_head, label, label_tail), axis=0)
            syllable = np.concatenate((tmp_head, syllable, tmp_tail), axis=0)
            end2end = np.concatenate((tmp_head, end2end, tmp_tail), axis=0)

        ## [帧数对齐] ##
        freame_len = np.shape(data)[0]
        label      = label[:freame_len, :]
        syllable   = syllable[:freame_len, :]
        end2end    = end2end[:freame_len, :]

        return [data], [label], [syllable], [vocab], [end2end]

class SpeechDatasetTrain(Dataset):
    def __init__(self, lmdb_path, lmdb_key, lmdbpb_path, lmdbpb_key, lmdbuni_path, lmdbuni_key, lmdbunipb_path, lmdbunipb_key, lmdbnone_path, lmdbnone_key, lmdbnoise_path, lmdbnoise_key, lmdbextra_path, lmdbextra_key, lmdbextrapb_path, lmdbextrapb_key, max_sent_frame=100000000, start_line=0, end_line=None, end_expand=None, num_cmds=0, train_sent_main=0, train_sent_pb=0, train_ratio_uni=0, train_ratio_unipb=0, train_ratio_none=0, train_ratio_extra=0, train_ratio_extrapb=0, train_padhead_frame=0, train_padtail_frame=0, world_size=1, rank=0, seed=None, is_ivw=False, is_mixup=False, mixup_cmdid=None, train_mainkwsid=None,  is_mc=False, train_ratio_inter=0, lmdbinter_path=None, lmdbinter_key=None, lmdbinterpb_path=None, lmdbinterpb_key=None, lmdbrir_path=None, lmdbrir_key=None, lmdbmusic_path=None, lmdbmusic_key=None, lmdbmcnoise_path=None, lmdbmcnoise_key=None):
        # 随机种子
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        
        # 初始化
        self.lmdb_path        = lmdb_path
        self.lmdb_key         = lmdb_key
        self.lmdbpb_path      = lmdbpb_path
        self.lmdbpb_key       = lmdbpb_key
        self.lmdbuni_path     = lmdbuni_path
        self.lmdbuni_key      = lmdbuni_key
        self.lmdbunipb_path   = lmdbunipb_path
        self.lmdbunipb_key    = lmdbunipb_key
        self.lmdbnone_path    = lmdbnone_path
        self.lmdbnone_key     = lmdbnone_key
        self.lmdbnoise_path   = lmdbnoise_path
        self.lmdbnoise_key    = lmdbnoise_key
        self.lmdbextra_path   = lmdbextra_path
        self.lmdbextra_key    = lmdbextra_key
        self.lmdbextrapb_path = lmdbextrapb_path
        self.lmdbextrapb_key  = lmdbextrapb_key

        self.num_cmds            = num_cmds
        self.train_ratio_uni     = train_ratio_uni
        self.train_ratio_unipb   = train_ratio_unipb
        self.train_ratio_none    = train_ratio_none
        self.train_ratio_extra   = train_ratio_extra
        self.train_ratio_extrapb = train_ratio_extrapb
        self.train_padhead_frame = train_padhead_frame
        self.train_padtail_frame = train_padtail_frame

        self.is_ivw      = is_ivw
        if self.is_ivw:
            self.noneid0  = noneid_ivw0
            self.noneid1  = noneid_ivw1
        else:
            self.noneid0  = noneid_csk0
            self.noneid1  = noneid_csk1
        self.is_mixup    = is_mixup
        self.mixup_cmdid = [int(i) for i in mixup_cmdid.split(",")]
        
        self.lmdb_env     = None
        self.lmdb_pb      = None
        self.lmdb_uni     = None
        self.lmdb_unipb   = None
        self.lmdb_none    = None
        for noise_index in range(len(lmdbnoise_key)):
            setattr(self, "lmdb_noise_"+str(noise_index), None)
        self.lmdb_extra   = None
        self.lmdb_extrapb = None
        
        self.rank            = rank
        self.world_size      = world_size
        self.start_line      = start_line
        self.end_line        = end_line
        self.end_expand      = end_expand
        self.max_sent_frame  = max_sent_frame
        self.train_sent_main = train_sent_main
        self.train_sent_pb   = train_sent_pb

        self.data_keys            = []
        self.data_lens            = []
        self.data_cmdid           = []
        self.data_types           = []
        self.datauni_keys         = []
        self.datauni_indexs       = []
        self.dataunipb_keys       = []
        self.dataunipb_indexs     = []
        self.datanone_keys        = []
        self.datanone_indexs      = []
        self.datanoise_keysDict   = {}
        self.dataextra_keys       = []
        self.dataextra_indexs     = []
        self.dataextrapb_keys     = []
        self.dataextrapb_indexs   = []

        # 初始化通用数据和噪声数据位置指针
        self.len_uni = 0
        self.pointer_uni = 0
        self.pointer_uni_bg = 0
        self.pointer_uni_ed = 0

        self.len_unipb = 0
        self.pointer_unipb = 0
        self.pointer_unipb_bg = 0
        self.pointer_unipb_ed = 0

        self.len_none = 0
        self.pointer_none = 0
        self.pointer_none_bg = 0
        self.pointer_none_ed = 0

        for noise_index in range(len(lmdbnoise_path)):
            setattr(self, "len_noise_"+str(noise_index), 0)
            setattr(self, "pointer_noise_"+str(noise_index), 0)
            setattr(self, "pointer_noise_bg_"+str(noise_index), 0)
            setattr(self, "pointer_noise_ed_"+str(noise_index), 0)

        self.len_extra = 0
        self.pointer_extra = 0
        self.pointer_extra_bg = 0
        self.pointer_extra_ed = 0

        self.len_extrapb = 0
        self.pointer_extrapb = 0
        self.pointer_extrapb_bg = 0
        self.pointer_extrapb_ed = 0

        # 主唤醒id和数据量
        self.mainkwsid_dict = {}
        if train_mainkwsid is not None:
            mainkwsid_list = train_mainkwsid.split(",")
            for item in mainkwsid_list:
                item_list = [int(i) for i in item.split(":")]
                assert(len(item_list)==3)
                self.mainkwsid_dict[item_list[0]]=item_list[1:]

        # 命令词仿真+回放数据 List
        data_keys_expand = self.__lmdbkey_expand__(lmdb_key, lmdbpb_key)
        self.__genKeysList_expand__(data_keys_expand) # 扩展后的命令词数据按GPU卡数分配

        # 通用仿真+回放数据 List
        if self.lmdbuni_key != None:
            self.datauni_keys, self.datauni_indexs = self.__genKeyList_Lmdb__(self.lmdbuni_key) # 通用仿真数据按GPU卡号分配
            self.len_uni = len(self.datauni_keys)
        if self.lmdbunipb_key != None:
            self.dataunipb_keys, self.dataunipb_indexs = self.__genKeyList_Lmdb__(self.lmdbunipb_key) # 通用回放数据按GPU卡号分配
            self.len_unipb = len(self.dataunipb_keys)

        # 反例数据 List
        if self.lmdbnone_key != None:
            self.datanone_keys, self.datanone_indexs = self.__genKeyList_Lmdb__(self.lmdbnone_key) # 反例数据不按GPU卡号分配
            self.len_none = len(self.datanone_keys)
        
        # 噪声数据 Dict
        if self.lmdbnoise_key != None:
            self.__genKeysDict_noiselmdb__(self.lmdbnoise_key) # 每个噪声类型按GPU卡号分配
            for key, value in self.datanoise_keysDict.items():
                setattr(self, "len_noise_"+str(key), len(value)) # 初始化每个噪声类型在每张GPU卡上的数据量
        self.typesList_noise = list(range(len(self.datanoise_keysDict.keys())))

        # 额外数据接口仿真+回放 List
        if self.lmdbextra_key != None:
            self.dataextra_keys, self.dataextra_indexs = self.__genKeyList_Lmdb__(self.lmdbextra_key) # 通用仿真数据按GPU卡号分配
            self.len_extra = len(self.dataextra_keys)
        if self.lmdbextrapb_key != None:
            self.dataextrapb_keys, self.dataextrapb_indexs = self.__genKeyList_Lmdb__(self.lmdbextrapb_key) # 通用回放数据按GPU卡号分配
            self.len_extrapb = len(self.dataextrapb_keys)

        # 多通道
        self.is_mc          = is_mc
        self.lmdbinter_path   = lmdbinter_path
        self.lmdbinter_key    = lmdbinter_key
        self.lmdbinterpb_path = lmdbinterpb_path
        self.lmdbinterpb_key  = lmdbinterpb_key
        self.lmdbrir_path     = lmdbrir_path
        self.lmdbrir_key      = lmdbrir_key
        self.lmdbmusic_path   = lmdbmusic_path
        self.lmdbmusic_key    = lmdbmusic_key
        self.lmdbmcnoise_path   = lmdbmcnoise_path
        self.lmdbmcnoise_key    = lmdbmcnoise_key

        self.train_ratio_inter  = train_ratio_inter

        self.lmdb_inter     = None
        self.lmdb_interpb   = None
        self.lmdb_rir       = None
        self.lmdb_music     = None
        self.lmdb_mcnoise   = None

        self.datainter_keys     = []
        self.datainter_indexs   = []
        self.datainterpb_keys   = []
        self.datainterpb_indexs = []
        self.datarir_keys       = []
        self.datarir_indexs     = []
        self.datamusic_keys     = []
        self.datamusic_indexs   = []
        self.datamcnoise_keys   = []
        self.datamcnoise_indexs = []

        self.len_inter          = 0
        self.pointer_inter      = 0
        self.pointer_inter_bg   = 0
        self.pointer_inter_ed   = 0

        self.len_interpb        = 0
        self.pointer_interpb    = 0
        self.pointer_interpb_bg = 0
        self.pointer_interpb_ed = 0

        self.len_rir            = 0
        self.pointer_rir        = 0
        self.pointer_rir_bg     = 0
        self.pointer_rir_ed     = 0

        self.len_music          = 0
        self.pointer_music      = 0
        self.pointer_music_bg   = 0
        self.pointer_music_ed   = 0

        self.len_mcnoise         = 0
        self.pointer_mcnoise     = 0
        self.pointer_mcnoise_bg  = 0
        self.pointer_mcnoise_ed  = 0

        if self.lmdbinter_key != None:
            self.datainter_keys, self.datainter_indexs = self.__genKeyList_Lmdb__(self.lmdbinter_key, len_limit=False)          # 数据按GPU卡号分配
            self.len_inter = len(self.datainter_keys)
        if self.lmdbinterpb_key != None:
            self.datainterpb_keys, self.datainterpb_indexs = self.__genKeyList_Lmdb__(self.lmdbinterpb_key, len_limit=False)    # 数据按GPU卡号分配
            self.len_interpb = len(self.datainterpb_keys)
        if self.lmdbrir_key != None:
            self.datarir_keys, self.datarir_indexs = self.__genKeyList_Lmdb__(self.lmdbrir_key, len_limit=False)                # 数据按GPU卡号分配
            self.len_rir = len(self.datarir_keys)
        if self.lmdbmusic_key != None:
            self.datamusic_keys, self.datamusic_indexs = self.__genKeyList_Lmdb__(self.lmdbmusic_key, len_limit=False)          # 数据按GPU卡号分配
            self.len_music = len(self.datamusic_keys)
        if self.lmdbmcnoise_key != None:
            self.datamcnoise_keys, self.datamcnoise_indexs = self.__genKeyList_Lmdb__(self.lmdbmcnoise_key, len_limit=False)          # 数据按GPU卡号分配
            self.len_mcnoise = len(self.datamcnoise_keys)

    def __len__(self):
        return len(self.data_keys)

    def __lmdbkey_expand__(self, lmdb_key, lmdbpb_key):
        # 命令词仿真数据
        key_dict = {}
        for i, line in enumerate(open(lmdb_key)):
            line = line.strip()
            if i < self.start_line or line == "":
                continue
            items = line.split()
            sent_frame = int(items[1]) # 句子帧数
            sent_id = int(items[2]) #句子词id
            if sent_frame <= self.max_sent_frame:
                if sent_id in key_dict.keys():
                    key_dict[sent_id].append(np.array([int(items[0]), sent_frame, sent_id, 1]))
                else:
                    key_dict[sent_id] = [np.array([int(items[0]), sent_frame, sent_id, 1])]

        # 命令词回放数据
        keypb_dict = {}
        if lmdbpb_key is not None:
            for i, line in enumerate(open(lmdbpb_key)):
                line = line.strip()
                if i < self.start_line or line == "":
                    continue
                items = line.split()
                sent_frame = int(items[1]) # 句子帧数
                sent_id = int(items[2]) #句子词id
                if sent_frame <= self.max_sent_frame:
                    if sent_id in keypb_dict.keys():
                        keypb_dict[sent_id].append(np.array([int(items[0]), sent_frame, sent_id, 0]))
                    else:
                        keypb_dict[sent_id] = [np.array([int(items[0]), sent_frame, sent_id, 0])]

        # 命令词仿真和回放数据混合
        key_list = []
        train_sent_main = self.train_sent_main
        train_sent_pb = self.train_sent_pb
        assert(train_sent_main >= train_sent_pb)
        assert(self.num_cmds >= 1)

        # 定义命令词边界 浅定制只有0
        if self.num_cmds == 1:
            range_bg = 0
            range_ed = self.num_cmds
        else:
            range_bg = 1
            range_ed = self.num_cmds

        for word_id in range(range_bg, range_ed):
            
            # 主唤醒词N倍数据量
            if word_id in self.mainkwsid_dict.keys(): 
                wkmain_value = self.mainkwsid_dict[word_id]
                train_sent_main = self.train_sent_main * wkmain_value[0]
                train_sent_pb = self.train_sent_pb * wkmain_value[1]
			
            # Case 1: 仿真和回放数据均存在
            if word_id in key_dict.keys() and word_id in keypb_dict.keys(): 
                remainder = train_sent_main
                # 获取回放数据
                valuepb = keypb_dict[word_id]
                keypb_len = len(valuepb)
                if train_sent_pb > 0:
                    remainder = max(0, train_sent_main - train_sent_pb)
                    ratio_pb = train_sent_pb // keypb_len + 1
                    if ratio_pb > 1:
                        valuepb = np.tile( np.array(valuepb), (ratio_pb, 1) )
                    self.rng.shuffle(valuepb)
                    key_list.extend( valuepb[ : train_sent_pb] )

                # 获取仿真数据
                value = key_dict[word_id]
                key_len = len(value)
                ratio = remainder // key_len + 1
                if ratio > 1:
                    value = np.tile(np.array(value), (ratio, 1))
                if remainder > 0:
                    self.rng.shuffle(value)
                    key_list.extend(value[: remainder])

            # Case 2: 仿真数据存在，回放数据不存在
            elif word_id in key_dict.keys() and word_id not in keypb_dict.keys():
                # 获取仿真数据
                value = key_dict[word_id]
                key_len = len(value)
                ratio = train_sent_main // key_len + 1
                if ratio > 1:
                    value = np.tile(np.array(value), (ratio, 1))
                self.rng.shuffle(value)
                key_list.extend(value[: train_sent_main])
            
            # Case 3: 仿真数据不存在，回放数据存在
            elif word_id not in key_dict.keys() and word_id in keypb_dict.keys():
                # 获取回放数据
                valuepb = keypb_dict[word_id]
                keypb_len = len(valuepb)
                ratio_pb = train_sent_main // keypb_len + 1
                if ratio_pb > 1:
                    valuepb = np.tile( np.array(valuepb), (ratio_pb, 1) )
                self.rng.shuffle(valuepb)
                key_list.extend( valuepb[ : train_sent_main] )

            # 重置仿真回放数据比
            train_sent_main = self.train_sent_main
            train_sent_pb = self.train_sent_pb

        self.rng.shuffle(key_list)
        return key_list

    def __genKeysList_expand__(self, keys_expand):
        for i, items in enumerate(keys_expand):
            if i < self.start_line:
                continue
            if self.end_expand is not None and i >= self.end_expand:
                break
            if (i - self.rank) % self.world_size == 0:
                sent_frame = int(items[1])
                if sent_frame <= self.max_sent_frame:
                    self.data_keys.append(int(items[0]))
                    self.data_lens.append(sent_frame)
                    self.data_cmdid.append(int(items[2]))
                    self.data_types.append(int(items[3]))

    def __genKeyList_Lmdb__(self, lmdb_key, len_limit=True):
        index_tmp = 0
        data_keys_out   = []
        data_indexs_out = []
        for i, line in enumerate(open(lmdb_key)):
            line = line.strip()
            if line=="":
                continue
            items = line.split()
            if (i - self.rank) % self.world_size == 0:
                sent_frame = int(items[1])
                if len_limit:
                    if sent_frame <= self.max_sent_frame:
                        data_keys_out.append(int(items[0]))
                        data_indexs_out.append(index_tmp)
                        index_tmp += 1
                else:
                    data_keys_out.append(int(items[0]))
                    data_indexs_out.append(index_tmp)
                    index_tmp += 1
        return data_keys_out, data_indexs_out
    
    def __genKeysDict_noiselmdb__(self, lmdb_key, min_keylen=4096):
        for index_keys, path_keys in enumerate(lmdb_key):
            lines  = open(path_keys).readlines()
            length = len(lines)
            if length<min_keylen:
                ratio_noise = min_keylen // length + 1
                lines  = np.tile( np.array(lines), ratio_noise )
                self.rng.shuffle(lines)
                length = len(lines)
            for i, line in enumerate(lines):
                line = line.strip()
                if line == "":
                    continue
                items = line.split()
                if (i - self.rank) % self.world_size == 0:
                    if index_keys in self.datanoise_keysDict.keys():
                        self.datanoise_keysDict[index_keys].append(int(items[0]))
                    else:
                        self.datanoise_keysDict[index_keys] = [int(items[0])]

    def __readLmdb__(self, lmdbname, data_index, index_list, pad_head=0, pad_tail=0, randpad=False, max_len=100000, rand_max=True):
        data_out     = []
        label_out    = []
        syllable_out = []
        vocab_out    = []
        end2end_out  = []

        ## [扰动pad_head长度] ##
        if randpad and pad_head>=30:
            pad_head = self.rng.integers(30, pad_head)

        ## [读取lmdb] ##
        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(data_index[index]).zfill(12).encode('utf-8')
                cursor.set_key(k)
                datum = SpeechDatum()
                datum.ParseFromString(cursor.value())
                data     = np.fromstring(datum.anc.data, dtype=np.int16)
                label    = np.fromstring(datum.anc.state_data, dtype=np.int16).reshape(-1, 2)
                syllable = np.fromstring(datum.anc.syllable_data, dtype=np.int16).reshape(-1, 1)
                vocab    = np.fromstring(datum.anc.vocab_data, dtype=np.int16).reshape(-1, 1)
                end2end  = np.fromstring(datum.anc.e2e_data, dtype=np.int16).reshape(-1, 1)

                ## [大于max长度语音截断] ##
                wave_len = np.shape(data)[0]
                frames_len = np.shape(label)[0]
                if frames_len > max_len:
                    if rand_max:
                        cut_bg = self.rng.integers(0, frames_len-max_len)
                        data = data[cut_bg*160 : (cut_bg + max_len)*160]
                        label = label [cut_bg : (cut_bg + max_len), :]
                    else:
                        data = data[-(max_len*160) : ]
                        label = label [-max_len:, :]
                    wave_len = np.shape(data)[0]
                    frames_len = np.shape(label)[0]
                
                ## [数据pad] ##
                if pad_head>0 or pad_tail>0:
                    index_bg     = pad_head*160
                    index_ed     = pad_tail*160
                    data_cat     = np.zeros( (wave_len+index_bg+index_ed), dtype="int16" )
                    label_cat    = np.ones( (frames_len+pad_head+pad_tail, 2), dtype="int16")*(-2)      
                    syllable_cat = np.ones( (frames_len+pad_head+pad_tail, 1), dtype="int16")*(-2)
                    end2end_cat  = np.ones( (frames_len+pad_head+pad_tail, 1), dtype="int16")*(-2)
                    data_cat[index_bg: wave_len+index_bg]          = data
                    label_cat[pad_head: frames_len+pad_head, :]    = label
                    syllable_cat[pad_head: frames_len+pad_head, :] = syllable
                    end2end_cat[pad_head: frames_len+pad_head, :]  = end2end
                else:
                    data_cat     = data
                    label_cat    = label
                    syllable_cat = syllable
                    end2end_cat  = end2end
                
                # vocab非帧对齐不需要pad
                vocab_cat    = vocab

                ## [数据输出] ##
                data_out.append(data_cat)
                label_out.append(label_cat)
                syllable_out.append(syllable_cat)
                vocab_out.append(vocab_cat)
                end2end_out.append(end2end_cat)
        return data_out, label_out, syllable_out, vocab_out, end2end_out

    def __readLmdbMC__(self, lmdbname, data_index, index_list, v_dim, h_dim):
        data_out  = []
        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(data_index[index]).zfill(12).encode('utf-8')           
                cursor.set_key(k)
                datum = SpeechDatum()
                datum.ParseFromString(cursor.value())
                data = np.frombuffer(datum.anc.data, dtype = np.int16).reshape(v_dim, h_dim)      
                data_out.append(data)
        return data_out

    def __maskBoundInfo__(self, label):
        label = np.copy(label[:, 0])
        label_uniq = []
        count_uniq = []
        node_th = None
        for key, value in groupby(list(label)):
            label_uniq.append(key)
            count_uniq.append(len(list(value)))
        
        ## [参数初始化] ##
        node_th           = self.noneid0-1
        label_uniq_filter = list(filter(lambda x:x<node_th and x>=0, label_uniq))
        if self.is_ivw:
            mask_num          = self.rng.integers(2, 5)
        else:
            mask_num          = self.rng.integers(6, 15)
        mask_start  = self.rng.integers(0, len(label_uniq_filter)-mask_num)
        
        ## [获取mask端点] ##
        index       = 0
        bound_start = 0
        bound_end   = 0
        for id in range(len(label_uniq)):
            cur_value = label_uniq[id]
            cur_count = count_uniq[id]

            if cur_value >= node_th or cur_value < 0:
                bound_start += cur_count
                bound_end += cur_count
            else:
                if index < mask_start:
                    bound_start += cur_count                  
                if index < mask_start + mask_num:
                    bound_end += cur_count
                if index >= mask_start + mask_num:
                    break
                index += 1
        
        ## [是否对mask部分缩放] ##
        del_flag  = self.rng.integers(0, 9)
        if del_flag % 2 == 0:
            bound_del = self.rng.integers(bound_start, bound_end)
        else:
            bound_del = bound_end
        return bound_start, bound_end, bound_del

    def __getitem__(self, index):
        # 仿真数据
        data_out       = []
        label_out      = []
        syllable_out   = []
        vocab_out      = []
        end2end_out    = []
        
        # 回放数据
        datapb_out     = []
        labelpb_out    = []
        syllablepb_out = []
        vocabpb_out    = []
        end2endpb_out  = []

        # 反例数据
        datanone_out     = []
        labelnone_out    = []
        syllablenone_out = []
        vocabnone_out    = []
        end2endnone_out  = []

        # 噪声数据
        datanoise_out  = []
        
        # 命令词数据
        sim_online = int(self.data_types[index])
        cmd_id = int(self.data_cmdid[index])
         # 命令词仿真数据
        if sim_online:
            data_cmd, label_cmd, syllable_cmd, vocab_cmd, end2end_cmd = self.__readLmdb__(self.lmdb_env, self.data_keys, [index])
            data_out     += data_cmd
            label_out    += label_cmd
            syllable_out += syllable_cmd
            vocab_out    += vocab_cmd
            end2end_out  += end2end_cmd
            # mask + mixup 构造反例
            if self.is_mixup:
                if index % 3 == 0 and cmd_id in self.mixup_cmdid:
                    bound_start, bound_end, bound_del = self.__maskBoundInfo__(label_cmd[0])
                    data_reverse     = data_cmd[0].copy().reshape(-1, 160)
                    label_reverse    = label_cmd[0].copy().reshape(-1, 1)
                    syllable_reverse = syllable_cmd[0].copy().reshape(-1, 1)
                    assert(np.shape(data_reverse)[0] == np.shape(label_reverse)[0] == np.shape(syllable_reverse)[0])
                    if bound_del < bound_end:
                        range_del        = list(range(bound_del, bound_end))
                        data_reverse     = np.delete(data_reverse, range_del, axis=0)
                        label_reverse    = np.delete(label_reverse, range_del, axis=0)
                        syllable_reverse = np.delete(syllable_reverse, range_del, axis=0)
                    data_reverse[bound_start:bound_del,:] = 0
                    # label_reverse[bound_start:bound_del,:] = -1 # mask后剩余帧级参与训练
                    label_reverse[:,:] = -1 # 帧级不参与训练
                    syllable_reverse[bound_start:bound_del,:] = -1
                    end2end_reverse = np.zeros(np.shape(label_reverse), dtype=np.int16)
                    data_out     += [data_reverse.reshape(-1)]
                    label_out    += [label_reverse]
                    syllable_out += [syllable_reverse]
                    vocab_out    += [[-2]]
                    end2end_out  += [end2end_reverse]
        # 命令词回放数据
        else:
            data_cmd, label_cmd, syllable_cmd, vocab_cmd, end2end_cmd = self.__readLmdb__(self.lmdb_pb, self.data_keys, [index])
            datapb_out     += data_cmd
            labelpb_out    += label_cmd
            syllablepb_out += syllable_cmd
            vocabpb_out    += vocab_cmd
            end2endpb_out  += end2end_cmd

        # 通用仿真数据
        if self.lmdbuni_key != None:
            if (self.pointer_uni + self.train_ratio_uni) > (self.pointer_uni_ed - self.pointer_uni_bg):
                self.pointer_uni = self.rng.integers(0, self.pointer_uni_ed - self.pointer_uni_bg - 1 - self.train_ratio_uni) # 随机起始点顺序读取
            sample_uni = list(range(self.pointer_uni, self.pointer_uni + self.train_ratio_uni)) 
            data_uni, label_uni, syllable_uni, vocab_uni, end2end_uni = self.__readLmdb__(self.lmdb_uni, self.datauni_keys, sample_uni)
            self.pointer_uni += self.train_ratio_uni
            data_out     += data_uni
            label_out    += label_uni
            syllable_out += syllable_uni
            vocab_out    += vocab_uni
            end2end_out  += end2end_uni

        # 通用回放数据
        if self.lmdbunipb_key != None:
            if (self.pointer_unipb + self.train_ratio_unipb) > (self.pointer_unipb_ed - self.pointer_unipb_bg ):
                self.pointer_unipb = self.rng.integers(0, self.pointer_unipb_ed - self.pointer_unipb_bg - 1 - self.train_ratio_unipb) # 随机起始点顺序读取
            sample_unipb = list(range(self.pointer_unipb, self.pointer_unipb + self.train_ratio_unipb)) 
            data_unipb, label_unipb, syllable_unipb, vocab_unipb, end2end_unipb = self.__readLmdb__(self.lmdb_unipb, self.dataunipb_keys, sample_unipb)
            self.pointer_unipb += self.train_ratio_unipb
            datapb_out     += data_unipb
            labelpb_out    += label_unipb
            syllablepb_out += syllable_unipb
            vocabpb_out    += vocab_unipb
            end2endpb_out  += end2end_unipb

        # 反例数据(命令词:反例数据 = 3:self.train_ratio_none)
        if self.lmdbnone_key != None and index%3==0: 
            if (self.pointer_none + self.train_ratio_none) > (self.pointer_none_ed - self.pointer_none_bg ):
                self.pointer_none = self.rng.integers(0, self.pointer_none_ed - self.pointer_none_bg - 1 - self.train_ratio_none) # 随机起始点顺序读取
            sample_none = list(range(self.pointer_none, self.pointer_none + self.train_ratio_none)) 
            data_none, label_none, syllable_none, vocab_none, end2end_none = self.__readLmdb__(self.lmdb_none, self.datanone_keys, sample_none)
            self.pointer_none += self.train_ratio_none
            datanone_out += data_none
            labelnone_out += label_none
            syllablenone_out += syllable_none
            vocabnone_out += vocab_none
            end2endnone_out += end2end_none

        # 噪声数据采样
        sample_types = self.rng.choice(self.typesList_noise, 1)
        train_ratio_noise = 1
        for noise_index in sample_types:
            lmdbname_noise = getattr(self, "lmdb_noise_"+str(noise_index))
            pointer_noise = getattr(self, "pointer_noise_"+str(noise_index))
            pointer_noise_bg = getattr(self, "pointer_noise_bg_"+str(noise_index))
            pointer_noise_ed = getattr(self, "pointer_noise_ed_"+str(noise_index))

            datanoise_keys = self.datanoise_keysDict[noise_index]
            if (pointer_noise + train_ratio_noise) > (pointer_noise_ed - pointer_noise_bg):
                pointer_noise = self.rng.integers(0, pointer_noise_ed - pointer_noise_bg - 1 - train_ratio_noise) # 随机起始点顺序读取
            sample_noise = list(range(pointer_noise, pointer_noise + train_ratio_noise))

            tmpnoise_out, _, _, _, _ = self.__readLmdb__(lmdbname_noise, datanoise_keys, sample_noise, max_len=1000)
            pointer_noise += train_ratio_noise
            setattr(self, "pointer_noise_"+str(noise_index), pointer_noise)
            datanoise_out += tmpnoise_out
            
        assert(len(data_out)==len(label_out)==len(syllable_out)==len(end2end_out))
        assert(len(datapb_out)==len(labelpb_out)==len(syllablepb_out)==len(end2endpb_out))

        # 额外仿真数据
        if self.lmdbextra_key != None:
            if (self.pointer_extra + self.train_ratio_extra) > (self.pointer_extra_ed - self.pointer_extra_bg):
                self.pointer_extra = self.rng.integers(0, self.pointer_extra_ed - self.pointer_extra_bg - 1 - self.train_ratio_extra) # 随机起始点顺序读取
            sample_extra = list(range(self.pointer_extra, self.pointer_extra + self.train_ratio_extra)) 
            data_extra, label_extra, syllable_extra, vocab_extra, end2end_extra = self.__readLmdb__(self.lmdb_extra, self.dataextra_keys, sample_extra)
            self.pointer_extra += self.train_ratio_extra
            data_out     += data_extra
            label_out    += label_extra
            syllable_out += syllable_extra
            vocab_out    += vocab_extra
            end2end_out  += end2end_extra

        # 额外回放数据
        if self.lmdbextrapb_key != None:
            if (self.pointer_extrapb + self.train_ratio_extrapb) > (self.pointer_extrapb_ed - self.pointer_extrapb_bg ):
                self.pointer_extrapb = self.rng.integers(0, self.pointer_extrapb_ed - self.pointer_extrapb_bg - 1 - self.train_ratio_extrapb) # 随机起始点顺序读取
            sample_extrapb = list(range(self.pointer_extrapb, self.pointer_extrapb + self.train_ratio_extrapb)) 
            data_extrapb, label_extrapb, syllable_extrapb, vocab_extrapb, end2end_extrapb = self.__readLmdb__(self.lmdb_extrapb, self.dataextrapb_keys, sample_extrapb)
            self.pointer_extrapb += self.train_ratio_extrapb
            datapb_out     += data_extrapb
            labelpb_out    += label_extrapb
            syllablepb_out += syllable_extrapb
            vocabpb_out    += vocab_extrapb
            end2endpb_out  += end2end_extrapb

        # 多通道数据
        datainter_out      = []
        datainterpb_out    = []
        datarir_out        = []
        datamusic_out      = []
        datamcnoise_out    = []
        if self.is_mc:
            sample_len = math.ceil(len(data_out)/float(self.pad_sentnum))
                        
            # 通用干扰数据
            if self.lmdbinter_key != None:
                if (self.pointer_inter + self.train_ratio_inter) > (self.pointer_inter_ed - self.pointer_inter_bg):
                    self.pointer_inter = self.rng.integers(0, self.pointer_inter_ed - self.pointer_inter_bg - 1 - self.train_ratio_inter) # 随机起始点顺序读取
                sample_inter = list(range(self.pointer_inter, self.pointer_inter + self.train_ratio_inter)) 
                data_inter, _, _, _, _ = self.__readLmdb__(self.lmdb_inter, self.datainter_keys, sample_inter)
                self.pointer_uni  += self.train_ratio_uni
                datainter_out     += data_inter

            # 回放点噪声源采样
            train_ratio_interpb = sample_len
            if self.lmdbinterpb_key != None:
                if (self.pointer_interpb + train_ratio_interpb) > (self.pointer_interpb_ed - self.pointer_interpb_bg ):
                    self.pointer_interpb = self.rng.integers(0, self.pointer_interpb_ed - self.pointer_interpb_bg - 1 - train_ratio_interpb) # 随机起始点顺序读取
                sample_interpb = list(range(self.pointer_interpb, self.pointer_interpb + train_ratio_interpb)) 
                tmpinterpb_out = self.__readLmdbMC__(self.lmdb_interpb, self.datainterpb_keys, sample_interpb, 2, -1)
                self.pointer_interpb += train_ratio_interpb
                datainterpb_out += tmpinterpb_out

            # rir数据采样
            train_ratio_rir = sample_len
            if self.lmdbrir_key != None:
                if (self.pointer_rir + train_ratio_rir) > (self.pointer_rir_ed - self.pointer_rir_bg ):
                    self.pointer_rir = self.rng.integers(0, self.pointer_rir_ed - self.pointer_rir_bg - 1 - train_ratio_rir) # 随机起始点顺序读取
                sample_rir = list(range(self.pointer_rir, self.pointer_rir + train_ratio_rir)) 
                tmpch_rir = self.__readLmdbMC__(self.lmdb_rir, self.datarir_keys, sample_rir, 8, -1)
                self.pointer_rir += train_ratio_rir
                datarir_out += tmpch_rir
            
            # music数据采样
            train_ratio_music = sample_len
            if self.lmdbmusic_key != None:
                if (self.pointer_music + train_ratio_music) > (self.pointer_music_ed - self.pointer_music_bg ):
                    self.pointer_music = self.rng.integers(0, self.pointer_music_ed - self.pointer_music_bg - 1 - train_ratio_music) # 随机起始点顺序读取
                sample_music = list(range(self.pointer_music, self.pointer_music + train_ratio_music)) 
                tmpmusic_out = self.__readLmdbMC__(self.lmdb_music, self.datamusic_keys, sample_music, 1, -1)
                self.pointer_music += train_ratio_music
                datamusic_out += tmpmusic_out

            # mcnoise数据采样
            train_ratio_mcnoise = sample_len
            if self.lmdbmcnoise_key != None:
                if (self.pointer_mcnoise + train_ratio_mcnoise) > (self.pointer_mcnoise_ed - self.pointer_mcnoise_bg ):
                    self.pointer_mcnoise = self.rng.integers(0, self.pointer_mcnoise_ed - self.pointer_mcnoise_bg - 1 - train_ratio_mcnoise) # 随机起始点顺序读取
                sample_mcnoise = list(range(self.pointer_mcnoise, self.pointer_mcnoise + train_ratio_mcnoise)) 
                tmpmcnoise_out = self.__readLmdbMC__(self.lmdb_mcnoise, self.datamcnoise_keys, sample_mcnoise, 2, -1)
                self.pointer_mcnoise += train_ratio_mcnoise
                datamcnoise_out += tmpmcnoise_out

        return data_out, label_out, syllable_out, vocab_out, end2end_out, datapb_out, labelpb_out, syllablepb_out, vocabpb_out, end2endpb_out, datanoise_out, datanone_out, labelnone_out, syllablenone_out, vocabnone_out, end2endnone_out, datainter_out, datainterpb_out, datarir_out, datamusic_out, datamcnoise_out

class BunchSampler(Sampler):
    def __init__(self, dataset_lengths: Sequence[int], batch_size: int, bunch_size: int, drop_last: bool, shuffle_batch: bool = True, iter_num=None, seed=None, is_sorted=False) -> None:
        self.lengths = dataset_lengths
        self.batch_size = batch_size
        self.bunch_size = bunch_size
        self.drop_last = drop_last
        self.shuffle_batch = shuffle_batch
        self.is_sorted = is_sorted
        
        self.generator = torch.Generator()
        self.rng = np.random.default_rng()
        if seed is not None:
            self.generator.manual_seed(seed)
            self.rng = np.random.default_rng(seed)

        if self.is_sorted:
            # 排序
            dict_length_indices = dict()
            for i, length in enumerate(self.lengths):
                if length in dict_length_indices:
                    dict_length_indices[length].append(i)
                else:
                    dict_length_indices[length] = [i]
            self.length_indices_list = sorted(dict_length_indices.items(), key=lambda x:x[0])
        else:
            # 不排序
            self.length_indices_list = []
            for i, length in enumerate(self.lengths):
                self.length_indices_list.append(tuple((length, i)))
            if self.shuffle_batch:
                self.rng.shuffle(self.length_indices_list)
        
        if iter_num is None:
            batch_sequence = self._prefetch_batch(static_batch_num=True)
            self.iter_num = len(batch_sequence)
        else:
            self.iter_num = iter_num

    def __iter__(self):
        batch_sequence = self._prefetch_batch()
        if self.shuffle_batch:
            indices = torch.randperm(len(batch_sequence), generator=self.generator).tolist()
        else:
            indices = list(range(len(batch_sequence)))
        if len(indices) > self.iter_num:
            indices = indices[:self.iter_num]
        while len(indices) < self.iter_num:
            indices = indices + indices[:self.iter_num-len(indices)]
        for index in indices:
            yield batch_sequence[index]
    
    def __len__(self):
        return self.iter_num

    def _prefetch_batch(self, static_batch_num=False):
        batch_sequence = []
        batch = []
        total_length = 0

        if self.is_sorted:
            # 排序
            for length, indices in self.length_indices_list:
                if length > self.bunch_size:
                    continue
                if self.shuffle_batch and not static_batch_num:
                    self.rng.shuffle(indices)
                for index in indices:
                    batch.append(index)
                    total_length += length
                    if total_length + length > self.bunch_size or len(batch) == self.batch_size:
                        batch_sequence.append(batch)
                        batch = []
                        total_length = 0
        else:
            # 不排序
            for length, index in self.length_indices_list:
                if length > self.bunch_size:
                    continue
                batch.append(index)
                total_length += length
                if total_length + length > self.bunch_size or len(batch) == self.batch_size:
                    batch_sequence.append(batch)
                    batch = []
                    total_length = 0

        if len(batch) > 0 and not self.drop_last:
            batch_sequence.append(batch)
        return batch_sequence

class CollaterTrain():
    def __init__(self, nmod_pad, train_e2e_winsize, lmdb_reverbfile, lmdb_cmdlist, lmdb_normfile, train_padhead_frame, train_padtail_frame, seed=None, is_pad=False, pad_sentnum=3, is_reverb=True, reverb_coef=None, is_addnoise=True, addnoise_coef=None, is_addnoisepb=True, addnoisepb_coef=None, is_ivw=False, is_mc=False):
        # 随机种子
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        
        # 初始化
        self.nmod_pad            = nmod_pad
        self.lmdb_reverbfile     = lmdb_reverbfile
        self.lmdb_normfile       = lmdb_normfile
        self.train_e2e_winsize   = train_e2e_winsize
        self.train_padhead_frame = train_padhead_frame
        self.train_padtail_frame = train_padtail_frame

        self.is_pad          = is_pad
        self.pad_sentnum     = pad_sentnum
        self.is_reverb       = is_reverb
        self.reverb_coef     = reverb_coef
        self.is_addnoise     = is_addnoise
        self.addnoise_coef   = addnoise_coef
        self.is_addnoisepb   = is_addnoisepb
        self.addnoisepb_coef = addnoisepb_coef
        self.is_ivw          = is_ivw
        if self.is_ivw:
            self.noneid0  = noneid_ivw0
            self.noneid1  = noneid_ivw1
        else:
            self.noneid0  = noneid_csk0
            self.noneid1  = noneid_csk1
        self.is_mc       = is_mc

        # 获取fea norm
        with open(self.lmdb_normfile, 'r') as nf:
            normfile_lines = nf.read().splitlines()
        self.mel_spec_mean = np.array([float(item) for item in normfile_lines[1:41]])
        self.mel_spec_std = np.array([float(item) for item in normfile_lines[42:82]]) 
        
        # 单通道仿真参数配置
        self.reverb_data = list(np.load(self.lmdb_reverbfile, allow_pickle=True))
        self.noise_snr = [3, 3, 5, 5, 5] + list(range(6, 21))
        self.rng.shuffle(self.noise_snr)
        self.noise_snrpb = list(range(10, 30))
        self.rng.shuffle(self.noise_snrpb)
        self.amp_list = list(range(1000, 20000, 200))
        self.rng.shuffle(self.amp_list)
        self.count= 0

        # 多通道仿真参数配置
        self.mcnoise_snr = [8, 11, 13, 14, 17, 20]
        self.mcddr_snr   = [-10, 0, 10, 20, 50, 80]
        self.mcinter_snr = [0, 5, 6, 7, 10, 12, 15]
        self.rng.shuffle(self.mcnoise_snr)
        self.rng.shuffle(self.mcddr_snr)
        self.rng.shuffle(self.mcinter_snr)

        #端到端分支词mask
        self.cmd_len_mask = func_calCmdBoundary(lmdb_cmdlist, self.train_e2e_winsize)
    
    def __padData__(self, data_in, label_in, syllable_in, vocab_in, end2end_in, pad_count):
        assert(len(data_in)==len(label_in)==len(syllable_in)==len(vocab_in)==len(end2end_in))
        data_out      = []
        label_out     = []
        syllable_out  = []
        vocab_out     = []
        end2end_out   = []
        sequence_len  = []
        sent1st_len   = []
        padsil_label0 = (self.noneid0-1)
        padsil_label1 = (self.noneid1-1)
        padsil_min    = 8
        max_discount  = 1.5
        
        # 数据乱序
        zip_list = list( zip(data_in, label_in, syllable_in, vocab_in, end2end_in) )
        self.rng.shuffle(zip_list)
        data_in, label_in, syllable_in, vocab_in, end2end_in = zip(*zip_list) # 多子句shuffle顺序

        total_len = len(data_in)
        pointer_bg = 0
        pointer_ed = pad_count # 每n句话拼一句
        while pointer_bg < total_len:
            tmp_data     = data_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_label    = label_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_syllable = syllable_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_vocab    = vocab_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_end2end  = end2end_in[pointer_bg : min(pointer_ed, total_len)]

            # 更新边界 && 初始化容器
            pointer_bg = pointer_ed
            pointer_ed += 3
            tmp_len    = []
            vocab_cat  = []

            # 数据拼接
            if len(tmp_data) > 1:
                max_value = 0.0
                tmp_data_out     = []
                tmp_label_out    = []
                tmp_syllable_out = []
                tmp_vocab_out = []
                tmp_end2end_out  = []
                wave_len = 0
                frames_len = 0

                # 音频归一和过滤
                for index, data_array in enumerate(tmp_data):
                    data_array = np.array( data_array, dtype=np.float32 )
                    data_sort = np.partition(data_array, -5)
                    amp_max = data_sort[-5]
                    if amp_max <= 100:
                        continue
                    if amp_max > max_value:
                        max_value = amp_max
                    wave_len += np.shape(data_array)[0]
                    frames_len += np.shape(tmp_label[index])[0]
                    tmp_data_out.append( data_array / float(amp_max) )
                    tmp_label_out.append( tmp_label[index] )
                    tmp_syllable_out.append( tmp_syllable[index] )
                    tmp_vocab_out.append( tmp_vocab[index] )
                    tmp_end2end_out.append( tmp_end2end[index] )

                    # 多通道仿真记录第一句话的长度
                    if len(tmp_data_out) == 1:
                        sent1st_len.append(self.train_padhead_frame*160 + wave_len)

                # 采样sil长度
                assert(self.train_padhead_frame >= padsil_min)
                wave_count = len(tmp_data_out)
                pad_mid = list(self.rng.choice(list(range(padsil_min, self.train_padhead_frame)), wave_count, replace=False))
                pad_mid.insert(0, self.train_padhead_frame)
                pad_mid[-1] = self.train_padtail_frame
                for pad_len in pad_mid:
                    wave_len += pad_len*160
                    frames_len += pad_len
                data_indexs = list(range(wave_count))
                self.rng.shuffle(data_indexs)
                
                # 初始化音频和标注
                data_cat = np.zeros( (wave_len), dtype=np.int16)
                label_cat = np.ones( (frames_len, 2), dtype=np.int16 )*(-1) #句子间的pad为-1
                syllable_cat = np.ones( (frames_len, 1), dtype=np.int16 )*(-1) #句子间的pad为-1
                end2end_cat = np.ones( (frames_len, 1), dtype=np.int16 )*(-1) #句子间的pad为-1

                wave_bg = 0
                frames_bg = 0
                max_value = max_value/max_discount #幅值规整
                for i, index in enumerate(data_indexs):
                    
                    # padding部分只学习1/3的sil标签
                    if pad_mid[i] > 16:
                        if i == 0:
                            label_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, 0] = padsil_label0
                            label_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, 1] = padsil_label1
                            syllable_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :] = 0
                            end2end_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :] = 0
                        else:
                            label_cat[ math.ceil( frames_bg+pad_mid[i]/3.0*2 ): frames_bg+pad_mid[i], 0] = padsil_label0
                            label_cat[ math.ceil( frames_bg+pad_mid[i]/3.0*2 ): frames_bg+pad_mid[i], 1] = padsil_label1
                            syllable_cat[ math.ceil( frames_bg+pad_mid[i]/3.0*2 ): frames_bg+pad_mid[i], :] = 0
                            end2end_cat[ math.ceil( frames_bg+pad_mid[i]/3.0*2 ): frames_bg+pad_mid[i], :] = 0
                    else:
                        label_cat[ frames_bg: frames_bg+pad_mid[i], 0] = padsil_label0
                        label_cat[ frames_bg: frames_bg+pad_mid[i], 1] = padsil_label1
                        syllable_cat[ frames_bg: frames_bg+pad_mid[i], :] = 0
                        end2end_cat[ frames_bg: frames_bg+pad_mid[i], :] = 0
                    
                    wave_bg += pad_mid[i]*160
                    frames_bg += pad_mid[i]

                    # 数据拼接
                    temp_waveLen = np.shape( tmp_data_out[index] )[0]
                    data_rescale = np.array( tmp_data_out[index] * max_value, dtype=np.int16 )
                    data_cat[wave_bg : wave_bg+temp_waveLen] = data_rescale

                    temp_framesLen = np.shape( tmp_label_out[index] )[0]
                    label_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_label_out[index]
                    syllable_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_syllable_out[index]
                    end2end_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_end2end_out[index]
                    
                    # 记录数据边界和vocab标注合并
                    tmp_len.append(np.array( [frames_bg, frames_bg+temp_framesLen] ))
                    vocab_cat.append(tmp_vocab_out[index])

                    wave_bg += temp_waveLen
                    frames_bg += temp_framesLen

                # 尾部数据拼接
                assert(frames_bg+pad_mid[-1]==frames_len)
                label_cat[ frames_bg: frames_len, 0] = padsil_label0
                label_cat[ frames_bg: frames_len, 1] = padsil_label1
                syllable_cat[ frames_bg: frames_len, :] = 0
                end2end_cat[ frames_bg: frames_len, :] = 0

            else:
                # 拼接sil采样
                assert(self.train_padhead_frame >= 8)
                pad_mid = []
                pad_mid.append(self.train_padhead_frame)
                pad_mid.append(self.train_padtail_frame)

                # 多通道仿真记录第一句话的长度
                sent1st_len.append(pad_mid[0]*160 + np.shape( tmp_data[0] )[0])

                # 初始化音频和标注
                wave_len = pad_mid[0]*160 + np.shape( tmp_data[0] )[0] + self.train_padtail_frame*160
                frames_len = pad_mid[0] + np.shape( tmp_label[0] )[0] + self.train_padtail_frame
                data_cat = np.zeros( (wave_len), dtype=np.int16)
                label_cat = np.ones( (frames_len, 2), dtype=np.int16 )*(-1) #句子间的pad为-1
                syllable_cat = np.ones( (frames_len, 1), dtype=np.int16 )*(-1) #句子间的pad为-1
                end2end_cat = np.ones( (frames_len, 1), dtype=np.int16 )*(-1) #句子间的pad为-1
                
                wave_bg = 0
                frames_bg = 0
                if pad_mid[0] > 16:
                    label_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, 0] = padsil_label0
                    label_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, 1] = padsil_label1
                    syllable_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :] = 0
                    end2end_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :] = 0
                else:
                    label_cat[ frames_bg: frames_bg+pad_mid[0], 0] = padsil_label0
                    label_cat[ frames_bg: frames_bg+pad_mid[0], 1] = padsil_label1
                    syllable_cat[ frames_bg: frames_bg+pad_mid[0], :] = 0
                    end2end_cat[ frames_bg: frames_bg+pad_mid[0], :] = 0

                wave_bg += pad_mid[0]*160
                frames_bg += pad_mid[0]

                # 数据拼接
                temp_waveLen = np.shape( tmp_data[0] )[0]
                data_cat[wave_bg : wave_bg+temp_waveLen] = tmp_data[0]

                temp_framesLen = np.shape( tmp_label[0] )[0]
                label_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_label[0]
                syllable_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_syllable[0]
                end2end_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_end2end[0]

                # 记录数据边界和vocab标注合并
                tmp_len.append(np.array( [frames_bg, frames_bg+temp_framesLen] ))
                vocab_cat.append(tmp_vocab[0])

                wave_bg += temp_waveLen
                frames_bg += temp_framesLen

                # 尾部数据拼接
                assert(frames_bg+pad_mid[-1]==frames_len)
                label_cat[ frames_bg: frames_len, 0] = padsil_label0
                label_cat[ frames_bg: frames_len, 1] = padsil_label1
                syllable_cat[ frames_bg: frames_len, :] = 0
                end2end_cat[ frames_bg: frames_len, :] = 0

            assert(len(label_cat)==len(syllable_cat)==len(end2end_cat))
            data_out.append(data_cat)
            label_out.append(label_cat)
            syllable_out.append(syllable_cat)
            vocab_out.append(vocab_cat)
            end2end_out.append(end2end_cat)
            sequence_len.append(np.array(tmp_len))
        
        return data_out, label_out, syllable_out, vocab_out, end2end_out, sequence_len, sent1st_len

    def __SimDataMultiCh__(self, data_list, inter_list, interpb_list, rir_list, music_list, noise_list, sent1st_len, is_pad, label_list, syllable_list, vocab_list, end2end_list, sequence_len):
        # 数据仿真
        D_ch1_rir=[]   
        D_ch2_rir=[]   
        s_ch1_rir=[]
        s_ch2_rir=[]
        i1_ch1_rir=[]
        i1_ch2_rir=[]
        s_dual_mic=[]
        i_dual_mic=[]
        clean_dual_mic=[]
        single_noise_mic=[]
        for key, value in enumerate(rir_list):
            value = value.transpose(1, 0)
            s_ch1_rir.append(value[:, 0])
            s_ch2_rir.append(value[:, 4])
            i1_ch1_rir.append(value[:, 1])
            i1_ch2_rir.append(value[:, 5])

            tmp_rir1 = value[:, 0].copy()
            tmp_rir1[900: ] = 0       # 直达声
            tmp_rir2 = value[:, 4].copy()
            tmp_rir2[900: ] = 0       # 直达声
            D_ch1_rir.append(tmp_rir1)
            D_ch2_rir.append(tmp_rir2)
            single_noise_mic.append(np.squeeze(noise_list[key][0,:]))

        # 加点噪声源
        if self.rng.random(1)[0] > 0.5:
            DDR_index  = self.rng.integers(0, len(self.mcddr_snr), size=len(single_noise_mic))
            DDR        = np.array(self.mcddr_snr)[DDR_index]
            inter_list = delta.addnoise(inter_list, single_noise_mic, DDR.tolist(), num_thread=4)
        else:
            inter_list = music_list

        clean_ch1_list = delta.conv_reverb(data_list,  D_ch1_rir, num_thread=4)  
        clean_ch2_list = delta.conv_reverb(data_list,  D_ch2_rir, num_thread=4)  
        data_ch1_list  =  delta.conv_reverb(data_list,  s_ch1_rir, num_thread=4)
        data_ch2_list  =  delta.conv_reverb(data_list,  s_ch2_rir, num_thread=4)
        inter_ch1_list = delta.conv_reverb(inter_list, i1_ch1_rir, num_thread=4)
        inter_ch2_list = delta.conv_reverb(inter_list, i1_ch2_rir, num_thread=4)

        for key, value in enumerate(rir_list):
            tmp_len1st = sent1st_len[key]
            s_dual     = np.concatenate( (np.expand_dims(data_ch1_list[key], 0), np.expand_dims(data_ch2_list[key], 0)), 0)
            i_dual     = np.concatenate( (np.expand_dims(inter_ch1_list[key], 0), np.expand_dims(inter_ch2_list[key], 0)), 0)
            c_dual     = np.concatenate( (np.expand_dims(clean_ch1_list[key], 0), np.expand_dims(clean_ch2_list[key], 0)), 0)
            
            # 复制第一句话,用于波束收敛
            s_dual     = np.concatenate((s_dual[ : , :tmp_len1st ], s_dual), 1)
            i_dual     = np.concatenate((i_dual[ : , :tmp_len1st ], i_dual), 1)
            c_dual     = np.concatenate((c_dual[ : , :tmp_len1st ], c_dual), 1)
            s_dual_mic.append(s_dual)
            i_dual_mic.append(i_dual)
            clean_dual_mic.append(c_dual)

        SIR_index = self.rng.integers(0, len(self.mcinter_snr), size=len(s_dual_mic))
        SIR       = np.array(self.mcinter_snr)[SIR_index]
        SNR_index = self.rng.integers(0, len(self.mcnoise_snr), size=len(s_dual_mic))
        SNR       = np.array(self.mcnoise_snr)[SNR_index]

        if self.rng.random(1)[0] > 0.5:
            noise_addList = delta.addnoise_mc(s_dual_mic, i_dual_mic, SIR.tolist(), num_thread=4) ## 加仿真干扰
            sim_flag = True
        else:
            noise_addList = delta.addnoise_mc(s_dual_mic, interpb_list, SIR.tolist(), num_thread=4)  ## 加回放干扰
            sim_flag = False

        noise_addList = delta.addnoise_mc(noise_addList, noise_list, SNR.tolist(), num_thread=4)
        FFT_len=1024
        real, imag = delta.stft_r2c(noise_addList, 512, num_thread=4)
        real_512c,  imag_512c = delta.stft_r2c(clean_dual_mic, 512, num_thread=4)
        real_1024,  imag_1024 = delta.stft_r2c(noise_addList, FFT_len, num_thread=4)
        real_1024c, imag_1024c = delta.stft_r2c(clean_dual_mic, FFT_len, num_thread=4)

        complex_mic_512=[]
        complex_mic_512c=[]
        complex_mic_1024=[]
        mask_mic=[]
        wk_info_frame_list=[]
        for i in range(len(real)):
            tmp = real[i] + 1j*imag[i]
            tmpc = real_512c[i] + 1j*imag_512c[i]
            tmp_1024 = real_1024[i] + 1j*imag_1024[i]            
            noise_power = (real_1024[i][0,:,:]-real_1024c[i][0,:,:])**2 + (imag_1024[i][0,:,:]-imag_1024c[i][0,:,:])**2
            clean_power = (real_1024c[i][0,:,:])**2 + (imag_1024c[i][0,:,:])**2
            mask_tmp = (clean_power/(clean_power + noise_power +1e-10))
            
            # 唤醒信息
            idx_tmp=np.where(np.mean(clean_power,1)>1000)[0]
            idx=np.zeros(clean_power.shape[0])
            idx[idx_tmp]=1
            idx1=idx[1:]
            idx2=idx[0:-1]-idx1
            idx3=np.where(idx2==-1)[0]
            idx4=np.where(idx2==1)[0]
            try:
                wk_info_frame=np.zeros((idx3.shape[0],2))
                for i in range(idx3.shape[0]):
                    wk_info_frame[i,0]=idx3[i]
                    wk_info_frame[i,1]=idx4[i]
            except:
                wk_info_frame = np.zeros((1,2))
                tmp_info = np.array([-10000,-10000]) 
                wk_info_frame[0,:] = tmp_info

            complex_mic_512.append(tmp)
            complex_mic_512c.append(tmpc)
            complex_mic_1024.append(tmp_1024)
            mask_mic.append(mask_tmp)
            wk_info_frame_list.append(wk_info_frame.astype('int32'))

        cfg_ini = "./save_mask.ini"
        GSCout_r, GSCout_i, GSCout_c_r,GSCout_c_i = delta.do_mab(cfg_ini, complex_mic_512, complex_mic_512c,num_thread=4)

        tdmvdr_out_r, tdmvdr_out_i, _, _, tdmvdr_n_r, tdmvdr_n_i = delta.do_tdmvdr(cfg_ini, complex_mic_1024, mask_mic, wk_info_frame_list,num_thread=4)
        selected_beam_r=[]   
        selected_beam_i=[]  
        clean_secected_r=[]
        clean_secected_i=[]
        for i in range(len(complex_mic_512c)):
            tmp_noise = GSCout_r[i]  +  1j*GSCout_i[i] 
            tmp_clean = GSCout_c_r[i] + 1j*GSCout_c_i[i]
            SDR=abs(tmp_clean[:,157:,:])/(abs(tmp_noise[:,157:,:]-tmp_clean[:,157:,:])+1e-5)
            SDR_last1=10*math.log10(sum(sum(SDR[0,:,:]))+1e-5)
            SDR_last2=10*math.log10(sum(sum(SDR[1,:,:]))+1e-5)
            SDR_last3=10*math.log10(sum(sum(SDR[2,:,:]))+1e-5)
            SDR_list=[SDR_last1,SDR_last2,SDR_last3]
            max_index = SDR_list.index(max(SDR_list)) # 返回最大值的索引
            selected_beam_r.append(GSCout_r[i][max_index,:,:])
            selected_beam_i.append(GSCout_i[i][max_index,:,:])
            clean_secected_r.append(GSCout_c_r[i][max_index,:,:])
            clean_secected_i.append(GSCout_c_i[i][max_index,:,:])            

        if self.rng.random(1)[0] > 0.5:
            wavs_mvdr = delta.stft_c2r(selected_beam_r, selected_beam_i, 512,num_thread=4)
        else:
            wavs_mvdr = delta.stft_c2r(tdmvdr_out_r, tdmvdr_out_i, 1024,num_thread=4)

        wavs_gsc4 = delta.stft_c2r(clean_secected_r, clean_secected_i, 512,num_thread=4)               
        wavs_n = delta.stft_c2r(tdmvdr_n_r, tdmvdr_n_i, 1024,num_thread=4) 

        # 输出
        data_out     = []
        label_out    = []
        syllable_out = []
        vocab_out    = []
        end2end_out  = []
        sequence_out = []
        if is_pad:
            for index, value in enumerate(wavs_mvdr):
                data_tmp = np.array(value[0, sent1st_len[index]:], dtype=np.int16)
                data_out.append( data_tmp )
            label_out = label_list[:]
            syllable_out = syllable_list[:]
            vocab_out = vocab_list[:]
            end2end_out = end2end_list[:]
            sequence_out = sequence_len[:]
        else:       
            for index, value in enumerate(wavs_mvdr):
                data_tmp = np.array(value[0, sent1st_len[index]:], dtype=np.int16)
                label_tmp = label_list[index]
                syllable_tmp = syllable_list[index]
                vocab_tmp = vocab_list[index]
                end2end_tmp = end2end_list[index]
                sequence_tmp = sequence_len[index]
                for subindex, subvalue in enumerate(sequence_tmp):
                    bg_node, ed_node = subvalue
                    data_out.append( data_tmp[bg_node*160:ed_node*160] )
                    label_out.append( label_tmp[bg_node:ed_node] )
                    syllable_out.append( syllable_tmp[bg_node:ed_node] )
                    vocab_out.append( vocab_tmp[subindex] )
                    end2end_out.append( end2end_tmp[bg_node:ed_node] )
        return data_out, label_out, syllable_out, vocab_out, end2end_out, sequence_out

    def __ampChange__(self, data_in, label_in, syllable_in, vocab_in, end2end_in, sequenceLen, ampList):
        data_out = []
        label_out = []
        syllable_out = []
        vocab_out = []
        end2end_out = []
        sequence_out = []
        for i, amp in enumerate(ampList):
            data_array = np.array(data_in[i], dtype=np.float32)
            data_sort = np.partition(data_array, -5)
            amp_max = data_sort[-5]
            if amp_max <= 100:
                continue
            data_out.append(np.array(data_array/amp_max*amp, dtype=np.int16))
            label_out.append(label_in[i])
            syllable_out.append(syllable_in[i])
            vocab_out.append(vocab_in[i])
            end2end_out.append(end2end_in[i])
            if sequenceLen != []:
                sequence_out.append(sequenceLen[i]) 
        return data_out, label_out, syllable_out, vocab_out, end2end_out, sequence_out

    def __calFeanorm__(self, data_in, label_in, syllable_in, end2end_in):
        data_out = []
        label_out = []
        syllable_out = []
        end2end_out = []
        for i, data_line in enumerate(data_in):
            data_out.append( (data_line - self.mel_spec_mean[None,:]) * self.mel_spec_std)
            label_out.append(label_in[i][:np.shape(data_line)[0], :])
            syllable_out.append(syllable_in[i][:np.shape(data_line)[0], :])
            end2end_out.append(end2end_in[i][:np.shape(data_line)[0], :])
        return data_out, label_out, syllable_out, end2end_out
    
    def __pad_nmod_alongB__(self, sequence, nmod_pad, val):
        maxlen = 0
        pad_list = []
        mask_list = []
        for nparray in sequence:
            if nparray.shape[0] > maxlen:
                maxlen = nparray.shape[0]
        if nmod_pad is not None:
            maxlen = maxlen if maxlen % nmod_pad == 0 else maxlen + nmod_pad - maxlen % nmod_pad
        for nparray in sequence:
            padlen = maxlen - nparray.shape[0]
            nparray = np.pad(nparray, ((0, padlen), (0, 0)), mode="constant", constant_values=(val,))
            nparray = nparray.transpose(1, 0)
            torcharray = torch.from_numpy(nparray).float()
            torchmask = torch.ones(1, torcharray.size()[1])
            if padlen > 0:
                torchmask[:, -padlen:] = 0
            torcharray = torcharray.reshape(1, 1, torcharray.size()[0], torcharray.size()[1])
            pad_list.append(torcharray)
            mask_list.append(torchmask)
        batch_array = torch.cat(pad_list, dim=0)
        batch_mask = torch.cat(mask_list, dim=0)
        return batch_array, batch_mask

    def __pad_nmod_alongT__(self, sequence, nmod_pad, val, length_del=None, node_in=None):
        padlen = 0
        totallen = 0
        pad_list = []
        mask_list = []
        node_out = []
        for i, nparray in enumerate(sequence):
            
            # 首句视野扰动丢数据
            if i == 0 and length_del != None:
                nparray = nparray[length_del:, :]

            # 丢数据后端点前移
            if node_in != None:
                subnode_in = node_in[i]
                for sublen in subnode_in:
                    if i == 0 and length_del != None:
                        node_out.append( np.array([sublen[0]+totallen-length_del, sublen[1]+totallen-length_del]) )
                    else:
                        node_out.append( np.array([sublen[0]+totallen, sublen[1]+totallen]) )

            totallen += nparray.shape[0]

            nparray = nparray.transpose(1, 0)
            torcharray = torch.from_numpy(nparray).float()
            torchmask = torch.ones(1, torcharray.size()[1])
            torcharray = torcharray.reshape(1, 1, torcharray.size()[0], torcharray.size()[1])
            pad_list.append(torcharray)
            mask_list.append(torchmask)

        if nmod_pad is not None:
            padlen = totallen if totallen % nmod_pad == 0 else totallen + nmod_pad - totallen % nmod_pad
            if padlen > totallen:
                paddata = torch.ones(1, 1, pad_list[0].size()[2], padlen-totallen)*val
                padmask = torch.zeros(1, padlen-totallen)
                pad_list.append(paddata)
                mask_list.append(padmask)
        batch_array = torch.cat(pad_list, dim=3)
        batch_mask = torch.cat(mask_list, dim=1)
        node_out = torch.from_numpy(np.array(node_out))
        return batch_array, batch_mask, node_out
 
    def __gen_vocablable__(self, vocab_in, node_in, min_datalen, nmod_pad, nmod_node, is_addend=False, endnode=1):
        max_datalen = 0
        seq_datalen = []
        max_unitlen = 0
        seq_unitlen = []
        vocab_out   = []
        node_out    = []
        ignore_out  = []
        for node_idx, (bg_node, ed_node) in enumerate(node_in):
            bg_nmod = math.floor(bg_node/nmod_node)
            ed_nmod = math.floor(ed_node/nmod_node)

            # 音频最小长度
            tmp_datalen = ed_nmod - bg_nmod
            if tmp_datalen < min_datalen:
                bg_nmod = max(0, ed_nmod-min_datalen)
                tmp_datalen = ed_nmod - bg_nmod

            # 音频最大长度
            if tmp_datalen > max_datalen:
                max_datalen = tmp_datalen

            tmp_unit = vocab_in[node_idx]
            if -2 in np.squeeze(tmp_unit):
                ignore_out.append(False)
            else:
                ignore_out.append(True)
            # 添加尾节点
            if is_addend:
                tmp_unit = np.append(tmp_unit, np.array([[endnode]]), axis=0)
            tmp_unitlen = len(tmp_unit)
            
            # 文本最大长度
            if tmp_unitlen > max_unitlen:
                max_unitlen = tmp_unitlen
            
            # 长度和端点输出
            seq_datalen.append(tmp_datalen)
            seq_unitlen.append(tmp_unitlen)
            node_out.append([bg_nmod, ed_nmod])
            vocab_out.append(np.squeeze(tmp_unit))

        # 文本补齐和mask
        max_datalen = max_datalen if max_datalen % nmod_pad == 0 else max_datalen + nmod_pad - max_datalen % nmod_pad
        node_num = len(node_out)
        matrix_unit  = np.ones((node_num, max_unitlen)) * (-2)
        matrix_attnmask = np.zeros((node_num, max_datalen, max_datalen))
        matrix_datamask = np.zeros((node_num, max_datalen))
        for i in range(max_datalen):
            matrix_attnmask[:, i, max(0, i-16):min(max_datalen, i+16)] = 1
        for index, (bg_nmod, ed_nmod) in enumerate(node_out):

            tmp_unitlen = seq_unitlen[index]
            tmp_datalen = seq_datalen[index]
            matrix_unit[index, :tmp_unitlen] = vocab_out[index]
            matrix_attnmask [index, tmp_datalen:, :] = 0
            matrix_attnmask [index, :, tmp_datalen:] = 0
            matrix_datamask [index, :tmp_datalen] = 1

        matrix_unit  = torch.from_numpy( matrix_unit )                    # (seq_num, max_unitlen)
        seq_unitlen  = torch.from_numpy( np.array(seq_unitlen) )          # (seq_num)
        matrix_attnmask = torch.from_numpy( matrix_attnmask )             # (seq_num, max_datalen, max_datalen)
        matrix_datamask = torch.from_numpy( matrix_datamask )             # (seq_num, max_datalen)
        seq_datalen  = torch.from_numpy( np.array(seq_datalen) )
        node_out = torch.from_numpy( np.array(node_out) )
        return matrix_unit, vocab_out, seq_unitlen, matrix_attnmask, matrix_datamask, seq_datalen, max_unitlen, max_datalen, node_out, ignore_out

    def __gen_syllablelable__(self, syllable_in, node_in, min_datalen, nmod_pad, nmod_node):
        syllable_in = syllable_in[:, :, :, ::nmod_node].squeeze()

        max_datalen = 0
        seq_datalen = []
        max_unitlen = 0
        seq_unit = []
        seq_unitlen = []
        node_out = []
        for bg_node, ed_node in node_in:
            bg_nmod = math.floor(bg_node/nmod_node)
            ed_nmod = math.floor(ed_node/nmod_node)

            # 音频最小长度
            tmp_datalen = ed_nmod - bg_nmod
            if tmp_datalen < min_datalen:
                bg_nmod = max(0, ed_nmod-min_datalen)
                tmp_datalen = ed_nmod - bg_nmod

            # 音频最大长度
            if tmp_datalen > max_datalen:
                max_datalen = tmp_datalen

            # 低帧率坍塌后的label & 跳过syllable有缺失的句子
            tmp_label = syllable_in[bg_nmod:ed_nmod]
            if -3 in tmp_label:
                continue
            tmp_unit = []
            for key, value in groupby(tmp_label):
                if key <= 1332 and key > 0:    # triphone(0~3002) 3003 </s> 3004 blank
                    tmp_unit.append( key ) # ctc-syllable(1~1332) 0 blank 1328 Error       
            tmp_unitlen = len(tmp_unit)

            # 文本最大长度
            if tmp_unitlen > max_unitlen:
                max_unitlen = tmp_unitlen
            
            # 文本长度和端点输出
            seq_datalen.append(tmp_datalen)
            seq_unit.append(tmp_unit)
            seq_unitlen.append(tmp_unitlen)
            node_out.append([bg_nmod, ed_nmod])

        # 文本补齐和mask
        max_datalen = max_datalen if max_datalen % nmod_pad == 0 else max_datalen + nmod_pad - max_datalen % nmod_pad
        node_num = len(node_out)
        matrix_unit  = np.ones((node_num, max_unitlen)) * (-2)
        matrix_attnmask = np.zeros((node_num, max_datalen, max_datalen))
        matrix_datamask = np.zeros((node_num, max_datalen))
        for i in range(max_datalen):
            matrix_attnmask[:, i, max(0, i-16):min(max_datalen, i+16)] = 1
        seq_datalen_expand = []
        for index, (bg_nmod, ed_nmod) in enumerate(node_out):

            tmp_unitlen = seq_unitlen[index]
            tmp_datalen = seq_datalen[index]
            matrix_unit[index, :tmp_unitlen] = seq_unit[index]
            matrix_attnmask [index, tmp_datalen:, :] = 0
            matrix_attnmask [index, :, tmp_datalen:] = 0
            matrix_datamask [index, :tmp_datalen] = 1

        matrix_unit  = torch.from_numpy( matrix_unit )                    # (seq_num, max_unitlen)
        seq_unitlen  = torch.from_numpy( np.array(seq_unitlen) )          # (seq_num)
        matrix_attnmask = torch.from_numpy( matrix_attnmask )             # (seq_num, max_datalen, max_datalen)
        matrix_datamask = torch.from_numpy( matrix_datamask )             # (seq_num, max_datalen)
        seq_datalen  = torch.from_numpy( np.array(seq_datalen) )
        node_out = torch.from_numpy( np.array(node_out) )
        return matrix_unit, seq_unit, seq_unitlen, matrix_attnmask, matrix_datamask, seq_datalen, max_unitlen, max_datalen, node_out

    def __call__(self, batch):
        # 仿真数据
        data_list       = []
        label_list      = []
        syllable_list   = []
        vocab_list      = []
        end2end_list    = []
        sequence_len    = []
        # 回放数据
        datapb_list     = []
        labelpb_list    = []
        syllablepb_list = []
        vocabpb_list    = []
        end2endpb_list  = []
        sequencepb_len  = []
        # 噪声数据
        datanosie_list  = []
        # 反例数据
        datanone_list     = []
        labelnone_list    = []
        syllablenone_list = []
        vocabnone_list    = []
        end2endnone_list  = []
        #多通道数据
        datamc_list     = []
        labelmc_list    = []
        syllablemc_list = []
        vocabmc_list    = []
        end2endmc_list  = []
        sequencemc_len  = []
        inter_list      = []
        interpb_list    = []
        music_list      = []
        mcnoise_list    = []
        rir_list        = []
        sent1stmc_len   = []

        # 多进程数据合并
        for data, label, syllable, vocab, end2end, datapb, labelpb, syllablepb, vocabpb, end2endpb, datanoise, datanone, labelnone, syllablenone, vocabnone, end2endnone_out, datainter, datainterpb, datarir, datamusic, datamcnoise in batch:
            # 单通道
            data_list.extend(data)
            label_list.extend(label)
            syllable_list.extend(syllable)
            vocab_list.extend(vocab)
            end2end_list.extend(end2end)
            datanosie_list.extend(datanoise)
            if len(datapb) != 0 :
                datapb_list.extend(datapb)
                labelpb_list.extend(labelpb)
                syllablepb_list.extend(syllablepb)
                vocabpb_list.extend(vocabpb)
                end2endpb_list.extend(end2endpb)
            if len(datanone) != 0:
                datanone_list.extend(datanone)
                labelnone_list.extend(labelnone)
                syllablenone_list.extend(syllablenone)
                vocabnone_list.extend(vocabnone)
                end2endnone_list.extend(end2endnone_out)
            # 多通道
            if self.is_mc and len(datainter) != 0:
                inter_list.extend(datainter)
                interpb_list.extend(datainterpb)
                rir_list.extend(datarir)
                music_list.extend(datamusic)
                mcnoise_list.extend(datamcnoise)

        # 多通道仿真
        sim_len = len(data_list)
        mc_ratio = 0.5
        if self.is_mc and sim_len>1:
            half_len = int(sim_len*mc_ratio)
            datamc_list = data_list[:half_len]
            labelmc_list = label_list[:half_len]
            syllablemc_list = syllable_list[:half_len]
            vocabmc_list = vocab_list[:half_len]
            end2endmc_list = end2end_list[:half_len]

            data_list = data_list[half_len:]
            label_list = label_list[half_len:]
            syllable_list = syllable_list[half_len:]
            vocab_list = vocab_list[half_len:]
            end2end_list = end2end_list[half_len:]
                        
            datamc_list, labelmc_list, syllablemc_list, vocabmc_list, end2endmc_list, sequencemc_len, sent1stmc_len = self.__padData__(datamc_list, labelmc_list, syllablemc_list, vocabmc_list, end2endmc_list, self.pad_sentnum)
            count_sent   = len(sent1stmc_len)
            inter_list   = inter_list[:count_sent]
            interpb_list = interpb_list[:count_sent]
            rir_list     = rir_list[:count_sent]
            music_list   = music_list[:count_sent]
            noise_list   = mcnoise_list[:count_sent]
            datamc_list, labelmc_list, syllablemc_list, vocabmc_list, end2endmc_list, sequencemc_len = self.__SimDataMultiCh__(datamc_list, inter_list, interpb_list, rir_list, music_list, noise_list, sent1stmc_len, self.is_pad, labelmc_list, syllablemc_list, vocabmc_list, end2endmc_list, sequencemc_len)

        # 反例/回放数据合并
        if len(datanone_list) != 0:
            datapb_list     += datanone_list
            labelpb_list    += labelnone_list
            syllablepb_list += syllablenone_list
            vocabpb_list    += vocabnone_list
            end2endpb_list  += end2endnone_list

        # 待仿真数据拼接
        if self.is_pad:
            data_list, label_list, syllable_list, vocab_list, end2end_list, sequence_len, _ = self.__padData__(data_list, label_list, syllable_list, vocab_list, end2end_list, self.pad_sentnum)
            datapb_list, labelpb_list, syllablepb_list, vocabpb_list, end2endpb_list, sequencepb_len, _ = self.__padData__(datapb_list, labelpb_list, syllablepb_list, vocabpb_list, end2endpb_list, self.pad_sentnum)

        # 单通道数据仿真
        if self.lmdb_reverbfile != None and len(datanosie_list) != 0:
            data_len       = len(data_list)
            addreverb_len  = 0
            addnoise_len   = 0
            addreverb_list = []
            addnoise_list  = []
            # 高保真加混响加噪
            if self.is_reverb:
                addreverb_len = min(math.floor(data_len*self.reverb_coef), data_len)
                if addreverb_len>0:
                    addreverb_list = delta.conv_reverb(data_list[:addreverb_len], self.reverb_data, random_seed=self.rng.integers(1, 5e2), num_thread=8)
                    addreverb_list = delta.addnoise(addreverb_list, datanosie_list, self.noise_snr, random_seed=self.rng.integers(1, 5e2), num_thread=8)
            # 高保真加噪声
            if self.is_addnoise:
                addnoise_len = min(math.ceil((data_len-addreverb_len)*self.addnoise_coef), data_len-addreverb_len)
                if addnoise_len>0:
                    addnoise_list = delta.addnoise(data_list[addreverb_len : addreverb_len+addnoise_len], datanosie_list, self.noise_snr, random_seed=self.rng.integers(1, 5e2), num_thread=8)
            data_list = addreverb_list+addnoise_list+data_list[addreverb_len+addnoise_len:]
            
            # 回放加噪声
            datapb_len      = len(datapb_list)
            addnoisepb_len  = 0
            addnoisepb_list = []
            if self.is_addnoisepb:
                addnoisepb_len = min(math.floor(data_len*self.addnoisepb_coef), data_len)
                if addnoisepb_len>0:
                    addnoisepb_list = delta.addnoise(datapb_list[:addnoisepb_len], datanosie_list, self.noise_snrpb, random_seed=self.rng.integers(1, 5e2), num_thread=8)
            datapb_list = addnoisepb_list + datapb_list[addnoisepb_len:]

        # 单通道仿真数据/回放数据/多通道仿真数据拼接
        if len(datapb_list) !=0 :
            data_list     += datapb_list
            label_list    += labelpb_list
            syllable_list += syllablepb_list
            vocab_list    += vocabpb_list
            end2end_list  += end2endpb_list
            sequence_len  += sequencepb_len
        if len(datamc_list) !=0 :
            data_list     += datamc_list
            label_list    += labelmc_list
            syllable_list += syllablemc_list
            vocab_list    += vocabmc_list
            end2end_list  += end2endmc_list
            sequence_len  += sequencemc_len

        # 幅值变换
        amp = self.rng.choice(self.amp_list, len(data_list), replace=False)
        data_list, label_list, syllable_list, vocab_list, end2end_list, sequence_len = self.__ampChange__(data_list, label_list, syllable_list, vocab_list, end2end_list, sequence_len, amp)

        # 提特征
        fb40_list = delta.build_fb40(data_list)
        if len(data_list) == 1:
            data_list = [fb40_list]
        else:
            data_list = fb40_list

        # FeaNorm 减均值除方差
        data_list, label_list, syllable_list, end2end_list = self.__calFeanorm__(data_list, label_list, syllable_list, end2end_list)

        if self.is_pad:            
            # 数据对齐 & numpy转torch & 句首非满视野扰动
            length_del = self.rng.integers(0, self.train_padhead_frame)
            data, data_mask, _ = self.__pad_nmod_alongT__(data_list, self.nmod_pad, 0, length_del)
            label, _, sequence_len = self.__pad_nmod_alongT__(label_list, self.nmod_pad, -2, length_del, sequence_len)
            syllable, _, _ = self.__pad_nmod_alongT__(syllable_list, self.nmod_pad, -2, length_del)
            end2end, _, _ = self.__pad_nmod_alongT__(end2end_list, self.nmod_pad, -2, length_del)
            # vocab标注格式转换
            tmp_vocab = []
            for item in vocab_list:
                tmp_vocab.extend(item)
            vocab_list = tmp_vocab[:]
        else:
            # 数据对齐 & numpy转torch
            sequence_len = [ [0, np.shape(item)[0]] for item in label_list ]
            data, data_mask = self.__pad_nmod_alongB__(data_list, self.nmod_pad, 0)
            label, _ = self.__pad_nmod_alongB__(label_list, self.nmod_pad, -2)
            syllable, _ = self.__pad_nmod_alongB__(syllable_list, self.nmod_pad, -2)
            end2end, _ = self.__pad_nmod_alongB__(end2end_list, self.nmod_pad, -2)
             
        # vocal标注转换
        assert(len(vocab_list) == len(sequence_len))
        win_unit, win_unitlist, win_unitlen, win_attnmask, win_datamask, win_datalen, max_unitlen, max_datalen, sequence_len, sequence_ignore = self.__gen_vocablable__(vocab_list, sequence_len, min_datalen=0, nmod_pad=self.nmod_pad, nmod_node=4)
        # win_unit, win_unitlist, win_unitlen, win_attnmask, win_datamask, win_datalen, max_unitlen, max_datalen, sequence_len = self.__gen_syllablelable__(syllable, sequence_len, min_datalen=64, nmod_pad=8, nmod_node=4)

        assert(data.size(3)==label.size(3)==syllable.size(3)==end2end.size(3))

        # 数据转换和输出
        labasr = label[:, :, 1, :].clone()
        labasr = labasr.squeeze(0).transpose(1, 0)
        label  = label[:, :, 0, :].clone()
        label_mask = label.clone()
        label_mask[label_mask >= 0] = 1
        label_mask[label_mask < 0] = 0
        label = label.squeeze(0).transpose(1, 0)
        syllable = syllable.squeeze(0).squeeze(0).transpose(1, 0)
        end2end = end2end.squeeze(0).squeeze(0).transpose(1, 0)
        label_mask = label_mask.squeeze(0).transpose(1, 0)
        meta = {}
        meta["mask"] = data_mask.contiguous().detach()
        meta["frames_label"] = label.float().contiguous().detach()
        meta["frames_asr"] = labasr.float().contiguous().detach()
        meta["syllable_label"] = syllable.float().contiguous().detach()
        meta["end2end_label"] = end2end.float().contiguous().detach()
        meta["frames_mask"] = label_mask.float().contiguous().detach()
        meta["e2e_winsize"] = self.train_e2e_winsize
        meta["cmdlen_mask"] = self.cmd_len_mask.float().contiguous().detach()
        meta['sequence_len'] = sequence_len.int().contiguous().detach()
        meta['sequence_ignore'] = sequence_ignore
        meta['win_unit'] = win_unit.long().contiguous().detach()
        meta['win_unitlist'] = win_unitlist
        meta['win_unitlen'] = win_unitlen.int().contiguous().detach()
        meta['win_attnmask'] = win_attnmask.int().contiguous().detach()
        meta['win_datamask'] = win_datamask.int().contiguous().detach()
        meta['win_datalen'] = win_datalen.int().contiguous().detach()
        meta['max_unitlen'] = torch.tensor(max_unitlen).detach()
        meta['max_datalen'] = torch.tensor(max_datalen).detach()
        meta["rnn_mask"] = data_mask.transpose(1, 0).unsqueeze(2).contiguous().detach()
        
        return data.detach(), meta

class CollaterEval():
    def __init__(self, nmod_pad, train_e2e_winsize, lmdb_cmdlist, train_padhead_frame, train_padtail_frame, seed):
        # 随机种子
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

        self.nmod_pad          = nmod_pad
        self.train_e2e_winsize = train_e2e_winsize
        self.train_padhead_frame = train_padhead_frame
        self.train_padtail_frame = train_padtail_frame
        self.cmd_len_mask      = func_calCmdBoundary(lmdb_cmdlist, self.train_e2e_winsize)

    def __pad_nmod__(self, sequence, nmod_pad, val):
        maxlen = 0
        pad_list = []
        mask_list = []
        for nparray in sequence:
            if nparray.shape[0] > maxlen:
                maxlen = nparray.shape[0]
        if nmod_pad is not None:
            maxlen = maxlen if maxlen % nmod_pad == 0 else maxlen + nmod_pad - maxlen % nmod_pad
        for nparray in sequence:
            padlen = maxlen - nparray.shape[0]
            nparray = np.pad(nparray, ((0, padlen), (0, 0)), mode="constant", constant_values=(val,))
            nparray = nparray.transpose(1, 0)
            torcharray = torch.from_numpy(nparray).float()
            torchmask = torch.ones(1, torcharray.size()[1])
            if padlen > 0:
                torchmask[:, -padlen:] = 0
            torcharray = torcharray.reshape(1, 1, torcharray.size()[0], torcharray.size()[1])
            pad_list.append(torcharray)
            mask_list.append(torchmask)
        batch_array = torch.cat(pad_list, dim=0)
        batch_mask = torch.cat(mask_list, dim=0)
        return batch_array, batch_mask
		
    def __gen_vocablable__(self, vocab_in, node_in, min_datalen, nmod_pad, nmod_node, is_addend=False, endnode=1):
        # 检查数据长度
        tmp_vocab = []
        for item in vocab_in:
            tmp_vocab.extend(item)
        vocab_in = tmp_vocab[:]
        assert(len(vocab_in) == len(node_in))

        max_datalen = 0
        seq_datalen = []
        max_unitlen = 0
        seq_unitlen = []
        vocab_out   = []
        node_out    = []
        ignore_out  = []
        for node_idx, (bg_node, ed_node) in enumerate(node_in):
            bg_nmod = math.floor(bg_node/nmod_node)
            ed_nmod = math.floor(ed_node/nmod_node)

            # 音频最小长度
            tmp_datalen = ed_nmod - bg_nmod
            if tmp_datalen < min_datalen:
                bg_nmod = max(0, ed_nmod-min_datalen)
                tmp_datalen = ed_nmod - bg_nmod

            # 音频最大长度
            if tmp_datalen > max_datalen:
                max_datalen = tmp_datalen
            
            tmp_unit = vocab_in[node_idx]
            if -2 in np.squeeze(tmp_unit):
                ignore_out.append(False)
            else:
                ignore_out.append(True)
            # 添加尾节点
            if is_addend:
                tmp_unit = np.append(tmp_unit, np.array([[endnode]]), axis=0)
            tmp_unitlen = len(tmp_unit)
            
            # 文本最大长度
            if tmp_unitlen > max_unitlen:
                max_unitlen = tmp_unitlen
            
            # 长度和端点输出
            seq_datalen.append(tmp_datalen)
            seq_unitlen.append(tmp_unitlen)
            vocab_out.append(np.squeeze(tmp_unit))
            node_out.append([bg_nmod, ed_nmod])

        # 文本补齐和mask
        max_datalen = max_datalen if max_datalen % nmod_pad == 0 else max_datalen + nmod_pad - max_datalen % nmod_pad
        node_num = len(node_out)
        matrix_unit  = np.ones((node_num, max_unitlen)) * (-2)
        matrix_attnmask = np.zeros((node_num, max_datalen, max_datalen))
        matrix_datamask = np.zeros((node_num, max_datalen))
        for i in range(max_datalen):
            matrix_attnmask[:, i, max(0, i-16):min(max_datalen, i+16)] = 1
        for index, (bg_nmod, ed_nmod) in enumerate(node_out):

            tmp_unitlen = seq_unitlen[index]
            tmp_datalen = seq_datalen[index]
            matrix_unit[index, :tmp_unitlen] = vocab_out[index]
            matrix_attnmask [index, tmp_datalen:, :] = 0
            matrix_attnmask [index, :, tmp_datalen:] = 0
            matrix_datamask [index, :tmp_datalen] = 1

        matrix_unit  = torch.from_numpy( matrix_unit )                    # (seq_num, max_unitlen)
        seq_unitlen  = torch.from_numpy( np.array(seq_unitlen) )          # (seq_num)
        matrix_attnmask = torch.from_numpy( matrix_attnmask )             # (seq_num, max_datalen, max_datalen)
        matrix_datamask = torch.from_numpy( matrix_datamask )             # (seq_num, max_datalen)
        seq_datalen  = torch.from_numpy( np.array(seq_datalen) )
        node_out = torch.from_numpy( np.array(node_out) )
        return matrix_unit, vocab_out, seq_unitlen, matrix_attnmask, matrix_datamask, seq_datalen, max_unitlen, max_datalen, node_out, ignore_out

    def __gen_syllablelable__(self, syllable_in, node_in, min_datalen, nmod_pad, nmod_node):
        syllable_in = syllable_in[:, :, :, ::nmod_node].squeeze()

        max_datalen = 0
        seq_datalen = []
        max_unitlen = 0
        seq_unit = []
        seq_unitlen = []
        node_out = []
        for bg_node, ed_node in node_in:
            bg_nmod = math.floor(bg_node/nmod_node)
            ed_nmod = math.floor(ed_node/nmod_node)

            # 音频最小长度
            tmp_datalen = ed_nmod - bg_nmod
            if tmp_datalen < min_datalen:
                bg_nmod = max(0, ed_nmod-min_datalen)
                tmp_datalen = ed_nmod - bg_nmod

            # 音频最大长度
            if tmp_datalen > max_datalen:
                max_datalen = tmp_datalen

            # 低帧率坍塌后的label & 跳过syllable有缺失的句子
            tmp_label = syllable_in[bg_nmod:ed_nmod]
            if -3 in tmp_label:
                continue
            tmp_unit = []
            for key, value in groupby(tmp_label):
                if key <= 1332 and key > 0:    # triphone(0~3002) 3003 </s> 3004 blank
                    tmp_unit.append( key ) # ctc-syllable(1~1332) 0 blank 1328 Error       
            tmp_unitlen = len(tmp_unit)

            # 文本最大长度
            if tmp_unitlen > max_unitlen:
                max_unitlen = tmp_unitlen
            
            # 文本长度和端点输出
            seq_datalen.append(tmp_datalen)
            seq_unit.append(tmp_unit)
            seq_unitlen.append(tmp_unitlen)
            node_out.append([bg_nmod, ed_nmod])

        # 文本补齐和mask
        max_datalen = max_datalen if max_datalen % nmod_pad == 0 else max_datalen + nmod_pad - max_datalen % nmod_pad
        node_num = len(node_out)
        matrix_unit  = np.ones((node_num, max_unitlen)) * (-2)
        matrix_attnmask = np.zeros((node_num, max_datalen, max_datalen))
        matrix_datamask = np.zeros((node_num, max_datalen))
        for i in range(max_datalen):
            matrix_attnmask[:, i, max(0, i-16):min(max_datalen, i+16)] = 1
        seq_datalen_expand = []
        for index, (bg_nmod, ed_nmod) in enumerate(node_out):

            tmp_unitlen = seq_unitlen[index]
            tmp_datalen = seq_datalen[index]
            matrix_unit[index, :tmp_unitlen] = seq_unit[index]
            matrix_attnmask [index, tmp_datalen:, :] = 0
            matrix_attnmask [index, :, tmp_datalen:] = 0
            matrix_datamask [index, :tmp_datalen] = 1

        matrix_unit  = torch.from_numpy( matrix_unit )                    # (seq_num, max_unitlen)
        seq_unitlen  = torch.from_numpy( np.array(seq_unitlen) )          # (seq_num)
        matrix_attnmask = torch.from_numpy( matrix_attnmask )             # (seq_num, max_datalen, max_datalen)
        matrix_datamask = torch.from_numpy( matrix_datamask )             # (seq_num, max_datalen)
        seq_datalen  = torch.from_numpy( np.array(seq_datalen) )
        node_out = torch.from_numpy( np.array(node_out) )
        return matrix_unit, seq_unit, seq_unitlen, matrix_attnmask, matrix_datamask, seq_datalen, max_unitlen, max_datalen, node_out

    def __call__(self, batch):
        data_list     = []
        label_list    = []
        syllable_list = []
        vocab_list    = []
        end2end_list  = []
        sequence_len  = []
        for data, label, syllable, vocab, end2end in batch:
            data_list.extend(data)
            label_list.extend(label)
            syllable_list.extend(syllable)
            end2end_list.extend(end2end)
            for eachindex, eachpblabel in enumerate(label_list):
                sequence_len.append(np.array([self.train_padhead_frame, eachpblabel.shape[0]-self.train_padtail_frame]))
                vocab_list.append([ vocab[eachindex] ])

        data, data_mask = self.__pad_nmod__(data_list, self.nmod_pad, 0)
        label, _ = self.__pad_nmod__(label_list, self.nmod_pad, -2)
        syllable, _ = self.__pad_nmod__(syllable_list, self.nmod_pad, -2)
        end2end, _ = self.__pad_nmod__(end2end_list, self.nmod_pad, -2)
        sequence_len = torch.from_numpy(np.array(sequence_len))
        win_unit, win_unitlist, win_unitlen, win_attnmask, win_datamask, win_datalen, max_unitlen, max_datalen, sequence_len, sequence_ignore = self.__gen_vocablable__(vocab_list, sequence_len, min_datalen=64, nmod_pad=8, nmod_node=4)
        # win_unit, win_unitlist, win_unitlen, win_attnmask, win_datamask, win_datalen, max_unitlen, max_datalen, sequence_len = self.__gen_syllablelable__(syllable, sequence_len, min_datalen=64, nmod_pad=8, nmod_node=4)

        assert(data.size(3)==label.size(3)==syllable.size(3)==end2end.size(3))

        # 数据转换和输出
        labasr = label[:, :, 1, :].clone()
        labasr = labasr.squeeze(0).transpose(1, 0)
        label  = label[:, :, 0, :].clone()
        label_mask = label.clone()
        label_mask[label_mask >= 0] = 1
        label_mask[label_mask < 0] = 0
        label = label.squeeze(0).transpose(1, 0)
        syllable = syllable.squeeze(0).squeeze(0).transpose(1, 0)
        end2end = end2end.squeeze(0).squeeze(0).transpose(1, 0)
        label_mask = label_mask.squeeze(0).transpose(1, 0)
        meta = {}
        meta["mask"] = data_mask.contiguous().detach()
        meta["frames_label"] = label.float().contiguous().detach()
        meta["frames_asr"] = labasr.float().contiguous().detach()
        meta["syllable_label"] = syllable.float().contiguous().detach()
        meta["end2end_label"] = end2end.float().contiguous().detach()
        meta["frames_mask"] = label_mask.float().contiguous().detach()
        meta["e2e_winsize"] = self.train_e2e_winsize
        meta["cmdlen_mask"] = self.cmd_len_mask.float().contiguous().detach()
        meta['sequence_len'] = sequence_len.int().contiguous().detach()
        meta['sequence_ignore'] = sequence_ignore
        meta['win_unit'] = win_unit.long().contiguous().detach()
        meta['win_unitlist'] = win_unitlist
        meta['win_unitlen'] = win_unitlen.int().contiguous().detach()
        meta['win_attnmask'] = win_attnmask.int().contiguous().detach()
        meta['win_datamask'] = win_datamask.int().contiguous().detach()
        meta['win_datalen'] = win_datalen.int().contiguous().detach()
        meta['max_unitlen'] = torch.tensor(max_unitlen).detach()
        meta['max_datalen'] = torch.tensor(max_datalen).detach()
        meta["rnn_mask"] = data_mask.transpose(1, 0).unsqueeze(2).contiguous().detach()
        return data.detach(), meta

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.lmdb_env = lmdb.Environment(dataset.lmdb_path, readonly=True, readahead=True, lock=False)

def worker_init_fn_online(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    num_workers = worker_info.num_workers
    worker_idx = worker_info.id

    # 随机种子
    rng = np.random.default_rng()
    seed = dataset.seed
    if seed is not None:
        rng = np.random.default_rng(seed)

    # 单通道初始化lmdb
    dataset.lmdb_env = lmdb.Environment(dataset.lmdb_path, readonly=True, readahead=True, lock=False)
    if dataset.lmdbpb_path != None:
        dataset.lmdb_pb = lmdb.Environment(dataset.lmdbpb_path, readonly=True, readahead=True, lock=False)

    if dataset.lmdbuni_path != None:
        # 获取每个进程的keys
        step_uni = int(dataset.len_uni/num_workers)
        dataset.pointer_uni_bg = step_uni*worker_idx
        dataset.pointer_uni_ed = min(step_uni*(worker_idx+1), dataset.len_uni)
        dataset.pointer_uni = rng.integers(0, dataset.pointer_uni_ed - dataset.pointer_uni_bg - 1) # 随机起始点顺序读取
        dataset.datauni_keys = dataset.datauni_keys[dataset.pointer_uni_bg : dataset.pointer_uni_ed]
        # 初始化lmdb
        dataset.lmdb_uni = lmdb.Environment(dataset.lmdbuni_path, readonly=True, readahead=True, lock=False)

    if dataset.lmdbunipb_path != None:
        # 获取每个进程的keys
        step_unipb = int(dataset.len_unipb/num_workers)
        dataset.pointer_unipb_bg = step_unipb*worker_idx
        dataset.pointer_unipb_ed = min(step_unipb*(worker_idx+1), dataset.len_unipb)
        dataset.pointer_unipb = rng.integers(0, dataset.pointer_unipb_ed - dataset.pointer_unipb_bg - 1) # 随机起始点顺序读取
        dataset.dataunipb_keys = dataset.dataunipb_keys[dataset.pointer_unipb_bg : dataset.pointer_unipb_ed]
        # 初始化lmdb
        dataset.lmdb_unipb = lmdb.Environment(dataset.lmdbunipb_path, readonly=True, readahead=True, lock=False)

    if dataset.lmdbnone_path != None:
        # 获取每个进程的keys
        step_none = int(dataset.len_none/num_workers)
        dataset.pointer_none_bg = step_none*worker_idx
        dataset.pointer_none_ed = min(step_none*(worker_idx+1), dataset.len_none)
        dataset.pointer_none = rng.integers(0, dataset.pointer_none_ed - dataset.pointer_none_bg - 1) # 随机起始点顺序读取
        dataset.datanone_keys = dataset.datanone_keys[dataset.pointer_none_bg : dataset.pointer_none_ed]
        # 初始化lmdb
        dataset.lmdb_none = lmdb.Environment(dataset.lmdbnone_path, readonly=True, readahead=True, lock=False)
    
    if dataset.lmdbnoise_path != None:
        for noise_index in range(len(dataset.lmdbnoise_path)):
            # 获取每个进程的keys
            len_noise = getattr(dataset, "len_noise_"+str(noise_index))
            step_noise = int(len_noise/num_workers)
            pointer_noise_bg = step_noise*worker_idx
            pointer_noise_ed = min(step_noise*(worker_idx+1), len_noise)
            setattr(dataset, "pointer_noise_bg_"+str(noise_index), pointer_noise_bg)
            setattr(dataset, "pointer_noise_ed_"+str(noise_index), pointer_noise_ed)
            sample_int = rng.integers(0, pointer_noise_ed - pointer_noise_bg - 1) # 随机起始点顺序读取
            setattr(dataset, "pointer_noise_"+str(noise_index), sample_int)
            dataset.datanoise_keysDict[noise_index] = dataset.datanoise_keysDict[noise_index][pointer_noise_bg: pointer_noise_ed]
            # 初始化lmdb
            setattr(dataset, "lmdb_noise_"+str(noise_index), lmdb.Environment(dataset.lmdbnoise_path[noise_index], readonly=True, readahead=True, lock=False))

    if dataset.lmdbextra_path != None:
        # 获取每个进程的keys
        step_extra = int(dataset.len_extra/num_workers)
        dataset.pointer_extra_bg = step_extra*worker_idx
        dataset.pointer_extra_ed = min(step_extra*(worker_idx+1), dataset.len_extra)
        dataset.pointer_extra = rng.integers(0, dataset.pointer_extra_ed - dataset.pointer_extra_bg - 1) # 随机起始点顺序读取
        dataset.dataextra_keys = dataset.dataextra_keys[dataset.pointer_extra_bg : dataset.pointer_extra_ed]
        # 初始化lmdb
        dataset.lmdb_extra = lmdb.Environment(dataset.lmdbextra_path, readonly=True, readahead=True, lock=False)

    if dataset.lmdbextrapb_path != None:
        # 获取每个进程的keys
        step_extrapb = int(dataset.len_extrapb/num_workers)
        dataset.pointer_extrapb_bg = step_extrapb*worker_idx
        dataset.pointer_extrapb_ed = min(step_extrapb*(worker_idx+1), dataset.len_extrapb)
        dataset.pointer_extrapb = rng.integers(0, dataset.pointer_extrapb_ed - dataset.pointer_extrapb_bg - 1) # 随机起始点顺序读取
        dataset.dataextrapb_keys = dataset.dataextrapb_keys[dataset.pointer_extrapb_bg : dataset.pointer_extrapb_ed]
        # 初始化lmdb
        dataset.lmdb_extrapb = lmdb.Environment(dataset.lmdbextrapb_path, readonly=True, readahead=True, lock=False)

    # 多通道初始化lmdb
    if dataset.lmdbinter_path != None:
        # 获取每个进程的keys
        step_inter = int(dataset.len_inter/num_workers)
        dataset.pointer_inter_bg = step_inter*worker_idx
        dataset.pointer_inter_ed = min(step_inter*(worker_idx+1), dataset.len_inter)
        dataset.pointer_inter = rng.integers(0, dataset.pointer_inter_ed - dataset.pointer_inter_bg - 1) # 随机起始点顺序读取
        dataset.datainter_keys = dataset.datainter_keys[dataset.pointer_inter_bg : dataset.pointer_inter_ed]
        # 初始化lmdb
        dataset.lmdb_inter = lmdb.Environment(dataset.lmdbinter_path, readonly=True, readahead=True, lock=False)
    
    if dataset.lmdbinterpb_path != None:
        # 获取每个进程的keys
        step_interpb = int(dataset.len_interpb/num_workers)
        dataset.pointer_interpb_bg = step_interpb*worker_idx
        dataset.pointer_interpb_ed = min(step_interpb*(worker_idx+1), dataset.len_interpb)
        dataset.pointer_interpb = rng.integers(0, dataset.pointer_interpb_ed - dataset.pointer_interpb_bg - 1) # 随机起始点顺序读取
        dataset.datainterpb_keys = dataset.datainterpb_keys[dataset.pointer_interpb_bg : dataset.pointer_interpb_ed]
        # 初始化lmdb
        dataset.lmdb_interpb = lmdb.Environment(dataset.lmdbinterpb_path, readonly=True, readahead=True, lock=False)
    
    if dataset.lmdbrir_path != None:
        # 获取每个进程的keys
        step_rir = int(dataset.len_rir/num_workers)
        dataset.pointer_rir_bg = step_rir*worker_idx
        dataset.pointer_rir_ed = min(step_rir*(worker_idx+1), dataset.len_rir)
        dataset.pointer_rir = rng.integers(0, dataset.pointer_rir_ed - dataset.pointer_rir_bg - 1) # 随机起始点顺序读取
        dataset.datarir_keys = dataset.datarir_keys[dataset.pointer_rir_bg : dataset.pointer_rir_ed]
        # 初始化lmdb
        dataset.lmdb_rir = lmdb.Environment(dataset.lmdbrir_path, readonly=True, readahead=True, lock=False)
    
    if dataset.lmdbmusic_path != None:
        # 获取每个进程的keys
        step_music = int(dataset.len_music/num_workers)
        dataset.pointer_music_bg = step_music*worker_idx
        dataset.pointer_music_ed = min(step_music*(worker_idx+1), dataset.len_music)
        dataset.pointer_music = rng.integers(0, dataset.pointer_music_ed - dataset.pointer_music_bg - 1) # 随机起始点顺序读取
        dataset.datamusic_keys = dataset.datamusic_keys[dataset.pointer_music_bg : dataset.pointer_music_ed]
        # 初始化lmdb
        dataset.lmdb_music = lmdb.Environment(dataset.lmdbmusic_path, readonly=True, readahead=True, lock=False)

    if dataset.lmdbmcnoise_path != None:
        # 获取每个进程的keys
        step_mcnoise = int(dataset.len_mcnoise/num_workers)
        dataset.pointer_mcnoise_bg = step_mcnoise*worker_idx
        dataset.pointer_mcnoise_ed = min(step_mcnoise*(worker_idx+1), dataset.len_mcnoise)
        dataset.pointer_mcnoise = rng.integers(0, dataset.pointer_mcnoise_ed - dataset.pointer_mcnoise_bg - 1) # 随机起始点顺序读取
        dataset.datamcnoise_keys = dataset.datamcnoise_keys[dataset.pointer_mcnoise_bg : dataset.pointer_mcnoise_ed]
        # 初始化lmdb
        dataset.lmdb_mcnoise = lmdb.Environment(dataset.lmdbmcnoise_path, readonly=True, readahead=True, lock=False)

def get_train_dataloader(args):
    speechdataset = SpeechDatasetTrain(
        args.lmdb_path, 
        args.lmdb_key,
        args.lmdbpb_path, 
        args.lmdbpb_key,
        args.lmdbuni_path, 
        args.lmdbuni_key,
        args.lmdbunipb_path, 
        args.lmdbunipb_key,
        args.lmdbnone_path, 
        args.lmdbnone_key,
        args.lmdbnoise_path,
        args.lmdbnoise_key,
        args.lmdbextra_path, 
        args.lmdbextra_key,
        args.lmdbextrapb_path, 
        args.lmdbextrapb_key,
        max_sent_frame=args.max_sent_frame, 
        start_line=args.start_line, 
        end_line=args.end_line,
        end_expand=args.end_expand,
        num_cmds=args.num_cmds,
        train_sent_main=args.train_sent_main,
        train_sent_pb=args.train_sent_pb,
        train_ratio_uni=args.train_ratio_uni,
        train_ratio_unipb=args.train_ratio_unipb,
        train_ratio_none=args.train_ratio_none,
        train_ratio_extra=args.train_ratio_extra, 
        train_ratio_extrapb=args.train_ratio_extrapb,
        train_padhead_frame=args.train_padhead_frame,
        train_padtail_frame=args.train_padtail_frame,
        rank=args.gpu_global_rank, 
        world_size=args.gpu_world_size,
        seed=args.seed,
        is_ivw=args.is_ivw,
        is_mixup=args.is_mixup,
        mixup_cmdid=args.mixup_cmdid,
		train_mainkwsid=args.train_mainkwsid,
        is_mc=args.is_mc,
        train_ratio_inter=args.train_ratio_inter,
        lmdbinter_path=args.lmdbinter_path,
        lmdbinter_key=args.lmdbinter_key,
        lmdbinterpb_path=args.lmdbinterpb_path,
        lmdbinterpb_key=args.lmdbinterpb_key,
        lmdbrir_path=args.lmdbrir_path,
        lmdbrir_key=args.lmdbrir_key,
        lmdbmusic_path=args.lmdbmusic_path,
        lmdbmusic_key=args.lmdbmusic_key,
        lmdbmcnoise_path=args.lmdbmcnoise_path,
        lmdbmcnoise_key=args.lmdbmcnoise_key
    )
    train_sampler = BunchSampler(
        dataset_lengths=speechdataset.data_lens, 
        batch_size=args.batch_size, 
        bunch_size=args.bunch_size, 
        drop_last=False, 
        shuffle_batch=True, 
        iter_num=args.train_iter_num, 
        seed=args.seed,
        is_sorted=args.is_sorted
    )
    train_dataloader = DataLoader(
        speechdataset, 
        batch_sampler=train_sampler, 
        num_workers=4, 
        collate_fn=CollaterTrain(
            args.nmod_pad, 
            args.train_e2e_winsize, 
            args.lmdb_reverbfile, 
            args.lmdb_cmdlist, 
            args.lmdb_normfile, 
            args.train_padhead_frame, 
            args.train_padtail_frame, 
            seed=args.seed,
            is_pad=args.is_pad,
            pad_sentnum=args.pad_sentnum,
            is_reverb=args.is_reverb,
            reverb_coef=args.reverb_coef,
            is_addnoise=args.is_addnoise,
            addnoise_coef=args.addnoise_coef,
            is_addnoisepb=args.is_addnoisepb,
            addnoisepb_coef=args.addnoisepb_coef,
            is_ivw=args.is_ivw,
            is_mc=args.is_mc
            ), 
        worker_init_fn=worker_init_fn_online, 
        multiprocessing_context="spawn"
    )
    return train_dataloader

def get_val_dataloader(args):
    speechdataset = SpeechDatasetEval(
        args.lmdb_path,
        args.lmdb_key,
        args.lmdb_normfile,
        max_sent_frame=args.max_sent_frame,
        start_line=0,
        end_line=args.val_sent_num,
        train_padhead_frame=args.train_padhead_frame,
        train_padtail_frame=args.train_padtail_frame,
        rank=0,
        world_size=1
    )
    val_dataloader = DataLoader(
        speechdataset, 
        batch_size=1, 
        num_workers=1, 
        shuffle=False,
        drop_last=False,
        collate_fn=CollaterEval(args.nmod_pad, args.train_e2e_winsize, args.lmdb_cmdlist, args.train_padhead_frame, args.train_padtail_frame, args.seed), 
        worker_init_fn=worker_init_fn, 
        multiprocessing_context="spawn"
    )
    return val_dataloader

def estimate_train_iter_num(args):    
    # 命令词仿真数据
    key_dict = {}
    for i, line in enumerate(open(args.lmdb_key)):
        line = line.strip()
        if i < args.start_line or line=="":
            continue
        items = line.split()
        sent_frame = int(items[1])
        sent_id = int(items[2])
        if sent_frame <= args.max_sent_frame:
            if sent_id in key_dict.keys():
                key_dict[sent_id].append(sent_frame)
            else:
                key_dict[sent_id] = [sent_frame]

    # 命令词回放数据
    keypb_dict = {}
    if args.lmdbpb_key is not None:
        for i, line in enumerate(open(args.lmdbpb_key)):
            line = line.strip()
            if i < args.start_line or line=="":
                continue
            items = line.split()
            sent_frame = int(items[1])
            sent_id = int(items[2])
            if sent_frame <= args.max_sent_frame:
                if sent_id in keypb_dict.keys():
                    keypb_dict[sent_id].append(sent_frame)
                else:
                    keypb_dict[sent_id] = [sent_frame]

    # 命令词仿真和回放数据混合
    max_train_iter_num = 0
    length_list = []
    train_sent_main = args.train_sent_main
    train_sent_pb = args.train_sent_pb
    assert(train_sent_main >= train_sent_pb)
    assert(args.num_cmds >= 1)

    # 定义命令词边界 浅定制只有0
    if args.num_cmds == 1:
        range_bg = 0
        range_ed = args.num_cmds
    else:
        range_bg = 1
        range_ed = args.num_cmds
    
    # 主唤醒id和数据量
    mainkwsid_dict = {}
    if args.train_mainkwsid is not None:
        mainkwsid_list = args.train_mainkwsid.split(",")
        for item in mainkwsid_list:
            item_list = [int(i) for i in item.split(":")]
            assert(len(item_list)==3)
            mainkwsid_dict[item_list[0]]=item_list[1:]

    for word_id in range(range_bg, range_ed):
        
        # 主唤醒词N倍数据量
        if word_id in mainkwsid_dict.keys(): 
            wkmain_value = mainkwsid_dict[word_id]
            train_sent_main = args.train_sent_main * wkmain_value[0]
            train_sent_pb = args.train_sent_pb * wkmain_value[1]
		
        # Case 1: 仿真和回放数据均存在
        if word_id in key_dict.keys() and word_id in keypb_dict.keys():
            remainder = train_sent_main

            # 获取回放数据
            valuepb = keypb_dict[word_id]
            keypb_len = len(valuepb)
            if train_sent_pb > 0:
                remainder = max(0, train_sent_main - train_sent_pb)
                ratio_pb = train_sent_pb // keypb_len + 1
                if ratio_pb > 1:
                    valuepb = np.tile( np.array(valuepb), ratio_pb )
                length_list.extend( valuepb[ : train_sent_pb] )
            
            # 获取仿真数据
            value = key_dict[word_id]
            key_len = len(value)
            ratio = remainder // key_len + 1
            if ratio > 1:
                value = np.tile(np.array(value), ratio )
            if remainder > 0:
                length_list.extend(value[: remainder])

        # Case 2: 仿真数据存在，回放数据不存在
        elif word_id in key_dict.keys() and word_id not in keypb_dict.keys():
            # 获取仿真数据
            value = key_dict[word_id]
            key_len = len(value)
            ratio = train_sent_main // key_len + 1
            if ratio > 1:
                value = np.tile(np.array(value), ratio )
            length_list.extend(value[: train_sent_main])
        
        # Case 3: 仿真数据不存在，回放数据存在
        elif word_id not in key_dict.keys() and word_id in keypb_dict.keys():
            # 获取回放数据
            valuepb = keypb_dict[word_id]
            keypb_len = len(valuepb)
            ratio_pb = train_sent_main // keypb_len + 1
            if ratio_pb > 1:
                valuepb = np.tile( np.array(valuepb), ratio_pb )
            length_list.extend( valuepb[ : train_sent_main] )

        # 重置仿真回放数据比
        train_sent_main = args.train_sent_main
        train_sent_pb = args.train_sent_pb

    # 是否设置最大keys长度
    if args.end_expand is not None:
        length_list = length_list[ : args.end_expand]
    
    # 计算batch迭代次数
    length_list = sorted(length_list)
    total_length = 0
    total_sent = 0
    for length in length_list:
        total_length += length
        total_sent += 1
        if total_length + length > args.bunch_size or total_sent == args.batch_size:
            max_train_iter_num += 1
            total_length = 0
            total_sent = 0
    if total_length > 0:
        max_train_iter_num += 1
    max_train_iter_num = max_train_iter_num // args.gpu_world_size
    return max_train_iter_num+100, len(length_list)
	
