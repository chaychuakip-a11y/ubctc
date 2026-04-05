from copy import copy
import math
from sys import flags
from numpy.core.fromnumeric import squeeze
from numpy.core.numeric import ones
from torch._C import dtype
try:
    from .datum_pb2 import SpeechDatum #Float MC
    from .datum1_pb2 import Datum1 #diffu rir
    from .datum2_pb2 import Datum2 # czzhu2
except:
    from datum_pb2 import SpeechDatum #Float MC
    from datum1_pb2 import Datum1 #diffu rir
    from datum2_pb2 import Datum2 # czzhu2
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
from typing import Sequence
from operator import index, itemgetter
from itertools import groupby, repeat
import os
import sys
import scipy.io as sio
import delta
import wave

import random

# 全局变量
sent_padcount = 3
padsil_label  = 15002
down_wav_flag = 0
voc_size      = 3261

def write_audio(audio_data, audio_name, nchannels=1):
    nchannels = 1
    sampwidth = 2
    framerate = 16000
    nframes   = len(audio_data)
    wave_file = wave.open(audio_name,"wb")
    wave_file.setparams(((nchannels, sampwidth, framerate, nframes, 'NONE', 'NONE')))
    wave_file.writeframes(np.array(audio_data, dtype="int16").tobytes())
    wave_file.close()
    return 0

def get_lmdb_key(lmdb_path):
    lmdb_key = lmdb_path + '/keys_lens.txt'
    if os.path.exists(lmdb_key):
        return lmdb_key
    elif os.path.exists(lmdb_path + '/key.txt'):
        lmdb_key = lmdb_path + '/key.txt'
        return lmdb_key
    else:
        assert(1==0)
        # , "lmdb has no keys in %s"%lmdb_path)

def down_lab(data, txt_name):
    ttt = 0
    if down_wav_flag:
        for idx in range(len(data )):
            with open(txt_name+'_%d.txt'%ttt, 'a') as f:
                for idxj in range(data[idx].shape[0]):
                    f.writelines('%d\t'%data[idx][idxj,0])
                f.writelines('\n')

def down_audio(audio_data, audio_name, nchannels=1):
    if down_wav_flag:
        for idx in range(len(audio_data )):
            write_audio(audio_data[idx], audio_name+'_%d.wav'%idx, nchannels)

def get_length(data, nmod=1):
    t = data.size(0)
    s = data.size(1)
    lengths= torch.zeros((s), dtype=torch.float32, device=data.device)
    for j in range(s):
        for i in range(t):
            if((data[t-1-i,j] == 1)):
                lengths[j]=int( (t-i+nmod-1)/nmod )
                break
    return lengths

class SpeechDatasetSingleCh(Dataset):
    def __init__(self, lmdb_path, lmdb_normfile, max_sent_frame=100000000, min_sent_frame=40, start_line=0, end_line=None, train_padtail_frame=0, world_size=1, rank=0):
        self.lmdb_path = lmdb_path
        self.lmdb_env  = None
        self.train_padtail_frame = train_padtail_frame
        self.data_keys = []
        self.data_lens = []
        self.data_id   = []
        self.max_sent_frame = max_sent_frame
        self.min_sent_frame = min_sent_frame
        for i,line in enumerate(open(get_lmdb_key(self.lmdb_path))):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            if (i - rank) % world_size == 0:
                items = line.strip().split()
                sent_frame = int(items[1])
                sent_id = int(items[2])
                if sent_frame <= self.max_sent_frame and sent_frame >= self.min_sent_frame:
                    self.data_keys.append(items[0])
                    self.data_lens.append(sent_frame)
                    self.data_id.append(sent_id)
        with open(lmdb_normfile, 'r') as nf:
            normfile_lines = nf.read().splitlines()
        self.mel_spec_mean = np.array([float(item) for item in normfile_lines[1:41]])
        self.mel_spec_std  = np.array([float(item) for item in normfile_lines[42:82]])
        self.padzeros_mch  = np.zeros((1, 40))

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        label_out    = []
        syllable_out = []
        wordid_out   = []
        txn = self.lmdb_env.begin()
        with txn.cursor() as cursor:
            k = self.data_keys[index].encode('utf-8')
            cursor.set_key(k)
            datum    = Datum2()
            datum.ParseFromString(cursor.value())
            data     = np.frombuffer(datum.anc.data, dtype=np.int16)
            label    = np.frombuffer(datum.anc.state_data, dtype=np.int16).reshape(-1, 1)
            syllable = np.fromstring(datum.anc.syllable_data, dtype=np.int16).reshape(-1, 1)
            rnnt_lab = syllable[syllable>=0]
            rnnt_lab[rnnt_lab > 1660] = rnnt_lab[rnnt_lab > 1660] - 6739
            res      = label.shape[0] - rnnt_lab.shape[0] -2
            rnnt_lab = np.concatenate(([voc_size-2], rnnt_lab, [voc_size-1], [-1]*res))
            syllable = rnnt_lab.reshape(-1, 1)
            wav_name = datum.anc.wave_name.decode()
            wordid   = wav_name.split('_')[-1]
            if wordid == 'CN':
                wordid = 0
            elif wordid == 'EN':
                wordid = 1
            elif wordid == 'CAN':
                wordid = 2
            elif wordid == 'CNEN':
                wordid = 3
            elif wordid == 'SC':
                wordid = 4
            else:
                wordid = -1
            
            data = delta.build_fb40(data, num_thread=8)
            data = (data - self.mel_spec_mean[None,:]) * self.mel_spec_std
            nframes = min(data.shape[0], label.shape[0], syllable.shape[0])
            data = data[:nframes,:]
            padzeros_mch = np.tile(self.padzeros_mch, (nframes, 1))
            data = np.concatenate( (data, padzeros_mch), axis=1)
            label_out.append(label[:nframes, :])
            syllable_out.append(syllable[:nframes, :])
            wordid_out.append(wordid)

        return [data], label_out, syllable_out, wordid_out

class SpeechDatasetMultiCh(Dataset):
    def __init__(self, lmdb_path, lmdb_key, lmdb_normfile, lmdb_syllableid, max_sent_frame=100000000, min_sent_frame=40, start_line=0, end_line=None, world_size=1, rank=0, is_ivw=False):
        self.lmdb_path = lmdb_path
        self.lmdb_key = lmdb_key
        self.lmdb_env = None
        self.data_keys = []
        self.data_lens = []
        self.data_id = []
        self.max_sent_frame = max_sent_frame
        self.min_sent_frame = min_sent_frame
        for i,line in enumerate(open(lmdb_key)):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            if (i - rank) % world_size == 0:
                items = line.strip().split()
                sent_frame = int(items[1])
                sent_id = int(items[2])
                if sent_frame <= self.max_sent_frame and sent_frame >= self.min_sent_frame:
                    self.data_keys.append(items[0])
                    self.data_lens.append(sent_frame)
                    self.data_id.append(sent_id)
        with open(lmdb_normfile, 'r') as nf:
            normfile_lines = nf.read().splitlines()
        self.mel_spec_mean_mc = np.array([float(item) for item in normfile_lines[1:41]])
        self.mel_spec_std_mc = np.array([float(item) for item in normfile_lines[42:82]])
        self.phone2syllableDict = np.load(lmdb_syllableid, allow_pickle=True)[()]

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        txn = self.lmdb_env.begin()
        with txn.cursor() as cursor:
            k = self.data_keys[index].encode('utf-8')
            cursor.set_key(k)
            datum = SpeechDatum()
            datum.ParseFromString(cursor.value())
            data = np.fromstring(datum.anc.data, dtype=np.float32).reshape(-1, 240)[:,:160]
            label = np.fromstring(datum.anc.state_data, dtype=np.int16).reshape(-1, 2)
            label_frame = label[:, 0].reshape(-1, 1)
            label_e2e = label[:, 1].reshape(-1, 1)
            data = (data - self.mel_spec_mean_mc[None,:]) * self.mel_spec_std_mc
        return [data], label

class SpeechDataset_Online(Dataset):
    def __init__(self,lmdbuni_path, lmdbunipb_path, lmdbnoise_path, lmdbinter_path, lmdbrir_path, \
                lmdbmusic_path, max_sent_frame=100000000, min_sent_frame=40, start_line=0, end_line=None, \
                train_ratio_uni=0, train_ratio_unipb=0, train_ratio_inter=0, \
                train_padhead_frame=0, train_padtail_frame=0, world_size=1, rank=0, seed=None):
        # 随机种子
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

        # 初始化
        self.lmdbuni_path     = lmdbuni_path
        self.lmdbunipb_path   = lmdbunipb_path
        self.lmdbnoise_path   = lmdbnoise_path
        self.lmdbinter_path   = lmdbinter_path
        self.lmdbrir_path     = lmdbrir_path
        self.lmdbmusic_path   = lmdbmusic_path

        self.train_ratio_uni     = train_ratio_uni
        self.train_ratio_unipb   = train_ratio_unipb
        self.train_ratio_inter   = train_ratio_inter
        self.train_padhead_frame = train_padhead_frame
        self.train_padtail_frame = train_padtail_frame

        self.lmdb_uni     = None
        self.lmdb_unipb   = None
        self.lmdb_noise   = None
        self.lmdb_inter   = None
        self.lmdb_interpb = None
        self.lmdb_rir     = None
        self.lmdb_music   = None

        self.rank              = rank
        self.world_size        = world_size
        self.start_line        = start_line
        self.end_line          = end_line
        self.max_sent_frame    = max_sent_frame
        self.min_sent_frame    = min_sent_frame

        self.datanoise_keys    = []
        self.datainter_keys    = []
        self.datarir_keys      = []
        self.datamusic_keys    = []

        self.data_uni_keys_lens   = []
        self.data_unipb_keys_lens = []

        # 通用回放数据
        if self.lmdbunipb_path != None:
            self.data_unipb_keys_lens = self.__genUniKeyList_Lmdb__(self.lmdbunipb_path) # 通用回放数据按GPU卡号分配

        # 通用仿真+回放数据 List
        if self.lmdbuni_path != None:
            self.data_uni_keys_lens = self.__genUniKeyList_Lmdb__(self.lmdbuni_path) # 通用仿真数据按GPU卡号分配

            if self.lmdbnoise_path != None:
                self.datanoise_keys, _ = self.__genKeyList_Lmdb__(self.lmdbnoise_path, len_limit=False) # 噪声数据按GPU卡号分配

            # 多通道数据
            if self.lmdbinter_path != None:
                self.datainter_keys, _ = self.__genKeyList_Lmdb__(self.lmdbinter_path, len_limit=False) # 数据按GPU卡号分配                
            if self.lmdbrir_path != None:
                self.datarir_keys, _   = self.__genKeyList_Lmdb__(self.lmdbrir_path, len_limit=False) # 数据按GPU卡号分配
            if self.lmdbmusic_path != None:
                self.datamusic_keys, _ = self.__genKeyList_Lmdb__(self.lmdbmusic_path, len_limit=False) # 数据按GPU卡号分配

    def __len__(self):
        return len(self.data_unipb_keys_lens)

    def __genUniKeyList_Lmdb__(self, lmdb_path, len_limit=True):
        '''
        与__genKeyList_Lmdb__的区别在 记录data_len用于估计迭代次数
        '''
        data_keys_lens = []
        for i, line in enumerate(open(get_lmdb_key(lmdb_path))):
            line = line.strip()
            if line == "":
                continue
            items = line.split()
            if (i - self.rank) % self.world_size == 0:
                sent_frame = int(items[1])
                if len_limit:
                    if sent_frame <= self.max_sent_frame and sent_frame >= self.min_sent_frame:
                        data_keys_lens.append([i, sent_frame])
                else:
                    data_keys_lens.append([i, sent_frame])
        return data_keys_lens

    def __genKeyList_Lmdb__(self, lmdb_path, len_limit=True):
        index_tmp = 0
        data_keys_out   = []
        data_indexs_out = []
        for i, line in enumerate( open(get_lmdb_key(lmdb_path) ) ):
            line = line.strip()
            if line=="":
                continue
            items = line.split()
            if (i - self.rank) % self.world_size == 0:
                sent_frame = int(items[1])
                if len_limit:
                    if sent_frame <= self.max_sent_frame and sent_frame >= self.min_sent_frame:
                        data_keys_out.append(int(items[0]))
                        data_indexs_out.append(index_tmp)
                        index_tmp += 1
                else:
                    data_keys_out.append(int(items[0]))
                    data_indexs_out.append(index_tmp)
                    index_tmp += 1
        return data_keys_out, data_indexs_out

    def __readLmdb__(self, lmdbname, data_index, index_list, pad_head=0, pad_tail=0, randpad=False, max_len=100000):
        data_out  = []
        label_out = []
        syllable_out = []

        if randpad and pad_head>=30:
            pad_head = self.rng.integers(30, pad_head)

        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(data_index[index]).zfill(12).encode('utf-8')
                cursor.set_key(k)
                datum = SpeechDatum()
                datum.ParseFromString(cursor.value())
                # data = np.frombuffer(datum.anc.data, dtype=np.int16)
                # label = np.frombuffer(datum.anc.state_data, dtype=np.int16).reshape(-1, 1)
                data  = np.array(np.fromstring(datum.anc.data, dtype=np.float32), dtype="int16")
                label = np.fromstring(datum.anc.state_data, dtype=np.int32).reshape(-1, 1)
                syllable = np.fromstring(datum.anc.state_data, dtype=np.int32).reshape(-1, 1)

                wave_len = np.shape(data)[0]
                frames_len = np.shape(label)[0]
                if frames_len > max_len:
                    cut_bg = self.rng.integers(0, frames_len-max_len)
                    data = data[cut_bg*160 : (cut_bg + max_len)*160]
                    label = label [cut_bg : (cut_bg + max_len), :]
                    wave_len = np.shape(data)[0]
                    frames_len = np.shape(label)[0]

                if pad_head>0 or pad_tail>0:
                    index_bg = pad_head*160
                    index_ed = pad_tail*160

                    data_cat = np.zeros( (wave_len+index_bg+index_ed), dtype="int16" )
                    data_cat[index_bg: wave_len+index_bg] = data

                    # label_cat = np.ones( (frames_len+pad_head+pad_tail, 1), dtype="int16")*(-2)
                    label_cat = np.ones( (frames_len+pad_head+pad_tail, 1), dtype="int32")*(-2)
                    label_cat[pad_head: frames_len+pad_head, :] = label
                    syllable_cat = np.ones( (frames_len+pad_head+pad_tail, 1), dtype="int32")*(-2)
                    syllable_cat[pad_head: frames_len+pad_head, :] = syllable
                else:
                    data_cat = data
                    label_cat = label
                    syllable_cat = syllable

                data_out.append(data_cat)
                label_out.append(label_cat)
                syllable_out.append(syllable_cat)
                
        return data_out, label_out, syllable_out

    def __readLmdbInt16__(self, lmdbname, index_list, pad_head=0, pad_tail=0, randpad=False, max_len=100000):
        data_out  = []
        label_out = []

        if randpad and pad_head>=30:
            pad_head = self.rng.integers(30, pad_head)

        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(index).zfill(12).encode('utf-8')
                cursor.set_key(k)
                datum = SpeechDatum()
                datum.ParseFromString(cursor.value())
                data = np.frombuffer(datum.anc.data, dtype=np.int16)
                label = np.frombuffer(datum.anc.state_data, dtype=np.int16).reshape(-1, 1)
                
                wave_len = np.shape(data)[0]
                frames_len = np.shape(label)[0]
                if frames_len > max_len:
                    cut_bg = self.rng.integers(0, frames_len-max_len)
                    data   = data[cut_bg*160 : (cut_bg + max_len)*160]
                    label  = label [cut_bg : (cut_bg + max_len), :]
                    wave_len   = np.shape(data)[0]
                    frames_len = np.shape(label)[0]

                if pad_head>0 or pad_tail>0:
                    index_bg = pad_head*160
                    index_ed = pad_tail*160

                    data_cat = np.zeros( (wave_len+index_bg+index_ed), dtype="int16" )
                    data_cat[index_bg: wave_len+index_bg] = data

                    label_cat = np.ones( (frames_len+pad_head+pad_tail, 1), dtype="int16")*(-2)
                    label_cat[pad_head: frames_len+pad_head, :] = label
                else:
                    data_cat  = data
                    label_cat = label

                data_out.append(data_cat)
                label_out.append(label_cat)
        return data_out, label_out

    def __readLmdb_czzhu2_Int16__(self, lmdbname, index, pad_tail=16, max_len=100000):
        data_out     = []
        label_out    = []
        syllable_out = []
        syllable_out = []
        wordid_out   = []

        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            k = str(index).zfill(12).encode('utf-8')
            cursor.set_key(k)
            datum    = Datum2()
            datum.ParseFromString(cursor.value())
            data     = np.frombuffer(datum.anc.data, dtype=np.int16)
            data_sum = np.sum(data)
            if data_sum == 0:
                return data_out, label_out, syllable_out, wordid_out

            label    = np.frombuffer(datum.anc.state_data, dtype=np.int16).reshape(-1, 1)
            syllable = np.fromstring(datum.anc.syllable_data, dtype=np.int16).reshape(-1, 1)

            rnnt_lab = syllable[syllable>=0]
            rnnt_lab[rnnt_lab > 1660] = rnnt_lab[rnnt_lab > 1660] - 6739
            res      = label.shape[0] - rnnt_lab.shape[0] -2
            rnnt_lab = np.concatenate(([voc_size-2], rnnt_lab, [voc_size-1], [-1]*res))
            syllable = rnnt_lab.reshape(-1, 1)

            wav_name = datum.anc.wave_name.decode()
            wordid   = wav_name.split('_')[-1]
            if wordid == 'CN':
                wordid = 0
            elif wordid == 'EN':
                wordid = 1
            elif wordid == 'CAN':
                wordid = 2
            elif wordid == 'CNEN':
                wordid = 3
            elif wordid == 'SC':
                wordid = 4
            else:
                wordid = -1

            wave_len   = np.shape(data)[0]
            frames_len = np.shape(label)[0]
            if frames_len > max_len:
                cut_bg = self.rng.integers(0, frames_len-max_len)
                data   = data[cut_bg*160 : (cut_bg + max_len)*160]
                label  = label [cut_bg : (cut_bg + max_len), :]
                syllable   = syllable [cut_bg : (cut_bg + max_len), :]
                wave_len   = np.shape(data)[0]
                frames_len = np.shape(label)[0]

            frames_pad = frames_len if frames_len % pad_tail == 0 else frames_len + pad_tail - frames_len % pad_tail
            pad_tail_t = frames_pad - frames_len
            if pad_tail_t>0:
                index_ed = pad_tail_t*160

                data_cat = np.zeros( (wave_len+index_ed), dtype="int16" )
                data_cat[: wave_len] = data

                label_cat = np.ones( (frames_len+pad_tail_t, 1), dtype="int16")*(-2)
                label_cat[: frames_len, :] = label

                syllabel_cat = np.ones( (frames_len+pad_tail_t, 1), dtype="int16")*(-2)
                syllabel_cat[: frames_len, :] = syllable
            else:
                data_cat  = data
                label_cat = label
                syllabel_cat = syllable

            data_out.append(data_cat)
            label_out.append(label_cat)
            syllable_out.append(syllabel_cat)
            wordid_out.append(wordid)
        return data_out, label_out, syllable_out, wordid_out

    def __readLmdbFloat__(self, lmdbname, data_index, index_list, pad_head=0, pad_tail=0, randpad=False, max_len=1000000):
        data_out  = []
        label_out = []

        if randpad and pad_head>=30:
            pad_head = self.rng.integers(30, pad_head)

        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(data_index[index]).zfill(12).encode('utf-8')
                cursor.set_key(k)
                datum = SpeechDatum()
                datum.ParseFromString(cursor.value())
                data = np.array(np.fromstring(datum.anc.data, dtype=np.float32), dtype="int16")
                label = np.array(np.fromstring(datum.anc.state_data, dtype=np.int32), dtype="int16").reshape(-1, 1)

                wave_len = np.shape(data)[0]
                frames_len = np.shape(label)[0]
                if frames_len > max_len:
                    cut_bg = self.rng.integers(0, frames_len-max_len)
                    data = data[cut_bg*160 : (cut_bg + max_len)*160]
                    label = label [cut_bg : (cut_bg + max_len), :]
                    wave_len = np.shape(data)[0]
                    frames_len = np.shape(label)[0]

                if pad_head>0 or pad_tail>0:
                    index_bg = pad_head*160
                    index_ed = pad_tail*160

                    data_cat = np.zeros( (wave_len+index_bg+index_ed), dtype="int16" )
                    data_cat[index_bg: wave_len+index_bg] = data

                    label_cat = np.ones( (frames_len+pad_head+pad_tail, 1), dtype="int16")*(-2)
                    label_cat[pad_head: frames_len+pad_head, :] = label
                else:
                    data_cat = data
                    label_cat = label

                data_out.append(data_cat)
                label_out.append(label_cat)
        return data_out, label_out

    def __readLmdbMC__(self, lmdbname, index_list, v_dim, h_dim):
        data_out  = []
        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(index).zfill(12).encode('utf-8')
                cursor.set_key(k)
                datum = SpeechDatum()
                datum.ParseFromString(cursor.value())
                data = np.frombuffer(datum.anc.data, dtype = np.int16).reshape(v_dim, h_dim)
                data_out.append(data)
        return data_out

    def __readLmdbDiffunoise__(self, lmdbname, index_list, v_dim, h_dim):
        data_out  = []
        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(index).zfill(12).encode('utf-8')
                cursor.set_key(k)
                datum = Datum1()
                datum.ParseFromString(cursor.value())
                data = np.frombuffer(datum.anc.data, dtype = np.int16).reshape(v_dim, h_dim)
                data_out.append(data)
        return data_out

    def __readLmdbRIR__(self, lmdbname, index_list, v_dim, h_dim):
        data_out1  = []
        data_out2  = []
        data_out3  = []
        txn = lmdbname.begin()
        with txn.cursor() as cursor:
            for index in index_list:
                k = str(index).zfill(12).encode('utf-8')
                cursor.set_key(k)
                datum = Datum1()
                datum.ParseFromString(cursor.value())
                data1 = np.frombuffer(datum.anc.data, dtype = np.int16).reshape(v_dim, -1, h_dim)
                data2 = np.frombuffer(datum.anc.state_data, dtype = np.float32).reshape(v_dim, h_dim)
                rir_name = datum.anc.wave_name.decode()
                data_out1.append(data1)
                data_out2.append(data2)
                data_out3.append(rir_name)
        data_out = [data_out1, data_out2, data_out3]
        return data_out

    def __maskBoundInfo__(self, label):
        label_uniq = []
        count_uniq = []
        node_th    = None
        for key, value in groupby(list(label)):
            label_uniq.append(key)
            count_uniq.append(len(list(value)))
        if self.is_ivw:
            node_th = noneid_ivw-4
            label_uniq_filter = list(filter(lambda x:x<node_th and x>=0, label_uniq))
            mask_num          = self.rng.integers(9, 18)
        else:
            node_th = noneid_csk-1
            label_uniq_filter = list(filter(lambda x:x<node_th and x>=0, label_uniq))
            mask_num          = self.rng.integers(2, 5)
        mask_start  = self.rng.integers(0, len(label_uniq_filter)-mask_num)
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
        bound_del = self.rng.integers(bound_start, bound_end)
        return bound_start, bound_end, bound_del

    def __padInter__(self, data_list):
        max_value = 0.0
        dataList_out = []
        wave_len = 0
        for index, data_array in enumerate(data_list):
            data_array = np.array( data_array, dtype=np.float32)
            data_sort = np.unique(data_array)
            if len(data_sort) <5:
                # data_array=data_array*0  ###############################################################  注意
                # amp_max = 1
                continue
            else:
                amp_max = data_sort[-3]
            if amp_max <= 100:
                continue
            if amp_max > max_value:
                max_value = amp_max
            wave_len += np.shape(data_array)[0]
            dataList_out.append( data_array / float(amp_max+1e-10) )

        pad_mid = list(self.rng.choice(list(range(5, 10)),len(dataList_out)-1, replace=False))
        pad_mid.insert(0, self.train_padhead_frame)
        pad_mid.append(self.train_padtail_frame)

        for pad_len in pad_mid:
            wave_len += pad_len*160
        data_index = list(range(len(dataList_out)))
        self.rng.shuffle(data_index)
        self_train_padhead_frame = 0
        data_cat = np.zeros( (self_train_padhead_frame*160+wave_len), dtype="int16")
        wave_bg = self_train_padhead_frame*160
        frames_bg = self_train_padhead_frame
        # max_value = max_value/1.5
        max_value = max_value/3
        for i, index in enumerate(data_index):
            temp_waveLen = np.shape( dataList_out[index] )[0]
            data_rescale = np.array( dataList_out[index] * max_value, dtype=np.int16 )
            data_cat[wave_bg : wave_bg+temp_waveLen] = data_rescale
            wave_bg = wave_bg + temp_waveLen + pad_mid[i]*160
        return [data_cat]

    def __getitem__(self, index):
        # 仿真数据
        data_out           = []
        label_out          = []
        syllable_out       = []
        wordid_out         = []

        # 回放数据
        datapb_out         = []
        labelpb_out        = []
        syllablepb_out     = []
        wordidpb_out       = []

        # 噪声数据
        datanoise_out      = []

        # 多通道数据
        datainter_out      = []
        labelinter_out     = []
        syllableinter_out  = []
        wordidinter_out    = []
        datarir_out        = []
        datamusic_out      = []

        index_type, index_id = index.split('_')
        index_id = int(index_id)
        # 通用回放数据
        if self.lmdbunipb_path != None and index_type == '0':
            data_unipb, label_unipb, syllable_unipb, wordidpb = self.__readLmdb_czzhu2_Int16__(self.lmdb_unipb, index_id, self.train_padtail_frame)
            datapb_out   += data_unipb
            labelpb_out  += label_unipb
            syllablepb_out += syllable_unipb
            wordidpb_out += wordidpb
        assert(len(datapb_out)==len(labelpb_out)==len(wordidpb_out))

        # 通用仿真主目标
        if self.lmdbuni_path != None and index_type == '1':
            data_uni, label_uni, syllable_uni, wordid_uni = self.__readLmdb_czzhu2_Int16__(self.lmdb_uni, index_id, self.train_padtail_frame)
            data_out   += data_uni
            label_out  += label_uni
            syllable_out += syllable_uni
            wordid_out += wordid_uni

            assert(len(data_out)==len(label_out)==len(wordid_out))

            # 通用仿真干扰噪声
            if self.lmdbinter_path != None:
                rand_idx = [self.rng.integers(0, len(self.datainter_keys)) for i in range(self.train_ratio_inter)]
                sample_inter_idx = [ self.datainter_keys[i] for i in rand_idx]
                data_inter, label_inter = self.__readLmdbInt16__(self.lmdb_inter, sample_inter_idx)
                datainter_out  += data_inter
                labelinter_out += label_inter
                wordidinter_out.extend( [0 for i in range(self.train_ratio_inter)] )

            if len(datainter_out) > 1:  ## 干扰噪声拼接
                datainter_out = self.__padInter__(datainter_out)

            assert(len(data_out)==len(label_out)==len(wordid_out))

            ### 多通道采样：
            sample_len = math.ceil(len(data_out)/float(sent_padcount))

            train_ratio_noise = 1
            # 噪声数据
            if self.lmdbnoise_path != None:
                rand_idx   = [self.rng.integers(0, len(self.datanoise_keys)) for i in range(train_ratio_noise)]
                sample_nose_idx = [ self.datanoise_keys[i] for i in rand_idx]
                data_noise = self.__readLmdbDiffunoise__(self.lmdb_noise, sample_nose_idx, 2, -1)
                datanoise_out += data_noise

            # rir数据采样
            if self.lmdbrir_path != None:
                rand_idx  = [self.rng.integers(0, len(self.datarir_keys)) for i in range(sample_len)]
                sample_rir_idx = [ self.datarir_keys[i] for i in rand_idx]
                tmpch_rir = self.__readLmdbRIR__(self.lmdb_rir, sample_rir_idx, 2, 3)
                datarir_out += tmpch_rir

            # music数据采样
            if self.lmdbmusic_path != None:
                rand_idx = [self.rng.integers(0, len(self.datamusic_keys)) for i in range(sample_len)]
                sample_music_idx = [ self.datamusic_keys[i] for i in rand_idx]
                tmpmusic_out = self.__readLmdbMC__(self.lmdb_music, sample_music_idx, 1, -1)
                datamusic_out += tmpmusic_out

            # 回放点噪声源采样
            if self.lmdbinter_path != None:
                rand_idx = [self.rng.integers(0, len(self.datainter_keys)) for i in range(sample_len)]
                sample_inter_idx = [ self.datainter_keys[i] for i in rand_idx]
                tmpinter_out = self.__readLmdbMC__(self.lmdb_inter, sample_inter_idx, 2, -1)
                datainter_out += tmpinter_out

        return data_out, label_out, syllable_out, wordid_out, \
            datapb_out, labelpb_out, syllablepb_out, wordidpb_out, \
            datanoise_out, datainter_out, labelinter_out, syllableinter_out, wordidinter_out, \
            datarir_out, datamusic_out, index

class BunchSampler(Sampler):
    def __init__(self, dataset_lengths: Sequence[int], dataset_lengths_clean: Sequence[int], batch_size: int, bunch_size: int, bunch_size_clean: int, drop_last: bool, shuffle_batch: bool = True, iter_num=None, seed=None, is_sort=False) -> None:
        self.lengths    = dataset_lengths
        self.lengths_clean = dataset_lengths_clean
        self.batch_size = batch_size
        self.bunch_size = bunch_size
        self.bunch_size_clean = bunch_size_clean
        self.drop_last  = drop_last
        self.shuffle_batch = shuffle_batch
        self.is_sort   = is_sort
        self.generator = torch.Generator()
        self.rng       = np.random.default_rng()
        if seed is not None:
            self.generator.manual_seed(seed)
            self.rng   = np.random.default_rng(seed*2)
        
        # 排序
        if is_sort:
            dict_length_indices = dict()
            for i, item in enumerate(self.lengths):
                key_idx, length = item
                if length == 0:
                    continue
                if length in dict_length_indices:
                    dict_length_indices[length].append(key_idx)
                else:
                    dict_length_indices[length] = [key_idx]
            self.length_indices_list = sorted(dict_length_indices.items(), key=lambda x:x[0])

            dict_length_indices_clean = dict()
            for i, item in enumerate(self.lengths_clean):
                key_idx, length = item
                if length == 0:
                    continue
                if length in dict_length_indices_clean:
                    dict_length_indices_clean[length].append(key_idx)
                else:
                    dict_length_indices_clean[length] = [key_idx]
            self.length_indices_list_clean = sorted(dict_length_indices_clean.items(), key=lambda x:x[0])

        else:
            self.length_indices_list = []
            for i, item in enumerate(self.lengths):
                key_idx, length = item
                if length == 0:
                    continue
                self.length_indices_list.append(tuple((length, key_idx)))
            if self.shuffle_batch:
                self.rng.shuffle(self.length_indices_list)

        if iter_num is None:
            batch_sequence, _ = self._prefetch_batch(static_batch_num=True)
            self.iter_num  = len(batch_sequence)
        else:
            self.iter_num  = iter_num

    def __iter__(self):
        batch_sequence, batch_sequence_clean = self._prefetch_batch()
        if self.shuffle_batch:
            indices = torch.randperm(len(batch_sequence), generator=self.generator).tolist()
        else:
            indices = list(range(len(batch_sequence)))
        if len(indices) > self.iter_num:
            indices = indices[:self.iter_num]
        while len(indices) < self.iter_num:
            indices = indices + indices[:self.iter_num-len(indices)]

        if self.shuffle_batch:
            indices_clean = torch.randperm(len(batch_sequence_clean), generator=self.generator).tolist()
        else:
            indices_clean = list(range(len(batch_sequence_clean)))
        indices_num_max   = len(batch_sequence)
        if len(indices_clean) > indices_num_max:
            indices_clean = indices_clean[:indices_num_max]
        while len(indices_clean) < indices_num_max:
            indices_clean = indices_clean + indices_clean[:indices_num_max-len(indices_clean)]
        
        for index in indices:
            out_batch_sequence = []
            for item in batch_sequence[index]:
                out_batch_sequence.append('0_%d'%item)
            for item in batch_sequence_clean[indices_clean[index]]:
                out_batch_sequence.append('1_%d'%item)
            yield out_batch_sequence

    def __len__(self):
        return self.iter_num

    def _prefetch_batch(self, static_batch_num=False):
        batch_sequence = []
        batch_sequence_clean = []
        batch          = []
        batch_clean    = []
        total_length   = 0
        total_length_clean = 0

        # 排序
        if self.is_sort:
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
            
            for length, indices in self.length_indices_list_clean:
                if length > self.bunch_size_clean:
                    continue
                if self.shuffle_batch and not static_batch_num:
                    self.rng.shuffle(indices)
                for index in indices:
                    batch_clean.append(index)
                    total_length_clean += length
                    if total_length_clean + length > self.bunch_size_clean or len(batch_clean) == self.batch_size:
                        batch_sequence_clean.append(batch_clean)
                        batch_clean = []
                        total_length_clean = 0
        else:
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
        if len(batch_clean) > 0 and not self.drop_last:
            batch_sequence_clean.append(batch_clean)
        return batch_sequence, batch_sequence_clean

class Collater_online():
    def __init__(self, nmod_pad, lmdb_reverbfile, lmdb_normfile, train_padhead_frame, train_padtail_frame, train_padmid_frame, seed=None, is_pad=False):
        # 随机种子
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed*3)

        # 初始化
        self.nmod_pad            = nmod_pad
        self.lmdb_reverbfile     = lmdb_reverbfile
        self.lmdb_normfile       = lmdb_normfile
        self.train_padhead_frame = train_padhead_frame
        self.train_padtail_frame = train_padtail_frame
        self.train_padmid_frame  = train_padmid_frame

        # 获取fea norm
        with open(self.lmdb_normfile, 'r') as nf:
            normfile_lines = nf.read().splitlines()
        self.mel_spec_mean = np.array([float(item) for item in normfile_lines[1:41]])
        self.mel_spec_std  = np.array([float(item) for item in normfile_lines[42:82]])
        self.padzeros_mch  = np.zeros((1, 40))

        self.doa   = [42.5, 90, 137.5]
        self.doa_constrained = [0, 85, 85, 95, 95, 180]
        self.ChiefBeam = 0
        self.NFFT  = 512
        self.shift = 256

        # 仿真参数配置
        self.noise_snr = [3,6,9,12,15,20] # 散噪信噪比 #[8,11,13,14,17,20]
        self.ddr_snr   = [-3,0,3,6,10,20] # 点噪信噪比 #[-10,0,10,20,50,80] 
        self.inter_snr = [0,1,3,6,9,12] #[5,6,7,10,12,15] [-6,-3,0,3,6,10]
        self.ii_snr    = [-2,-1,0,1,2,3]
        self.rng.shuffle(self.noise_snr)
        self.rng.shuffle(self.ddr_snr)
        self.rng.shuffle(self.inter_snr)
        self.rng.shuffle(self.ii_snr)
        self.amp_list  = list(range(2000, 26000))
        self.rng.shuffle(self.amp_list)
        self.count     = 0
        self.is_pad    = is_pad

    def __padData__(self, data_in, label_in, syllable_in, wordid_in):
        assert(len(data_in)==len(label_in)==len(wordid_in))

        zip_list = list( zip(data_in, label_in, syllable_in, wordid_in) )
        self.rng.shuffle(zip_list)
        data_in, label_in, syllable_in, wordid_in = zip(*zip_list) # 多子句shuffle顺序
        data_out      = []
        label_out     = []
        syllable_out  = []
        wordid_out    = []
        sent1st_out   = []
        sented_out    = [] #起止点

        total_len = len(data_in)
        pointer_bg = 0
        pointer_ed = sent_padcount # 每n句话拼一句
        while pointer_bg < total_len:
            tmp_data     = data_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_label    = label_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_syllable = syllable_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_wordid   = wordid_in[pointer_bg : min(pointer_ed, total_len)]
            tmp_wented   = []

            # 更新音频拼接选取边界
            pointer_bg = pointer_ed
            pointer_ed += 3

            if len(tmp_data) > 1:
                max_value = 0.0
                tmp_data_out     = []
                tmp_label_out    = []
                tmp_syllable_out = []
                tmp_wordid_out   = []
                wave_len   = 0
                frames_len = 0
                for index, data_array in enumerate(tmp_data):
                    data_array = np.array( data_array, dtype=np.float32 )
                    data_sort  = np.unique(data_array)
                    if len(data_sort) <5:
                        continue
                    amp_max    = data_sort[-3]
                    if amp_max <= 100:
                        continue
                    if amp_max > max_value:
                        max_value = amp_max
                    wave_len   += np.shape(data_array)[0]
                    frames_len += np.shape(tmp_label[index])[0]

                    # 音频幅值归一
                    tmp_data_out.append( data_array / float(amp_max) )
                    tmp_label_out.append( tmp_label[index] )
                    tmp_syllable_out.append( tmp_syllable[index] )
                    tmp_wordid_out.append( tmp_wordid[index] )

                    # 多通道仿真记录第一句话的长度
                    if len(tmp_data_out) == 1:
                        sent1st_out.append(self.train_padhead_frame*160 + wave_len)

                # 拼接sil采样
                assert(self.train_padhead_frame >= 8)
                wave_count = len(tmp_data_out)
                # pad_mid    = list(self.rng.choice(list(range(8, self.train_padhead_frame)), wave_count-1, replace=False))
                pad_mid    = [self.train_padmid_frame for i in range(wave_count-1)]
                pad_mid.insert(0, self.train_padhead_frame)
                # pad_mid.append(self.train_padtail_frame)
                for pad_len in pad_mid:
                    wave_len   += pad_len*160
                    frames_len += pad_len
                data_indexs = list(range(wave_count))
                self.rng.shuffle(data_indexs)

                # 初始化音频和标注
                data_cat   = np.zeros( (wave_len), dtype=np.int16)
                label_cat  = np.ones( (frames_len, 1), dtype=np.int16 )*(-2)
                syllable_cat = np.copy(label_cat)
                wordid_cat = np.copy(label_cat)

                wave_bg   = 0
                frames_bg = 0
                max_value = max_value/4
                for i, index in enumerate(data_indexs):
                    if i == 0:
                        label_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :]    = -3
                        syllable_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :] = -3
                        wordid_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :]   = -3
                    else:
                        label_cat[ frames_bg: frames_bg+pad_mid[i], :]    = -3
                        syllable_cat[ frames_bg: frames_bg+pad_mid[i], :] = -3
                        wordid_cat[ frames_bg: frames_bg+pad_mid[i], :]   = -3

                    wave_bg += pad_mid[i]*160
                    frames_bg += pad_mid[i]

                    # 数据拼接
                    temp_waveLen = np.shape( tmp_data_out[index] )[0]
                    data_rescale = np.array( tmp_data_out[index] * max_value, dtype=np.int16 )
                    data_cat[wave_bg : wave_bg+temp_waveLen] = data_rescale

                    temp_framesLen = np.shape( tmp_label_out[index] )[0]
                    label_cat[frames_bg : frames_bg+temp_framesLen, :]    = tmp_label_out[index]
                    syllable_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_syllable_out[index]
                    wordid_cat[frames_bg : frames_bg+temp_framesLen, :]   = tmp_wordid_out[index]

                    tmp_wented.append([frames_bg, 0])
                    wave_bg   += temp_waveLen
                    frames_bg += temp_framesLen
                    tmp_wented[i][1] = frames_bg
            else:
                # 拼接sil采样
                assert(self.train_padhead_frame >= 8)
                pad_mid = []
                pad_mid.append(self.train_padhead_frame)

                # 多通道仿真记录第一句话的长度
                sent1st_out.append(pad_mid[0]*160 + np.shape( tmp_data[0] )[0])

                # 初始化音频和标注
                wave_len   = pad_mid[0]*160 + np.shape( tmp_data[0] )[0] + self.train_padtail_frame*160
                frames_len = pad_mid[0] + np.shape( tmp_label[0] )[0] + self.train_padtail_frame
                data_cat   = np.zeros( (wave_len), dtype=np.int16)
                label_cat  = np.ones( (frames_len, 1), dtype=np.int16 )*(-2)
                syllable_cat = np.copy(label_cat)
                wordid_cat = np.copy(label_cat)

                wave_bg   = 0
                frames_bg = 0
                if pad_mid[0] > 16:
                    label_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :]    = -3
                    syllable_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :] = -3
                    wordid_cat[ min(240, self.train_padhead_frame) : self.train_padhead_frame, :]   = -3
                else:
                    label_cat[ frames_bg: frames_bg+pad_mid[0], :]    = -3
                    syllable_cat[ frames_bg: frames_bg+pad_mid[0], :] = -3
                    wordid_cat[ frames_bg: frames_bg+pad_mid[0], :]   = -3

                wave_bg   += pad_mid[0]*160
                frames_bg += pad_mid[0]

                # 数据拼接
                temp_waveLen = np.shape( tmp_data[0] )[0]
                data_cat[wave_bg : wave_bg+temp_waveLen] = tmp_data[0]

                temp_framesLen = np.shape( tmp_label[0] )[0]
                label_cat[frames_bg : frames_bg+temp_framesLen, :]    = tmp_label[0]
                syllable_cat[frames_bg : frames_bg+temp_framesLen, :] = tmp_syllable[0]
                wordid_cat[frames_bg : frames_bg+temp_framesLen, :]   = tmp_wordid[0]

                tmp_wented.append([frames_bg, 0])
                wave_bg   += temp_waveLen
                frames_bg += temp_framesLen
                tmp_wented[0][1] = frames_bg

            assert(np.shape(label_cat)==np.shape(syllable_cat)==np.shape(wordid_cat))
            data_out.append(data_cat)
            label_out.append(label_cat)
            syllable_out.append(syllable_cat)
            wordid_out.append(wordid_cat)
            sented_out.append(tmp_wented)

        return data_out, label_out, syllable_out, wordid_out, sent1st_out, sented_out

    def __ampChangeSingleCh__(self, data_in, label_in, syllable_in, wordid_in, sequenceLen, ampList):
        data_out = []
        label_out = []
        syllable_out = []
        wordid_out = []
        sequence_out = []
        for i, amp in enumerate(ampList):
            data_tmp = data_in[i]
            lab_tmp  = label_in[i]
            syll_tmp = syllable_in[i]
            word_tmp = wordid_in[i]
            seq_tmp  = sequenceLen[i]
            if self.is_pad:
                wave_len   = data_tmp.shape[0]
                frames_len = wave_len//160
                pad_tail_t = self.train_padmid_frame
                index_ed   = pad_tail_t*160

                data_cat   = np.zeros( (wave_len+index_ed), dtype="int16" )
                data_cat[: wave_len] = data_tmp
                data_tmp   = data_cat

                label_cat  = np.ones( (frames_len+pad_tail_t, 1), dtype="int16")*(-3)
                label_cat[: frames_len, :] = lab_tmp
                lab_tmp    = label_cat

                syllabel_cat = np.ones( (frames_len+pad_tail_t, 1), dtype="int16")*(-3)
                syllabel_cat[: frames_len, :] = syll_tmp
                syll_tmp   = syllabel_cat

                word_cat   = np.ones( (frames_len+pad_tail_t, 1), dtype="int16")*(-3)
                word_cat[: frames_len, :] = word_tmp
                word_tmp   = word_cat

                seq_tmp    += index_ed
                
            temp = np.array(data_tmp, dtype=np.float32)
            amp_maxList = list(np.unique(temp))
            if len(amp_maxList) <5:
                continue
            amp_max = amp_maxList[-3]
            if amp_max <= 100:
                continue
            data_out.append(np.array(temp/amp_max*amp, dtype=np.int16))
            label_out.append(lab_tmp)
            syllable_out.append(syll_tmp)
            wordid_out.append(word_tmp)
            sequence_out.append(seq_tmp)
        return data_out, label_out, syllable_out, wordid_out, sequence_out

    def __ampChangeMultiCh__(self, data_in1, data_in2, n_in, label_in, syllable_in, wordid_in, sequenceLen, ampList):
        data_out1 = []
        data_out2 = []
        data_out3 = []
        label_out = []
        syllable_out = []
        wordid_out = []
        sequence_out = []
        for i, amp in enumerate(ampList):
            temp = np.array(data_in1[i], dtype=np.float32)
            temp_gsc = np.array(data_in2[i], dtype=np.float32)
            temp_last = np.concatenate((temp, temp_gsc),axis=0)

            temp1 = np.array(n_in[i], dtype=np.float32)
            amp_maxList = list(np.unique(temp_last))
            if len(amp_maxList) <5:
                continue
            amp_max = amp_maxList[-3]
            if amp_max <= 100:
                pass # need fix czzhu2
                # continue
            data_out1.append(np.array(temp/amp_max*amp, dtype=np.int16))
            data_out2.append(np.array(temp_gsc/amp_max*amp, dtype=np.int16))
            data_out3.append(np.array(temp1/amp_max*amp, dtype=np.int16))

            label_out.append(label_in[i])
            syllable_out.append(syllable_in[i])
            wordid_out.append(wordid_in[i])
            sequence_out.append(sequenceLen[i])
        if len(data_out1) == 0:
            a=1
        return data_out1, data_out2, data_out3, label_out, syllable_out, wordid_out, sequence_out

    def __calFeanorm_pad0__(self, data_in1, data_in2, data_in3, label_in, syllable_in, wordid_in, datapb_in, labelpb_in, syllablepb_in, wordidpb_in):
        data_noisy_pb  = []
        labelpb_out    = []
        syllablepb_out = []
        wordidpb_out   = []

        data_noisy   = []
        data_clean   = []
        label_out    = []
        syllable_out = []
        wordid_out   = []
        sequence_out = []

        # 单通道回放数据
        if len(datapb_in) != 0:
            for key, value in enumerate(datapb_in):
                try:
                    nframes = min(value.shape[0], labelpb_in[key].shape[0], syllablepb_in[key].shape[0], np.array(wordidpb_in[key]).shape[0])
                except:
                    print ('err in playback audio')
                    continue
                if nframes <= 40:
                    print ('dropout wav which is too short')
                    continue
                padzeros_mch = np.tile(self.padzeros_mch, (nframes, 1))
                value = (value - self.mel_spec_mean[None,:]) * self.mel_spec_std
                value = np.concatenate( (value, padzeros_mch), axis=1)
                data_noisy_pb.append( value )
                labelpb_out.append(labelpb_in[key][:nframes, :])
                syllablepb_out.append(syllablepb_in[key][:nframes, :])
                wordidpb_out.append(wordidpb_in[key][:nframes, :])
                sequence_out.append(nframes)
        
        # 多通道仿真数据
        if len(data_in1) !=0:
            for key, value in enumerate(data_in1):
                min_len   = min(value.shape[0], data_in2[key].shape[0], data_in3[key].shape[0], label_in[key].shape[0], syllable_in[key].shape[0], wordid_in[key].shape[0])

                tmp_S     = (value[:min_len,:] - self.mel_spec_mean[None,:]) * self.mel_spec_std
                tmp_N     = (data_in3[key][:min_len,:] - self.mel_spec_mean[None,:]) * self.mel_spec_std
                tmp_C     = (data_in2[key][:min_len,:] - self.mel_spec_mean[None,:]) * self.mel_spec_std
                tmp_zeros = np.zeros((min_len, value.shape[1]))

                tmp_noise = np.concatenate( (tmp_S, tmp_N), axis=1)
                tmp_clean = np.concatenate( (tmp_C, tmp_zeros), axis=1)
                data_noisy.append( tmp_noise )
                data_clean.append( tmp_clean )
                label_out.append(label_in[key][:min_len, :])
                syllable_out.append(syllable_in[key][:min_len, :])
                wordid_out.append(wordid_in[key][:min_len, :])
                sequence_out.append(min_len)

        return data_noisy_pb, labelpb_out, syllablepb_out, wordidpb_out, data_noisy, data_clean, label_out, syllable_out, wordid_out, sequence_out

    def __calFeanorm_padN__(self, data_in1, data_in2, data_in3, label_in, syllable_in, wordid_in, datapb_in, labelpb_in, syllablepb_in, wordidpb_in):
        data_noisy    = []
        data_clean   = []
        label_out    = []
        syllable_out = []
        wordid_out   = []
        sequence_out = []

        # 单通道回放数据
        if len(datapb_in) != 0:
            for key, value in enumerate(datapb_in):
                try:
                    nframes = min(value.shape[0], labelpb_in[key].shape[0], syllablepb_in[key].shape[0], np.array(wordidpb_in[key]).shape[0])
                except:
                    print ('err in playback audio')
                    continue
                if nframes <= 40:
                    continue
                padzeros_mch = np.tile(self.padzeros_mch, (nframes, 1))
                value = (value - self.mel_spec_mean[None,:]) * self.mel_spec_std
                value = np.concatenate( (value, padzeros_mch), axis=1)
                data_noisy.append( value )
                data_clean.append( value )
                label_out.append(labelpb_in[key][:nframes, :])
                syllable_out.append(syllablepb_in[key][:nframes, :])
                wordid_out.append(np.array(wordidpb_in[key])[:nframes, :])
                sequence_out.append(nframes)
        
        # 多通道仿真数据
        if len(data_in1) !=0:
            for key, value in enumerate(data_in1):
                min_len   = min(value.shape[0], data_in2[key].shape[0], data_in3[key].shape[0], label_in[key].shape[0], syllable_in[key].shape[0], wordid_in[key].shape[0])

                tmp_S     = (value[:min_len,:] - self.mel_spec_mean[None,:]) * self.mel_spec_std
                tmp_N     = (data_in3[key][:min_len,:] - self.mel_spec_mean[None,:]) * self.mel_spec_std
                tmp_C     = (data_in2[key][:min_len,:] - self.mel_spec_mean[None,:]) * self.mel_spec_std
                tmp_zeros = np.zeros((min_len, value.shape[1]))

                tmp_noise = np.concatenate( (tmp_S, tmp_N), axis=1)
                tmp_clean = np.concatenate( (tmp_C, tmp_N), axis=1)
                data_noisy.append( tmp_noise )
                data_clean.append( tmp_clean )
                label_out.append(label_in[key][:min_len, :])
                syllable_out.append(syllable_in[key][:min_len, :])
                wordid_out.append(wordid_in[key][:min_len, :])
                sequence_out.append(min_len)

        return data_noisy, data_clean, label_out, syllable_out, wordid_out, sequence_out

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

    def __pad_nmod_alongT__(self, sequence, nmod_pad, val, length_del):
        padlen   = 0
        totallen = 0
        pad_list = []
        mask_list= []
        max_len  = 0
        bg_ed    = []
        for i, nparray in enumerate(sequence):
            if i == 0:
                nparray = nparray[length_del:, :]
            if max_len < nparray.shape[0]:
                max_len = nparray.shape[0]
            bg_ed.append([totallen])
            totallen   += nparray.shape[0]
            bg_ed[i].append(totallen)
            bg_ed[i].append(nparray.shape[0])
            nparray    = nparray.transpose(1, 0)
            torcharray = torch.from_numpy(nparray).float()
            torchmask  = torch.ones(1, torcharray.size()[1])
            torcharray = torcharray.reshape(1, 1, torcharray.size()[0], torcharray.size()[1])
            pad_list.append(torcharray)
            mask_list.append(torchmask)

        if nmod_pad is not None:
            padlen = totallen if totallen % nmod_pad == 0 else totallen + nmod_pad - totallen % nmod_pad
            if padlen > totallen:
                paddata = torch.ones(1, 1, pad_list[0].size()[2], padlen-totallen)*val
                padmask = torch.zeros(1, padlen-totallen)
                bg_ed[-1][1] += padlen-totallen
                pad_list.append(paddata)
                mask_list.append(padmask)
        batch_array = torch.cat(pad_list, dim=3)
        batch_mask  = torch.cat(mask_list, dim=1)
        bg_ed       = torch.Tensor(bg_ed)
        return batch_array, batch_mask, max_len, bg_ed
    
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

    def __get_sent_from_MC__(self, fbdata_out1, fbdata_out2, fbdata_out3, label_list, syllable_list, wordid_list, sented_out):
        fb_out1 = []
        fb_out2 = []
        fb_out3 = []
        label_out    = []
        syllable_out = []
        wordid_out   = []

        for idxi in range(len(fbdata_out1)):
            min_len = min(fbdata_out1[idxi].shape[0], fbdata_out2[idxi].shape[0], fbdata_out3[idxi].shape[0], label_list[idxi].shape[0], syllable_list[idxi].shape[0], wordid_list[idxi].shape[0])
            for idxj in range(len(sented_out[idxi])):
                s = sented_out[idxi][idxj][0]
                e = sented_out[idxi][idxj][1]
                if e > min_len:
                    e = min_len
                fb_out1.append( fbdata_out1[idxi][s:e] )
                fb_out2.append( fbdata_out2[idxi][s:e] )
                fb_out3.append( fbdata_out3[idxi][s:e] )
                label_out.append( label_list[idxi][s:e] )
                syllable_out.append( syllable_list[idxi][s:e] )
                wordid_out.append( wordid_list[idxi][s:e] )

        return fb_out1, fb_out2, fb_out3, label_out, syllable_out, wordid_out

    def build_fb40(self, data_list, num_thread=8):
        data_len = []
        data_out = []
        
        for item in data_list:
            data_len.append(item.shape[0]//160)
        data_list = delta.build_fb40(data_list, num_thread)
        if len(data_len) == 1:
            data_list = [data_list]
        for idx, item in enumerate(data_list):
            data_res = data_len[idx] - item.shape[0]
            if data_res >=1:
                padzeros_mch = np.tile(self.padzeros_mch, (data_res, 1))
                value = np.concatenate( (item, padzeros_mch), axis=0)
                data_out.append(value)
            elif data_res < 0:
                print ('error in build fb')
                data_out.append(item)
            else:
                data_out.append(item)
        return data_out

    def __call__(self, batch):
        # 仿真数据
        data_list        = []
        label_list       = []
        syllable_list    = []
        wordid_list      = []
        sequence_len     = []

        # 回放数据
        datapb_list      = []
        labelpb_list     = []
        syllablepb_list  = []
        wordidpb_list    = []
        sequencepb_len   = []

        # 多通道数据
        inter_list       = []
        music_list       = []
        noise_list       = []
        rir_list         = []
        rirname_list     = []
        sent1st_out      = []

        # 仿真输出数据
        fbdata_out1      = []
        fbdata_out2      = []
        fbdata_out3      = []

        # 多进程数据合并
        for data, label, syllable, wordid, datapb, labelpb, syllablepb, wordidpb, datanoise, datainter, labelinter, syllableinter, wordidinter, datarir, datamusic, idx_T in batch:
            if len(datapb) != 0 :
                datapb_list.extend(datapb)
                labelpb_list.extend(labelpb)
                syllablepb_list.extend(syllablepb)
                wordidpb_list.extend([np.ones((subdata.shape[0], 1))*wordidpb[idx] for idx, subdata in enumerate(labelpb)] )
                sequencepb_len.extend([np.shape(subdata)[0] for subdata in datapb])
            
            if len(data) != 0:
                data_list.extend(data)
                label_list.extend(label)
                syllable_list.extend(syllable)
                wordid_list.extend([np.ones((subdata.shape[0], 1))*wordid[idx] for idx, subdata in enumerate(label)] )
                sequence_len.extend([np.shape(subdata)[0] for subdata in data])

                noise_list.extend(datanoise)
                inter_list.extend(datainter)
                rir_list.extend(datarir[0]) # datarir: [[rir], [hinfo2], [rir_name]]
                rirname_list.extend(datarir[2])
                music_list.extend(datamusic)

        # 单通道回放数据幅值扰动
        if len(datapb_list) != 0 :
            amp_pb     = self.rng.choice(self.amp_list, len(datapb_list), replace=False)
            datapb_list, labelpb_list, syllablepb_list, wordidpb_list, sequencepb_len = self.__ampChangeSingleCh__(datapb_list, labelpb_list, syllablepb_list, wordidpb_list, sequencepb_len, amp_pb)
            datapb_len = len(datapb_list)
            if datapb_len > 0 :
                datapb_list = self.build_fb40(datapb_list, num_thread=8)
        
        # 待仿真数据拼接
        if len(data_list) != 0:
            down_audio(data_list, 'pad_before', 1)
            down_lab(label_list, 'pad_before_lab')
            down_lab(syllable_list, 'pad_before_sys')
            down_lab(wordid_list, 'pad_before_word')
            data_list, label_list, syllable_list, wordid_list, sent1st_out, sented_out = self.__padData__(data_list, label_list, syllable_list, wordid_list)
            down_audio(data_list, 'pad_after', 1)
            down_lab(label_list, 'pad_after_lab')
            down_lab(syllable_list, 'pad_after_sys')
            down_lab(wordid_list, 'pad_after_word')
            
            count_sent   = len(sent1st_out)
            noise_list   = noise_list[:count_sent]
            inter_list   = inter_list[:count_sent]
            rir_list     = rir_list[:count_sent]
            rirname_list = rirname_list[:count_sent]
            music_list   = music_list[:count_sent]

            # 数据仿真
            s_rir  = []
            i1_rir = []
            i2_rir = []

            MicDist_list = []
            single_noise_mic = []
            beam_pos     = []
            for key, value in enumerate(rirname_list):
                rirname  = value.split('@')
                MicDist  = float(rirname[0]) / 1000.0
                MicDist_list.append(MicDist)
                angle2   = float(rirname[2])
                if angle2 < 90.0: # 副后
                    s_rir.append(rir_list[key][:, :, 0].copy()) # 主驾
                    i1_rir.append(rir_list[key][:, :, 2].copy()) # 副驾
                    i2_rir.append(rir_list[key][:, :, 1].copy()) # 副后
                    beam_pos.append(1)
                else:             # 主后
                    s_rir.append(rir_list[key][:, :, 2].copy()) # 副驾
                    i1_rir.append(rir_list[key][:, :, 0].copy()) # 主驾
                    i2_rir.append(rir_list[key][:, :, 1].copy()) # 副后
                    beam_pos.append(0)
                # print('MicDist:', MicDist, 'angle2:', angle2, 'beam_pos:', beam_pos)

                single_noise_mic.append(np.squeeze(noise_list[key][0,:]))
            # print('s_rir:', len(s_rir), 's_rir[0]:', s_rir[0].shape) # s_rir: 24 s_rir[0]: (2, 2051)

            # 1 加点噪声源
            DDR_index  = self.rng.integers(0, len(self.ddr_snr), size=len(single_noise_mic))
            DDR        = np.array(self.ddr_snr)[DDR_index].tolist()
            temp5 = self.rng.random(1)[0]
            if temp5 < 0.25:
                if self.rng.random(1)[0] < 0.4:
                    data_list  = delta.addnoise(data_list, single_noise_mic, DDR, num_thread=8) # 目标加点噪
                    down_audio(data_list, 'pad_after_1noise', 1)
                else:
                    inter_list = delta.addnoise(inter_list, single_noise_mic, DDR, num_thread=8) # 干扰加点噪

            #2 多通道仿真
            s_list      = delta.conv_reverb_mc(data_list, s_rir, dropout=True, num_thread=8)
            inter1_list = delta.conv_reverb_mc(inter_list, i1_rir, dropout=True, num_thread=8)
            down_audio(s_list, 'pad_after_2rir', 2)
            down_audio(inter1_list, 'inter1_list_2rir', 2)
            if self.rng.random(1)[0] > 0.7:
                random.shuffle(inter_list)
                inter2_list = delta.conv_reverb_mc(inter_list, i2_rir, dropout=True, num_thread=8)
            else:
                inter2_list = delta.conv_reverb_mc(music_list, i2_rir, dropout=True, num_thread=8)

            #3 加干扰
            IIR_index = self.rng.integers(0, len(self.ii_snr), size=len(s_list))
            IIR       = np.array(self.ii_snr)[IIR_index]
            SIR_index = self.rng.integers(0, len(self.inter_snr), size=len(s_list))
            SIR       = np.array(self.inter_snr)[SIR_index]
            SNR_index = self.rng.integers(0, len(self.noise_snr), size=len(s_list))
            SNR       = np.array(self.noise_snr)[SNR_index]

            temp4 = self.rng.random(1)[0]
            if temp4 < 0.3:
                noisy_addList = s_list ## 不加干扰
            elif temp4 < 0.6:
                noisy_addList, _ = delta.addnoise_mc(s_list, inter1_list, SIR.tolist(), num_thread=8) ## 加单点仿真干扰
                down_audio(noisy_addList, 'pad_after_3addonenoise', 2)
            else:
                i_dual_mic, _    = delta.addnoise_mc(inter1_list, inter2_list, IIR.tolist(), num_thread=8)
                noisy_addList, _ = delta.addnoise_mc(s_list, i_dual_mic, SIR.tolist(), num_thread=8)  ## 加两点仿真干扰
                down_audio(noisy_addList, 'pad_after_3addtwonoise', 2)

            #4 加散噪
            if temp5 > 0.2:
                noisy_addList, _ = delta.addnoise_mc(noisy_addList, noise_list, SNR.tolist(), num_thread=8) # 加散噪
                down_audio(noisy_addList, 'pad_after_4addsan', 2)

            #5 mab
            beamS = []
            beamN = []
            beamC = []
            for ii in range(len(noisy_addList)):
                AECout  = noisy_addList[ii]
                MicDist = MicDist_list[ii]
                if beam_pos[ii] == 1:
                    indxS = 0
                    indxN = 1
                else:
                    indxS = 1
                    indxN = 0
                # print('AECout:', AECout.shape, 'MicDist:', MicDist, 'beam_pos:', beam_pos) # AECout: (2, 169280) MicDist: 0.136 beam_pos: [1, 1, 0]
                data_len0 = sent1st_out[ii]
                Tframe0   = data_len0 // 256 # Tframe0: 610
                AECout    = np.concatenate((AECout[:, :Tframe0*256],AECout), axis=1)

                n_mic     = AECout.shape[0]
                n_sample  = AECout.shape[1]
                Tframe    = (n_sample - self.NFFT)//self.shift + 1
                AECout    = AECout[:, : self.NFFT + (Tframe - 1) * self.shift]

                aec_out   = np.zeros((n_mic, 256, Tframe)) # aec_out: (2, 256, 1356) float64
                lasty     = np.zeros((n_mic, 256, Tframe))
                for tt in range(Tframe):
                    aec_out[:, :, tt] = AECout[:, tt*self.shift:(tt+1)*self.shift] # [2, 256, Tframe]
                lasty     = 0 * aec_out
                leak      = np.ones((aec_out.shape[2], 2))

                mab_out, m_H, Wfix, Wanc, m_Eq = delta.do_mab_car(aec_out.transpose(1, 0, 2), lasty.transpose(1, 0, 2), leak, self.doa, self.doa_constrained, self.ChiefBeam, MicDist)
                # print('mab_out:', mab_out.shape, 'm_H:', m_H.shape, 'Wfix:', Wfix.shape, 'Wanc:', Wanc.shape, 'm_Eq:', m_Eq.shape, 'sent1st_out[ii]:', sent1st_out[ii])
                # mab_out: (257, 3, 866) m_H: (257, 3, 866) Wfix: (257, 3, 866) Wanc: (257, 3, 866) m_Eq: (257, 3, 866) sent1st_out[ii]: 52960
                mab_out = mab_out[:, :, Tframe0:]
                m_H     = m_H[:, :, Tframe0:].transpose(1, 2, 0)
                Wfix    = Wfix[:, :, Tframe0:].transpose(1, 2, 0)
                Wanc    = Wanc[:, :, Tframe0:].transpose(1, 2, 0)
                m_Eq    = m_Eq[:, :, Tframe0:].transpose(1, 2, 0)

                aec_out = aec_out[:, :, Tframe0:]

                OUT     = mab_out[:, :2, :].transpose(1, 2, 0)
                OUT_R   = np.real(OUT) # OUT_R: (2, 2343, 257) float32
                OUT_I   = np.imag(OUT)

                out     = delta.stft_c2r([OUT_R.copy()], [OUT_I.copy()], 512, num_thread=8)[0].astype('int16')
                out     = out[:, 256:] # out: (2, 168704) int16
                down_audio(out, 'pad_after_5mab', 2)

                clean   = s_list[ii].copy()
                Clean_R, Clear_I = delta.stft_r2c([clean], 512, num_thread=8)
                Clean   = Clean_R[0] + 1j*Clear_I[0] # Clean: (2, 2343, 257) complex64
                X1      = Clean[0, :, :] # [T, 257]
                X2      = Clean[1, :, :]
                Clean_Out = np.zeros((1, X1.shape[0], X1.shape[1]), dtype=np.complex64)
                m_U     = X2 - m_H[indxS, :, :]*X1
                m_U     = m_U * m_Eq[indxS, :, :]
                if indxS == 0:
                    jj  = 0
                else:
                    jj  = 2
                m_FB    = (X1 + X2*np.conj(Wfix[jj, :, :])) / 2
                Clean_Out[0, :, :] = m_FB - Wanc[indxS, :, :]*m_U
                Clean_Out_R = np.real(Clean_Out)
                Clean_Out_I = np.imag(Clean_Out)
                clean_out   = delta.stft_c2r([Clean_Out_R.copy()], [Clean_Out_I.copy()], 512, num_thread=8)[0].astype('int16') # clean_out: (1, 168960) int16
                down_audio(clean_out, 'pad_after_5mab_clean', 1)

                Noise_Out = np.zeros((1, X1.shape[0], X1.shape[1]), dtype=np.complex64)
                m_U       = X2 - m_H[indxN, :, :] * X1
                m_U       = m_U * m_Eq[indxN, :, :]
                if indxN == 0:
                    jj = 0
                else:
                    jj = 2
                m_FB    = (X1 + X2 * np.conj(Wfix[jj, :, :])) / 2
                Noise_Out[0, :, :] = m_FB - Wanc[indxN, :, :] * m_U

                P_Clean = np.sum(np.abs(Clean_Out) ** 2)
                P_Noise = np.sum(np.abs(Noise_Out) ** 2)
                try:
                    SNR_C   = 10 * math.log10( P_Clean / (P_Noise+1e-6) )
                except:
                    a=1

                if SNR_C <= 0:
                    Noise_Out_R = np.real(Noise_Out)
                    Noise_Out_I = np.imag(Noise_Out)
                    noise_out   = delta.stft_c2r([Noise_Out_R.copy()], [Noise_Out_I.copy()], 512, num_thread=8)[0].squeeze(0).astype('int16')
                    Tt          = min(out.shape[1], clean_out.shape[1], noise_out.shape[0])
                    out         = out[:, :Tt]
                    clean_out   = clean_out[:, :Tt]
                    noise_out   = noise_out[:Tt]
                    out[indxS, :] = (clean_out.squeeze(0)).copy()
                    out[indxN, :] = (clean_out.squeeze(0)*0).copy()

                beamS.append(out[indxS, :])
                beamN.append(out[indxN, :])
                beamC.append(clean_out.squeeze(0))
                down_audio(beamS, 'pad_after_5mab_out_S', 1)
                down_audio(beamN, 'pad_after_5mab_out_N', 1)
                down_audio(beamC, 'pad_after_5mab_out_C', 1)

            amp = self.rng.choice(self.amp_list, len(data_list), replace=False)
            data_out1, data_out2, data_out3, label_list, syllable_list, wordid_list, sequence_len = self.__ampChangeMultiCh__(beamS, beamC, beamN, label_list, syllable_list, wordid_list, sequence_len, amp)
            down_audio(beamS, 'pad_a_S', 1)
            down_audio(beamC, 'pad_a_C', 1)
            down_audio(data_out1, 'pad_f_S', 1)
            down_audio(data_out2, 'pad_f_C', 1)
            down_audio(data_out3, 'pad_f_N', 1)

            # 提特征 main clean n1
            fbdata_out1 = self.build_fb40(data_out1, num_thread=8)
            fbdata_out2 = self.build_fb40(data_out2, num_thread=8)
            fbdata_out3 = self.build_fb40(data_out3, num_thread=8)

            if not self.is_pad:
                fbdata_out1, fbdata_out2, fbdata_out3, label_list, syllable_list, wordid_list = self.__get_sent_from_MC__(fbdata_out1, fbdata_out2, fbdata_out3, label_list, syllable_list, wordid_list, sented_out)

        list_datapb, list_labelpb, list_syllablepb, list_wordidpb, list_data_noisy, list_data_clean, list_label, list_syllable, list_wordid, sequence_list \
            = self.__calFeanorm_pad0__(fbdata_out1, fbdata_out2, fbdata_out3, label_list, syllable_list, wordid_list, datapb_list, labelpb_list, syllablepb_list, wordidpb_list)
        # data_noisy_pb, labelpb_out, syllablepb_out, wordidpb_out, data_noisy, data_clean, label_out, syllable_out, wordid_out, sequence_out

        if self.is_pad:
            # 句首非满视野扰动
            # length_del = self.rng.integers(0, self.train_padhead_frame)
            length_del = 0
            sequence_len = [ [0, np.shape(item)[0]] for item in label_list ]
            # 单通道数据pad至16帧，在幅值扰动（依据ispad）pad间隔；多通道数据在paddata中pad间隔
            data_noisy, data_mask, fb_maxlen, fb_bg_ed = self.__pad_nmod_alongT__(data_noisy_list, self.nmod_pad, 0, length_del)
            data_clean, _,_,_ = self.__pad_nmod_alongT__(data_clean_list, self.nmod_pad, 0, length_del)
            label, _,_,_      = self.__pad_nmod_alongT__(label_list, self.nmod_pad, -2, length_del)
            data_mask = label.squeeze(1).squeeze(1).clone()
            data_mask[data_mask>-2] = 1
            data_mask[data_mask<=-2]= 0
            # syllable, _,_,_ = self.__pad_nmod_alongT__(syllable_list, self.nmod_pad, -2, length_del)
            syllable, _       = self.__pad_nmod_alongB__(syllable_list, self.nmod_pad, -2)
            wordid, _,_,_     = self.__pad_nmod_alongT__(wordid_list, self.nmod_pad, -2, length_del)
        else:
            sequence_len   = [ [0, np.shape(item)[0]] for item in label_list ]
            data_noisypb, data_maskpb = self.__pad_nmod_alongB__(list_datapb, self.nmod_pad, 0)
            labelpb, _     = self.__pad_nmod_alongB__(list_labelpb, self.nmod_pad, -2)
            syllablepb, _  = self.__pad_nmod_alongB__(list_syllablepb, self.nmod_pad, -2)
            wordidpb, _    = self.__pad_nmod_alongB__(list_wordidpb, self.nmod_pad, -2)
            
            if len(data_list) != 0:
                data_noisy, data_mask = self.__pad_nmod_alongB__(list_data_noisy, self.nmod_pad, 0)
                data_clean, data_mask = self.__pad_nmod_alongB__(list_data_clean, self.nmod_pad, 0)
                label, _     = self.__pad_nmod_alongB__(list_label, self.nmod_pad, -2)
                syllable, _  = self.__pad_nmod_alongB__(list_syllable, self.nmod_pad, -2)
                wordid, _    = self.__pad_nmod_alongB__(list_wordid, self.nmod_pad, -2)

        down_lab(syllable_list, 'pad_final_sys')
        
        data_noisypb = torch.cat( data_noisypb.chunk(2, 2), dim=1 ) #b c h t
        point_pb     = len(datapb_list)
        if len(data_list) != 0:
            data_noisy = torch.cat( data_noisy.chunk(2, 2), dim=1 ) #b c h t
            data_clean = torch.cat( data_clean.chunk(2, 2), dim=1 )
            data       = torch.cat((data_noisy, data_clean), 0)
            label      = torch.cat((label, label), 0)
            syllable   = torch.cat((syllable, syllable), 0)
            wordid     = torch.cat((wordid, wordid), 0)
            data_mask  = torch.cat((data_mask, data_mask), 0)
        
        if self.is_pad:
            data       = data[0,0,:,:].unsqueeze(0).unsqueeze(0) # need to fix 区分回放和仿真
            label      = label[0,:,:,:].unsqueeze(0) # need to fix 区分回放和仿真
            syllable   = syllable[point_pb:,:,:,:] # need to fix 区分回放和仿真
            wordid     = wordid[0,:,:,:].unsqueeze(0) # need to fix 区分回放和仿真
        else:
            pass

        # assert(data.size(3)==label.size(3)==wordid.size(3))
        # assert(data.shape[0]==data_mask.shape[0])
        # assert(data.shape[-1]==data_mask.shape[1])

        # 数据转换和输出
        meta = {}
        if len(data_list) != 0:
            label_mask = label.clone()
            label[label<=-2] = -1
            label_mask[label_mask > -2] = 1
            label_mask[label_mask <= -2] = 0
            label      = label.squeeze(1).squeeze(1) #b t
            wordid     = wordid.squeeze(1).squeeze(1)
            label_mask = label_mask.squeeze(1).squeeze(1)

            syllable_mask = syllable.clone()
            syllable_mask[syllable_mask >= 0] = 1
            syllable_mask[syllable_mask < 0]  = 0
            maxlen        = syllable_mask.sum(3).max().int()
            syllable      = syllable[:, :, :, :maxlen]
            syllable_mask = syllable_mask[:, :, :, :maxlen]
            syllable      = syllable.squeeze(1).squeeze(1)
            syllable      = syllable.transpose(1, 0)
            syllable_mask = syllable_mask.squeeze(1).squeeze(1)
            syllable_mask = syllable_mask.transpose(1, 0)
            
            meta['noise'] = {}
            meta['noise']["mask"]           = data_mask.contiguous()
            meta['noise']["frames_label"]   = label.long().contiguous() #b, t
            meta['noise']["frames_mask"]    = label_mask.float().contiguous()
            meta['noise']["syllable_label"] = syllable.long().contiguous()
            meta['noise']["att_label"]      = syllable.long().contiguous()
            meta['noise']["syllable_mask"]  = syllable_mask.float().contiguous()
            meta['noise']["wordid_label"]   = wordid.long().contiguous()
            meta['noise']['sequence_len']   = torch.Tensor(sequence_len)
            meta['noise']["rnn_mask"]       = data_mask.transpose(1, 0).unsqueeze(2).contiguous()
            meta['noise']["inputs_length"]  = get_length(meta['noise']["rnn_mask"]).contiguous()
            meta['noise']["inputs_pad_length"]  = get_length(torch.ones_like(meta['noise']["rnn_mask"])).contiguous()
            # meta["inputs_length"]  = get_length(torch.ones_like(meta["rnn_mask"]) ).contiguous()
            meta['noise']["targets_length"] = get_length(meta['noise']["syllable_mask"]).contiguous()
            # meta["point_pb"]       = point_pb
            meta['noise']['data']    = data.detach()
        else:
            a=1
        # 数据转换和输出
        meta['pb']   = {}
        label_maskpb = labelpb.clone()
        labelpb[labelpb<=-2] = -1
        label_maskpb[label_maskpb > -2] = 1
        label_maskpb[label_maskpb <= -2] = 0
        labelpb      = labelpb.squeeze(1).squeeze(1) #b t
        wordidpb     = wordidpb.squeeze(1).squeeze(1)
        label_maskpb = label_maskpb.squeeze(1).squeeze(1)

        syllable_maskpb = syllablepb.clone()
        syllable_maskpb[syllable_maskpb >= 0] = 1
        syllable_maskpb[syllable_maskpb < 0]  = 0
        maxlenpb        = syllable_maskpb.sum(3).max().int()
        syllablepb      = syllablepb[:, :, :, :maxlenpb]
        syllable_maskpb = syllable_maskpb[:, :, :, :maxlenpb]
        syllablepb      = syllablepb.squeeze(1).squeeze(1)
        syllablepb      = syllablepb.transpose(1, 0)
        syllable_maskpb = syllable_maskpb.squeeze(1).squeeze(1)
        syllable_maskpb = syllable_maskpb.transpose(1, 0)

        meta['pb']["mask"]           = data_maskpb.contiguous()
        meta['pb']["frames_label"]   = labelpb.long().contiguous() #b, t
        meta['pb']["frames_mask"]    = label_maskpb.float().contiguous()
        meta['pb']["syllable_label"] = syllablepb.long().contiguous()
        meta['pb']["att_label"]      = syllablepb.long().contiguous()
        meta['pb']["syllable_mask"]  = syllable_maskpb.float().contiguous()
        meta['pb']["wordid_label"]   = wordidpb.long().contiguous()
        meta['pb']['sequence_len']   = torch.Tensor(sequence_len)
        meta['pb']["rnn_mask"]       = data_maskpb.transpose(1, 0).unsqueeze(2).contiguous()
        meta['pb']["inputs_length"]  = get_length(meta['pb']["rnn_mask"]).contiguous()
        meta['pb']["inputs_pad_length"]  = get_length(torch.ones_like(meta['pb']["rnn_mask"])).contiguous()
        # meta["inputs_length"]  = get_length(torch.ones_like(meta["rnn_mask"]) ).contiguous()
        meta['pb']["targets_length"] = get_length(meta['pb']["syllable_mask"]).contiguous()

        if self.is_pad:
            meta["fb_max_len"] = fb_maxlen
            meta["fb_bg_ed"]   = fb_bg_ed
            meta["inputs_length"] = fb_bg_ed[:,-1]
            # if 1:
                # data_, label_, data_mask_ = data_reshape(data, meta["frames_label"], meta["fb_bg_ed"], meta["fb_max_len"], self.nmod_pad)
        if 0:
            out = {}
            for key in meta:
                if isinstance(meta[key], int) or meta[key]==None:
                    continue
                meta[key] = meta[key].squeeze().numpy()
                out[key]= meta[key]
            np.save('z.npy', out)
        
        return data_noisypb.detach(), meta

class Collater():
    def __init__(self, nmod_pad):
        self.nmod_pad = nmod_pad

    def __pad_nmod__(self, sequence, nmod_pad, val):
        maxlen    = 0
        pad_list  = []
        mask_list = []
        for nparray in sequence:
            if nparray.shape[0] > maxlen:
                maxlen = nparray.shape[0]
        if nmod_pad is not None:
            maxlen  = maxlen if maxlen % nmod_pad == 0 else maxlen + nmod_pad - maxlen % nmod_pad
        for nparray in sequence:
            padlen  = maxlen - nparray.shape[0]
            nparray = np.pad(nparray, ((0, padlen), (0, 0)), mode="constant", constant_values=(val,))
            nparray = nparray.transpose(1, 0)
            torcharray = torch.from_numpy(nparray).float()
            torchmask  = torch.ones(1, torcharray.size()[1])
            if padlen > 0:
                torchmask[:, -padlen:] = 0
            torcharray = torcharray.reshape(1, 1, torcharray.size()[0], torcharray.size()[1])
            pad_list.append(torcharray)
            mask_list.append(torchmask)
        batch_array = torch.cat(pad_list, dim=0)
        batch_mask  = torch.cat(mask_list, dim=0)
        return batch_array, batch_mask

    def __call__(self, batch):
        data_list     = []
        label_list    = []
        syllable_list = []
        wordid_list   = []
        sequence_len  = []
        for data, label, syllable, wordid in batch:
            data_list.extend(data)
            label_list.extend(label)
            syllable_list.extend(syllable)
            wordid_list.extend([np.ones((subdata.shape[0], 1))*wordid[idx] for idx, subdata in enumerate(label)] )
            sequence_len.extend([np.shape(subdata)[0] for subdata in data])

        data, data_mask = self.__pad_nmod__(data_list, self.nmod_pad, 0)
        label, _    = self.__pad_nmod__(label_list, self.nmod_pad, -1)
        syllable, _ = self.__pad_nmod__(syllable_list, self.nmod_pad, -1)
        wordid, _   = self.__pad_nmod__(wordid_list, self.nmod_pad, -1)

        # 多通道数据转换格式
        data       = torch.cat( data.chunk(2, 2), dim=1 )
        data       = data[:,0,:,:].unsqueeze(1)

        # 数据转换和输出
        label_mask = label.clone()
        label[label==-2] = -1
        label_mask[label_mask >= -2] = 1
        label_mask[label_mask < -2] = 0
        label      = label.squeeze(1).squeeze(1) #b t
        wordid     = wordid.squeeze(1).squeeze(1)
        label_mask = label_mask.squeeze(1).squeeze(1)

        syllable_mask = syllable.clone()
        syllable_mask[syllable_mask >= 0] = 1
        syllable_mask[syllable_mask < 0]  = 0
        maxlen        = syllable_mask.sum(3).max().int()
        syllable      = syllable[:, :, :, :maxlen]
        syllable_mask = syllable_mask[:, :, :, :maxlen]
        syllable      = syllable.squeeze(1).squeeze(1)
        syllable      = syllable.transpose(1, 0)
        syllable_mask = syllable_mask.squeeze(1).squeeze(1)
        syllable_mask = syllable_mask.transpose(1, 0)
        meta       = {}

        meta["mask"]           = data_mask.contiguous()
        meta["frames_label"]   = label.long().contiguous()
        meta["frames_mask"]    = label_mask.float().contiguous()
        meta["syllable_label"] = syllable.long().contiguous()
        meta["att_label"]      = syllable.long().contiguous()
        meta["syllable_mask"]  = syllable_mask.float().contiguous()
        meta["wordid_label"]   = wordid.long().contiguous()
        meta['sequence_len']   = torch.Tensor(sequence_len)
        meta["rnn_mask"]       = data_mask.transpose(1, 0).unsqueeze(2).contiguous()
        meta["inputs_length"]  = get_length(meta["rnn_mask"]).contiguous()
        meta["inputs_pad_length"]  = get_length(torch.ones_like(meta["rnn_mask"])).contiguous()
        meta["targets_length"] = get_length(meta["syllable_mask"]).contiguous()

        return data.detach(), meta

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset     = worker_info.dataset
    dataset.lmdb_env = lmdb.Environment(dataset.lmdb_path, readonly=True, readahead=True, lock=False)

def worker_init_fn_online(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset     = worker_info.dataset
    num_workers = worker_info.num_workers
    worker_idx  = worker_info.id

    # 随机种子
    rng  = np.random.default_rng()
    seed = dataset.seed
    if seed is not None:
        rng = np.random.default_rng(seed*4)

    # 初始化lmdb
    if dataset.lmdbunipb_path != None:
        # 初始化lmdb
        dataset.lmdb_unipb = lmdb.Environment(dataset.lmdbunipb_path, readonly=True, readahead=True, lock=False)

    if dataset.lmdbuni_path != None:
        # 初始化lmdb
        dataset.lmdb_uni = lmdb.Environment(dataset.lmdbuni_path, readonly=True, readahead=True, lock=False)

        if dataset.lmdbnoise_path != None:
            # 初始化lmdb
            dataset.lmdb_noise = lmdb.Environment(dataset.lmdbnoise_path, readonly=True, readahead=True, lock=False)

        if dataset.lmdbinter_path != None:
            # 初始化lmdb
            dataset.lmdb_inter = lmdb.Environment(dataset.lmdbinter_path, readonly=True, readahead=True, lock=False)

        if dataset.lmdbrir_path != None:
            # 初始化lmdb
            dataset.lmdb_rir = lmdb.Environment(dataset.lmdbrir_path, readonly=True, readahead=True, lock=False)

        if dataset.lmdbmusic_path != None:
            # 初始化lmdb
            dataset.lmdb_music = lmdb.Environment(dataset.lmdbmusic_path, readonly=True, readahead=True, lock=False)

def get_train_dataloader(args):
    speechdataset = SpeechDataset_Online(
        args.lmdbuni_path,
        args.lmdbunipb_path,
        args.lmdbnoise_path,
        args.lmdbinter_path,
        args.lmdbrir_path,
        args.lmdbmusic_path,
        max_sent_frame=args.max_sent_frame,
        min_sent_frame=args.min_sent_frame,
        start_line=args.start_line,
        end_line=args.end_line,
        train_ratio_uni=args.train_ratio_uni,
        train_ratio_unipb=args.train_ratio_unipb,
        train_ratio_inter=args.train_ratio_inter,
        train_padhead_frame=args.train_padhead_frame,
        train_padtail_frame=args.train_padtail_frame,
        rank=args.gpu_global_rank,
        world_size=args.gpu_world_size,
        seed=args.seed,
    )
    train_sampler = BunchSampler(
        dataset_lengths=speechdataset.data_unipb_keys_lens,
        dataset_lengths_clean=speechdataset.data_uni_keys_lens,
        batch_size=args.batch_size,
        bunch_size=args.bunch_size,
        bunch_size_clean=args.bunch_size_clean,
        drop_last=False,
        shuffle_batch=True,
        # shuffle_batch=False,
        iter_num=args.train_iter_num,
        seed=args.seed,
        is_sort=args.is_sort
    )
    train_dataloader = DataLoader(
        speechdataset,
        batch_sampler=train_sampler,
        num_workers=8, #czzhu2
        collate_fn=Collater_online(
            args.nmod_pad,
            args.lmdb_reverbfile,
            args.lmdb_normfile,
            args.train_padhead_frame,
            args.train_padtail_frame,
            args.train_padmid_frame,
            seed=args.seed,
            is_pad=args.is_pad
            ),
        worker_init_fn=worker_init_fn_online,
        multiprocessing_context="spawn"
    )
    # args.train_iter_num, sent_nums = estimate_train_iter_num(args, speechdataset.data_lens)
    return train_dataloader, args.train_iter_num

def get_val_dataloader(args):
    speechdataset = SpeechDatasetSingleCh(
        args.cv_lmdb_path,
        args.lmdb_normfile,
        max_sent_frame=args.max_sent_frame,
        min_sent_frame=args.min_sent_frame,
        start_line=0,
        end_line=args.val_sent_num,
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
        collate_fn=Collater(args.nmod_pad),
        worker_init_fn=worker_init_fn,
        multiprocessing_context="spawn"
    )
    return val_dataloader

def get_val_dataloader_mc(args):
    speechdataset = SpeechDatasetMultiCh(
        args.lmdbmc_path,
        args.lmdbmc_key,
        args.lmdb_normfile,
        args.lmdb_syllableid,
        max_sent_frame=args.max_sent_frame,
        min_sent_frame=args.min_sent_frame,
        start_line=0,
        end_line=args.val_sent_num,
        rank=0,
        world_size=1,
        is_ivw=args.is_ivw
    )
    val_dataloader = DataLoader(
        speechdataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        collate_fn=Collater(args.nmod_pad),
        worker_init_fn=worker_init_fn,
        multiprocessing_context="spawn"
    )
    return val_dataloader

def estimate_train_iter_num(args, length_list=[]):
    max_train_iter_num = 0
    if len(length_list) == 0:
        for i, line in enumerate(open(get_lmdb_key(args.lmdbunipb_path))):
            line = line.strip()
            if line == "":
                continue
            items = line.split()
            sent_frame = int(items[1])
            sent_id = int(items[2])
            if sent_frame <= args.max_sent_frame and sent_frame >= args.min_sent_frame:
                length_list.append(sent_frame)

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
    if total_length > 0:
        max_train_iter_num += 1
    max_train_iter_num = max_train_iter_num // args.gpu_world_size
    return max_train_iter_num-300, len(length_list) #迭代次数，总句子数

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdbuni_path',   type=str, default='/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb_clean/lmdb0/', help='Path of the lmdb universal dataset.')
    parser.add_argument('--lmdbunipb_path', type=str, default='/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb/lmdb0/', help='Path of the lmdb universal dataset.')
    # parser.add_argument('--lmdbunipb_path', type=str, default='/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb/lmdbCV/', help='Path of the lmdb universal dataset.')
    parser.add_argument('--lmdbnoise_path', type=str, default='/yrfs4/acousticpro/xfyi/Car/lmdb/lmdb_diffunoise_2mic', help='Path of the lmdb noise dataset.')
    parser.add_argument('--lmdbinter_path', type=str, default='/train8/asrkws/kaishen2/Data/Universal_CHS_srf/srf_rand_0.38sp_0.38normal_partof38/lmdb1', help='Path of the lmdb none dataset.')
    parser.add_argument('--lmdbrir_path',   type=str, default='/yrfs4/acousticpro/xfyi/Car/lmdb/lmdb_rir_2mic_all', help='Path of the lmdb none dataset.')
    parser.add_argument('--lmdbmusic_path', type=str, default='/train8/asrkws/kaishen2/Data/Universal_MC_music/lmdb0', help='Path of the lmdb none dataset.')
    parser.add_argument('--cv_lmdb_path',   type=str, default='/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb/lmdbCV/', help='Path of the cv lmdb dataset.')
    parser.add_argument('--lmdb_reverbfile',type=str, default='/home/asr/kaishen2/bin/kws/reverb/reverb.npy', help='Path of the lmdb norm file.')
    parser.add_argument('--lmdb_syllableid',type=str, default='/home/asr/kaishen2/bin/kws/kws3003_phone2syllableDict.npy', help='Path of the lmdb syllable file.')
    parser.add_argument('--lmdb_normfile',  type=str, default='/home/asr/kaishen2/bin/kws/fea_fb40.norm', help='Path of the lmdb norm file.')
    parser.add_argument('--train_ratio_unipb', type=int, default=1, help='Number of sentences for train per command')
    parser.add_argument('--train_ratio_uni',type=int, default=2, help='Number of sentences for train per command')
    parser.add_argument('--train_ratio_inter', type=int, default=4, help='Number of sentences for train per command')
    parser.add_argument('--train_padhead_frame', type=int, default=192, help='Pad Number of sentences for train in head(frames)')
    parser.add_argument('--train_padtail_frame', type=int, default=16, help='Pad Number of sentences for train in tai(frames)')
    parser.add_argument('--train_padmid_frame',  type=int, default=16, help='Pad Number of sentences for train in tai(frames)')
    parser.add_argument('--val_sent_num', type=int, default=1000, help='Number of sentences for val')
    parser.add_argument('--max_sent_frame', type=int, default=900, help='Max sent frame')
    parser.add_argument('--min_sent_frame', type=int, default=40, help='Max sent frame')
    parser.add_argument('--is_pad', type=bool, default=False, help='True means:1,b*t, False means:b,t')
    parser.add_argument('--is_sort', type=bool, default=True, help='sort by length')

    parser.add_argument('--batch_size', type=int, default=1024, help='Max sentence number used for training and testing')
    parser.add_argument('--bunch_size', type=int, default=3000, help='Total frame size one batch used for training and testing')
    parser.add_argument('--bunch_size_clean', type=int, default=512, help='Total frame size one batch used for training and testing')
    parser.add_argument('--train_iter_num', type=int, help='Iter num')
    parser.add_argument('--gpu_world_size', type=int, default=4, help='Gpu nums')
    parser.add_argument('--gpu_global_rank',type=int, default=0, help='Gpu rank')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--nmod_pad', type=int, default=4, help='Low frame')
    parser.add_argument('--start_line', type=int, default=0, help='Start line')
    parser.add_argument('--end_line', type=int, default=0, help='End line')

    args = parser.parse_args()
    args.end_line   = int(os.popen('wc -l {}'.format(get_lmdb_key(args.lmdbunipb_path))).read().split()[0])
    
    # train_iter_num, end_expand = estimate_train_iter_num(args) #迭代次数，总句子数
    # val_dataloader = get_val_dataloader(args)

    # for i, data_element in enumerate(val_dataloader):
    #     data, meta = data_element
    #     print (i, data.shape)
    train_iter_num, sent_num = estimate_train_iter_num(args)
    args.train_iter_num = train_iter_num
    train_dataloader, train_iter_num = get_train_dataloader(args)
    print (train_iter_num)
    for i, data_element in enumerate(train_dataloader):
        data, meta = data_element
        print (i, data.shape)
        a=1
        # for key in meta:
        #     if isinstance(meta[key], int) or meta[key]==None:
        #         continue
        #     meta[key] = meta[key]#.cuda()
    pass