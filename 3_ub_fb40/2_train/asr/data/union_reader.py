# modified from dlp's pfile.py
# by ssyan2

import os
import numpy as np
import random
import copy
import lmdb
import delta
import struct
import time

try:
    from .datum_pb2 import SpeechDatum #Float MC

except:
    from datum_pb2 import SpeechDatum #Float MC

"""
归一化类
输入归一化文件地址，返回mean与var
"""
class Normfile():
    def __init__(self, filename):
        assert os.path.isfile(filename), "file not exists {0}".format(filename)
        fp = open(filename, 'r')
        content = fp.readlines()
        fp.close()
        self.mean_dim = int(content[0].split()[-1])
        self.var_dim = int(content[self.mean_dim+1].split()[-1])
        assert self.mean_dim == self.var_dim, "{0}: mean dim {1} != var dim {2}".format(filename, self.mean_dim, self.var_dim)
        assert self.mean_dim + self.var_dim + 2 == len(content), "norm file error: {0}".format(filename)
        mean = np.array([float(line[:-1]) for line in content[1:self.mean_dim+1]], dtype=np.float32)
        var = np.array([float(line[:-1]) for line in content[self.mean_dim+2:]], dtype=np.float32)
        self.mean = np.array([float(line[:-1]) for line in content[1:self.mean_dim+1]], dtype=np.float32)
        self.var = np.array([float(line[:-1]) for line in content[self.mean_dim+2:]], dtype=np.float32)

"""
pfile文件解析器
输入pfile文件地址，获取相关信息，关键信息有：
seq_info 为样本信息，[样本id,数据读取地址,数据帧长度]
并定义了一个静态函数estimate_num_batch用以估计batch数
"""
class PfileInfo():
    def __init__(self, filename):
        assert os.path.isfile(filename), "file not exists: {}".format(filename)
        print("*"*10)
        print("filename=", filename)
        self.filename = filename
        self.fp = open(filename, 'r')
        line = self.fp.readline()
        items = line.split()
        self.header_version = int(items[2])
        self.header_size = int(items[-1])
        self.num_sentences = int(self.fp.readline().split()[1])
        self.num_frames = int(self.fp.readline().split()[1])
        self.first_feature_column = int(self.fp.readline().split()[1])
        self.dim_features = int(self.fp.readline().split()[1])
        self.first_label_column = int(self.fp.readline().split()[1])
        self.dim_labels = int(self.fp.readline().split()[1])
        self.data_format = self.fp.readline().split()[1].replace('d', 'i')
        self.frame_length = len(self.data_format)
        # always is 2, maybe should be fixed
        self.real_data_start = self.first_feature_column
        # sent start pos
        self.fp.close()
        self.fp = open(filename, "rb")
        self.fp.seek(self.header_size + self.frame_length*self.num_frames*struct.calcsize('f'))
        raw_binary = self.fp.read(struct.calcsize('I')*(self.num_sentences+1))
        self.seq_start_pos = struct.unpack('>'+str(self.num_sentences+1)+'I', raw_binary)
        # raw_binary = self.fp.read(struct.calcsize('i')*(self.num_sentences+1))
        # self.seq_start_pos = struct.unpack('>'+str(self.num_sentences+1)+'i', raw_binary)
        assert self.num_sentences + 1 == len(self.seq_start_pos), "pfile tail is not correct"
        assert self.num_frames == self.seq_start_pos[-1], "frame number {0} mismatch {1} frames in header".format(self.seq_start_pos[-1], self.num_frames)
        self.seq_info = []
        for i in range(self.num_sentences):
            self.seq_info.append([i, self.seq_start_pos[i], self.seq_start_pos[i+1] - self.seq_start_pos[i]])

    def __del__(self):
        self.fp.close()
    
    # 根据数据信息估算batch num，可能会存在误差
    @staticmethod
    def estimate_num_batch(seq_info, bunchsize, maxsentframe, maxnumsent, nmod_pad):
        num_batch = 0
        
        print('bunchsize.. ', bunchsize)
        ### from zqjin
        total_seqlen = 0

        for seqid, start, seqlen in seq_info:
            seqlen = seqlen if seqlen % nmod_pad == 0 else seqlen + nmod_pad - seqlen % nmod_pad
            total_seqlen += seqlen
        
        num_batch = round( total_seqlen / bunchsize)

        return num_batch

        current_maxsentframe = 0
        current_batch_sent = 0
        
        seq_info = sorted(seq_info, key=lambda e: e[2])
        for seqid, start, seqlen in seq_info:
            seqlen = seqlen if seqlen % nmod_pad == 0 else seqlen + nmod_pad - seqlen % nmod_pad
            if maxsentframe is not None:
                if seqlen > maxsentframe:
                    continue
            if seqlen > bunchsize:
                continue
            else:
                if seqlen > current_maxsentframe:
                    total_frame = seqlen * (1 + current_batch_sent)
                    current_maxsentframe = seqlen
                else:
                    total_frame = current_maxsentframe * (1 + current_batch_sent)
                if total_frame <= bunchsize:
                    current_batch_sent = current_batch_sent + 1
                else:
                    num_batch = num_batch + 1
                    current_batch_sent = 1
                if maxnumsent <= current_batch_sent:
                    num_batch = num_batch + 1
                    current_batch_sent = 0
                    current_maxsentframe = 0
        return num_batch

"""
仿照PfileInfo的lmdb文件解析器
输入lmdb文件地址，获取相关信息，关键信息有：
seq_info 为样本信息，[样本id,数据读取地址,数据帧长度]
并定义了一个静态函数estimate_num_batch用以估计batch数
定义了一个静态函数get_lmdb_item用以获取实际数据
"""
class LmdbInfo():

    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.seq_info = self.__read_seq_info()
        self.lmdb_data = self.__read_lmdb()
        self.num_sentences = len(self.seq_info)

    def __read_lmdb(self):
        lmdb_data = lmdb.Environment(self.file_dir, readonly=True, readahead=True, lock=False)
        return lmdb_data
    

    def __read_seq_info(self):
        with open(self.get_lmdb_key(self.file_dir),"r") as fr:
            lines = fr.readlines()
        seq_info = []
        for ind, line in enumerate(lines):
            line_split = line.strip().split(" ")
            seq_info.append([ind, line_split[0], int(line_split[1])])
        
        if 0 < len(seq_info):
            return seq_info
        else:
            return None
    
    def __del__(self):
        if None != self.lmdb_data:
            self.lmdb_data.close()
    
    # 用以解析一条数据
    @staticmethod
    def get_lmdb_item(txn,ind_str):
        with txn.cursor() as cursor:
            k = str(ind_str).zfill(12).encode('utf-8')
            cursor.set_key(k)
            datum_ = SpeechDatum()
            datum_.ParseFromString(cursor.value())

        return datum_

    # 用以获取key路径
    @staticmethod
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

    # 根据数据信息估算batch num，可能会存在误差
    @staticmethod
    def estimate_num_batch(seq_info, bunchsize, maxsentframe, maxnumsent, nmod_pad):
        num_batch = 0
        current_maxsentframe = 0
        current_batch_sent = 0
        
        seq_info = sorted(seq_info, key=lambda e: e[2])
        for seqid, start, seqlen in seq_info:
            seqlen = seqlen if seqlen % nmod_pad == 0 else seqlen + nmod_pad - seqlen % nmod_pad
            if maxsentframe is not None:
                if seqlen > maxsentframe:
                    continue
            if seqlen > bunchsize:
                continue
            else:
                if seqlen > current_maxsentframe:
                    total_frame = seqlen * (1 + current_batch_sent)
                    current_maxsentframe = seqlen
                else:
                    total_frame = current_maxsentframe * (1 + current_batch_sent)
                if total_frame <= bunchsize:
                    current_batch_sent = current_batch_sent + 1
                else:
                    num_batch = num_batch + 1
                    current_batch_sent = 1
                if maxnumsent <= current_batch_sent:
                    num_batch = num_batch + 1
                    current_batch_sent = 0
                    current_maxsentframe = 0
        return num_batch

"""
简化自PfileChunkReader
用以从文件读取cache并返回cachesize个句子
"""
class PfileChunkLoader():
    def __init__(self, pfileinfo, pfilelabinfo, start_sent, end_sent, cachesize=10000,val=False):
        # pcli2: assume pfileinfo checking is done
        self.pfileinfo = pfileinfo
        self.pfilelabinfo = pfilelabinfo
        self.batch_list = []
        self.pfile_cache = None
        self.pfile_name = pfileinfo.filename
        self.seq_info = pfileinfo.seq_info[start_sent:end_sent]
        self.seq_length = len(self.seq_info)
        self.cachesize = cachesize
        random.seed(float(time.time()))
        # # if len(self.seq_info) > self.cachesize:
        # #     self.cache_start_index = random.randint(0, len(self.seq_info) - self.cachesize -1) 
        # # else:
        # self.cache_start_index = 0 
        # self.cache_end_index = self.cache_start_index + (cachesize if cachesize <= self.seq_length else self.seq_length - 1)
        
        ### from zqjin
        if len(self.seq_info) - self.cachesize -1 > 0:
            self.cache_start_index = 0 + random.randint(0, len(self.seq_info) - self.cachesize -1) if not val else 0
        else:
            self.cache_start_index = 0
        self.cache_end_index = self.cache_start_index + self.cachesize

        self.fp = open(self.pfile_name, "rb")
       # print("xxtong")

        print("ChunkLoader",len(self.seq_info), start_sent, end_sent, self.cache_start_index, self.cachesize)

        if self.pfilelabinfo is not None:
            self.pfilelab_cache = None
            self.pfilelab_name = pfilelabinfo.filename
            self.labseq_info = pfilelabinfo.seq_info[start_sent:end_sent]
            self.lab_fp = open(self.pfilelab_name, "rb")

    # 用以根据seq_info，遍历获取cache_size个数据
    def __cache_into_dict(self, pfileinfo, seq_info, cache_start_index, cache_end_index, fp):
        if pfileinfo.data_format[2] == 'f':
            dtype = np.dtype("float32")
        if pfileinfo.data_format[2] == 'i':
            dtype = np.dtype("int32")
        # print('cache_end_index[1], cache_start_index[1]: ', cache_end_index, cache_start_index)
        # print('seq_info[cache_start_index][1]: ', seq_info[cache_start_index][1])
        # print('seq_info[cache_end_index][1]: ', len(seq_info), cache_end_index)
        size = seq_info[cache_end_index][1] - seq_info[cache_start_index][1]
        start = seq_info[cache_start_index][1]
        fp.seek(pfileinfo.header_size + len(pfileinfo.data_format)*dtype.itemsize*start)
        raw_binary = fp.read(len(pfileinfo.data_format)*dtype.itemsize*size)
        binary_start = 0
        binary_end = 0
        pfilecache = {}
        for i in range(cache_start_index, cache_end_index):
            repeat_format = seq_info[i][2] * pfileinfo.data_format
            binary_start = binary_end
            binary_end = binary_end + len(pfileinfo.data_format) * dtype.itemsize * seq_info[i][2]
            value = np.frombuffer(raw_binary[binary_start: binary_end], dtype=dtype)
            value = value.byteswap()
            value = np.reshape(value, [-1, pfileinfo.frame_length])
            value = value[:, pfileinfo.real_data_start:]
            pfilecache[seq_info[i][0]] = value
        return pfilecache

    # 用以获取一块cache
    def __make_cache(self):
        self.pfile_cache = self.__cache_into_dict(self.pfileinfo, self.seq_info, self.cache_start_index, self.cache_end_index, self.fp)
        if self.pfilelabinfo is not None:
            self.labpfile_cache = self.__cache_into_dict(self.pfilelabinfo, self.labseq_info, self.cache_start_index, self.cache_end_index, self.lab_fp)

    # 用以从外部调用得到cache
    def get_cache(self):
        # 若cache_end_index大于seq_length，从头取补齐
        if self.cache_end_index > self.seq_length - 1:
            # 读取尾部
            print("end cache",self.cache_end_index,self.cache_start_index,self.seq_length)
            
            cache_end_index_ = self.cache_end_index
            if self.cache_end_index > self.seq_length-1:   ###zhy, init failed
                cache_end_index_ = self.seq_length

            self.cache_end_index = self.seq_length - 1

            self.__make_cache()
            pfile_cache_ = copy.deepcopy(self.pfile_cache)
            if self.pfilelabinfo is not None:
                labpfile_cache_ = copy.deepcopy(self.labpfile_cache)
                

            # 读取头部补齐
            self.cache_start_index = 0
            self.cache_end_index = cache_end_index_ - self.seq_length

            self.__make_cache()
            # 首尾合并
            self.pfile_cache.update(pfile_cache_)
            if self.pfilelabinfo is not None:
                self.labpfile_cache.update(labpfile_cache_)
        else:
            self.__make_cache()
        
        # cache_index 后移
        self.cache_start_index = self.cache_end_index if self.cache_end_index != self.seq_length - 1 else 0
        self.cache_end_index = self.cache_start_index + self.cachesize
        
        if self.pfilelabinfo is not None:
            return self.pfile_cache, self.labpfile_cache 
            
        else:
            return self.pfile_cache

    # 销毁文件
    def __del__(self):
        self.fp.close()
        if self.pfilelabinfo is not None:
            self.lab_fp.close()


"""
仿照自PfileChunkLoader
用以从文件读取cache并返回cachesize个句子
"""
class LmdbChunkLoader():
    def __init__(self, lmdbinfo, start_sent, end_sent, cachesize=10000, val=False):

        self.lmdbinfo = lmdbinfo
        self.batch_list = []
        self.lmdb_cache = None
        self.seq_list = None
        self.seq_info = lmdbinfo.seq_info[start_sent:end_sent]
        self.seq_length = len(self.seq_info)
        self.cachesize = cachesize
        self.cache_start_index = self.cache_start_index = 0 + random.randint(0, len(self.seq_info) - self.cachesize -1) if not val else 0
        self.cache_end_index = self.cache_start_index + self.cachesize

        print("ChunkLoader",len(self.seq_info), start_sent, end_sent, self.cache_start_index, self.cachesize)
        
    # 用以根据seq_info，遍历获取cache_size个数据
    def __cache_into_list(self):
        # 若cache_end_index大于seq_length，从头取补齐
        if self.cache_end_index > self.seq_length:
            seq_list = self.seq_info[self.cache_start_index:] + self.seq_info[:self.cache_end_index - self.seq_length]
        else:
            seq_list = self.seq_info[self.cache_start_index:self.cache_end_index]
        
        txn = self.lmdbinfo.lmdb_data.begin()
        # sl[1] 为读取info，sl：[样本id,数据读取地址,数据帧长度]
        datums = [LmdbInfo.get_lmdb_item(txn,sl[1]) for sl in seq_list]

        self.cache_start_index = self.cache_end_index if self.cache_end_index != self.seq_length else 0
        self.cache_end_index = self.cache_start_index + self.cachesize
        return datums, seq_list

    # 用以从外部调用得到cache
    def get_cache(self):
        self.lmdb_cache, self.seq_list = self.__cache_into_list()
        return self.lmdb_cache, self.seq_list

    def __del__(self):
        if None != self.lmdbinfo.lmdb_data:
            self.lmdbinfo.lmdb_data.close()

"""
多输入混合类型chunk读取器类
输入数据集信息，通过调用getbatch返回一个batch数据
"""
class MultiUnionChunkReader():
    def __init__(self, work_data_infos, bunchsize=16000, maxsentframe=1000, maxnumsent=1000, nmod_pad=64, cache_sent_num=10000, shuffle=True, random_seed=0, batch_ctrl=False, val=False):
        
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.batch_ctrl = batch_ctrl
        self.val = val
        self.bunchsize = bunchsize
        self.have_lab = True


        self.train_data_infos = copy.deepcopy(work_data_infos)
        self.data_list = self.train_data_infos["data_list"]
        # get cfg rate
        self.mix_rates = []
        for dl in self.data_list:
            self.mix_rates.append(self.train_data_infos[dl]["mix_rate"])
        
        # deal batch rate
        if self.batch_ctrl:
            self.batch_rates = []
            for dl in self.data_list:
                mix_rate = self.train_data_infos[dl]["mix_rate"]
                batch_rate = round(mix_rate/min(self.mix_rates))
                self.batch_rates.append(batch_rate)
            self.mix_rates = [br/self.batch_rates[0] for br in self.batch_rates]
            self.train_data_infos["batch_rates"] = self.batch_rates
        
        self.train_data_infos["mix_rates"] = self.mix_rates
        self.split_seq_infos = {}
        #convt bunchsize to cache_sent_num to control rate By ssyan2
        if self.val:
            self.cache_sent_nums = [self.train_data_infos[dl]["validation_iternum"] for dl in self.data_list]
        else:
            self.cache_sent_nums = [round(cache_sent_num*mr/sum(self.mix_rates)) for mr in self.mix_rates]
        
        print("multi union dataloader info: mix_rates-",self.mix_rates," ;cache_sent_nums-",self.cache_sent_nums," ;cache_sent_sums-",sum(self.cache_sent_nums))

        # 根据数据类别获取loader
        for ind, dl in enumerate(self.data_list):
            self.train_data_infos[dl]["cache_sent_num"] = self.cache_sent_nums[ind]
            if "lmdb" == self.train_data_infos[dl]["data_type"]:
                self.train_data_infos[dl]["lmdb_data_info"] = LmdbInfo(self.train_data_infos[dl]["lmdb_file"])
                start_index_, end_index_ = self.train_data_infos[dl]["start_index"],self.train_data_infos[dl]["end_index"]
                cachesize_ = self.train_data_infos[dl]["cache_sent_num"]
                split_seq_info_ = self.train_data_infos[dl]["lmdb_data_info"]

                lmdb_chunk_loader_ = LmdbChunkLoader(split_seq_info_,start_index_, end_index_, cachesize_,self.val)
                self.train_data_infos[dl]["lmdb_chunk_loader"] = lmdb_chunk_loader_

            elif "pfile" == self.train_data_infos[dl]["data_type"]:
                self.train_data_infos[dl]["pfile_fea_info"] = PfileInfo(self.train_data_infos[dl]["pfile_fea"])
                self.train_data_infos[dl]["pfile_lab_info"] = PfileInfo(self.train_data_infos[dl]["pfile_lab"])
                pfileinfo_, pfilelabinfo_ = self.train_data_infos[dl]["pfile_fea_info"], self.train_data_infos[dl]["pfile_lab_info"]
                start_index_, end_index_ = self.train_data_infos[dl]["start_index"],self.train_data_infos[dl]["end_index"]
                cachesize_ = self.train_data_infos[dl]["cache_sent_num"]
                pfile_chunk_loader_ = PfileChunkLoader(pfileinfo_, pfilelabinfo_, start_index_, end_index_, cachesize_, self.val)
                self.train_data_infos[dl]["pfile_chunk_loader"] = pfile_chunk_loader_

            else:
                print("type error",dl)
                exit()
                
        
        self.aug_infos = {}
        # 用以获取仿真信息
        for ind, dl in enumerate(self.data_list):
            if "aug_info_str" in self.train_data_infos[dl].keys():
                aug_info_str_ = self.train_data_infos[dl]["aug_info_str"]
                aug_info = self.__get_aug_info(aug_info_str_)
            else: 
                aug_info = {}
            self.train_data_infos[dl]["aug_info"] = aug_info
            self.aug_infos[dl] = aug_info

        self.batch_list = []

    # 用以从字符解析仿真配置
    def __get_aug_info(self,aug_info_str):
        info = {}
        if "" == aug_info_str:
            return info
        all_keys = ["reverb","noise","amp"]
        for al_1 in aug_info_str.replace(" ","").lower().split("&"):
            if "none" in al_1:
                info = {} 
                break

            elif "all" in al_1:
                al_1_split = al_1.split(":")
                all_rate = eval(al_1_split[1])
                info = {k:all_rate for k in all_keys}
                break

            else:
                al_1_split = al_1.split(":")
                k = al_1_split[0]
                assert k in all_keys,f"please check {k} in {all_keys}"
                v = eval(al_1_split[1])
                info[k] = v
        return info
    
    # 用以从loader读入cache数据
    def __get_cache(self):

        for ind, dl in enumerate(self.data_list):
            
            if "lmdb" == self.train_data_infos[dl]["data_type"]:
                ret_cache = self.train_data_infos[dl]["lmdb_chunk_loader"].get_cache()
                datums_, seq_list_ = ret_cache
                self.train_data_infos[dl]["lmdb_cache"] = datums_
                self.train_data_infos[dl]["seq_list"] = seq_list_
                #print(dl,"cache len",len(datums_),self.cache_sent_nums[ind])

            elif "pfile" == self.train_data_infos[dl]["data_type"]:
                ret_cache = self.train_data_infos[dl]["pfile_chunk_loader"].get_cache()
            
                if self.train_data_infos[dl]["pfile_lab_info"]:
                    pfile_cache_, labpfile_cache_ = ret_cache
                else:
                    pfile_cache_ = ret_cache
                    labpfile_cache_ = None
                self.train_data_infos[dl]["pfile_cache"] = pfile_cache_
                self.train_data_infos[dl]["pfile_lab_cache"] = labpfile_cache_
                #print(dl,"cache len",len(pfile_cache_),self.cache_sent_nums[ind])

            else:
                print("type error",dl)
                exit()
            

    # 用以合并cache到的sep_list，并遍历打上data_type信息。all_seqlen_list[0]：[key,帧长,数据类型]
    def __merge_seqlen_list(self):

        all_seqlen_list = []

        for ind, dl in enumerate(self.data_list):
            seqlen_list = []
            if "lmdb" == self.train_data_infos[dl]["data_type"]:
                for index, keys in enumerate(self.train_data_infos[dl]["seq_list"]):
                    seqlen_list.append([index, keys[2], dl])

            elif "pfile" == self.train_data_infos[dl]["data_type"]:

                for key in self.train_data_infos[dl]["pfile_cache"]:
                    seqlen_list.append([key, self.train_data_infos[dl]["pfile_cache"][key].shape[0], dl])

            else:
                print("type error",dl)
                exit()
            all_seqlen_list.extend(seqlen_list)
        
        return all_seqlen_list

    # 用以拆分seqlen_list为batch
    def __shuffle_and_batch_multi(self):

        self.__get_cache()
        seqlen_list = self.__merge_seqlen_list()

        random.shuffle(seqlen_list)

        # 用以产生少许扰动，使得数据排序有小变动，但基本还是按照帧长排序
        if self.maxnumsent > 1:
            noisy_num = 2
            # 帧长度扰动
            seq_len_noisy = [e[1] + random.randint(0,noisy_num) for e in seqlen_list]
            seqlen_list = sorted(seqlen_list, key=lambda e: seq_len_noisy[seqlen_list.index(e)])

        # 根据帧长拼接batch
        current_batch = []
        current_batch_sent = 0
        current_maxsentframe = 0
        for seqid, seqlen, dl in seqlen_list:
            seqlen = seqlen if seqlen % self.nmod_pad == 0 else seqlen + self.nmod_pad - seqlen % self.nmod_pad
            if self.maxsentframe is not None:
                if seqlen > self.maxsentframe:
                    continue
            if seqlen > self.bunchsize:
                continue
            if seqlen > current_maxsentframe:
                total_frame = seqlen * (1 + current_batch_sent)
                current_maxsentframe = seqlen
            else:
                total_frame = current_maxsentframe * (1 + current_batch_sent)
            if total_frame <= self.bunchsize:
                current_batch_sent = current_batch_sent + 1
                current_batch.append([seqid,seqlen,dl])
            else:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch.append([seqid,seqlen,dl])
                current_batch_sent = 1
            if self.maxnumsent <= current_batch_sent:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_sent = 0
                current_maxsentframe = 0
            # 最后一次别忘了^^
        self.batch_list.append(current_batch)

        if self.shuffle:
            #random.seed(self.random_seed)
            random.shuffle(self.batch_list)
            random.shuffle(self.batch_list)
    
    # 用以拆分seqlen_list为batch，batch级别比例控制
    def __shuffle_and_batch_multi_batch_ctrl(self):

        self.__get_cache()
        seqlen_list = self.__merge_seqlen_list()
        
        random.shuffle(seqlen_list)
        # 用以产生少许扰动，使得数据排序有小变动，但基本还是按照帧长排序
        if self.maxnumsent > 1:
            noisy_num = 2
            # 帧长度扰动
            seq_len_noisy = [e[1] + random.randint(0,noisy_num) for e in seqlen_list]
            seqlen_list = sorted(seqlen_list, key=lambda e: seq_len_noisy[seqlen_list.index(e)])

        # 根据data_list[0]数据估算采样数目
        sample_num = len(list(filter(lambda x:self.data_list[0]==x[2],seqlen_list)))//self.batch_rates[0]

        # 拆分为最小batch比例单位的数据，如比例[2,1,1]，先去除2,1,1条，后面在根据bunchsize 拼接为nx[2,1,1]
        sample_batch = []
        # 取sample_num个最小batch比例单位的数据
        for inx in range(sample_num):
            sample_batch_min = []
            # 遍历数据集取对应batch比例条数
            for ind,dl_name in enumerate(self.data_list):
                batch_num = self.batch_rates[ind]
                try:
                    # 若某些超出了，会出错
                    sample_one = list(filter(lambda x:dl_name==x[2],seqlen_list))[inx*batch_num:inx*batch_num+batch_num]
                except:
                    # 超出了从尾部取，因为一般都是后面不够
                    sample_one = list(filter(lambda x:dl_name==x[2],seqlen_list))[-1*batch_num:]
                sample_batch_min.extend(sample_one)
            
            # 获取sample_batch_min最大帧
            current_maxsentframe = max(list(map(lambda x: x[1],sample_batch_min)))
            total_frame = current_maxsentframe * len(sample_batch_min)
            # 若sample_batch_min的total_frame小于bunchsize，直接用
            if total_frame < self.bunchsize:
                sample_batch.append(sample_batch_min)
            # 否则按照bunchsize继续拆分，防止使用小显存gpu时，最小单位batch比例也大于bunchsize
            else:
                sample_batch_min_split = []
                sample_batch_min_split_count = []
                np.random.shuffle(sample_batch_min)
                for sbm in sample_batch_min:
                    sample_batch_min_split_count.append(sbm)
                    current_maxsentframe = max(list(map(lambda x: x[1],sample_batch_min_split_count)))
                    total_frame = current_maxsentframe * len(sample_batch_min_split_count)
                    if total_frame < self.bunchsize:
                        sample_batch_min_split.append(sbm)
                    else:
                        sample_batch.append(sample_batch_min_split)
                        
                        sample_batch_min_split = []
                        sample_batch_min_split_count = []
                        sample_batch_min_split.append(sbm)
                        sample_batch_min_split_count.append(sbm)
                # 最后一次别忘了~~我最开始就忘了^>
                sample_batch.append(sample_batch_min_split)

        # 下面根据bunchsize拼接最小batch比例单位的数据
        self.batch_list = []
        current_batch = []
        current_batch_count = []
        for batch_min in sample_batch:
            current_batch_count.extend(batch_min)
            current_maxsentframe = max(list(map(lambda x: x[1],current_batch_count)))
            total_frame = current_maxsentframe * len(current_batch_count)
            if total_frame < self.bunchsize:
                current_batch.extend(batch_min)
            else:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_count = []
                current_batch.extend(batch_min)
                current_batch_count.extend(batch_min)
        # 最后一次别忘了^^
        self.batch_list.append(current_batch)
        #print("is shuffle",self.shuffle)
        if self.shuffle:
            pass
            #print("befor shuffle",[len(bl) for bl in self.batch_list[-20:]],len(self.batch_list))
            #random.seed(self.random_seed)
            #random.shuffle(self.batch_list)
            #充分打乱
            #random.shuffle(self.batch_list)
            #print("after shuffle",[len(bl) for bl in self.batch_list[-20:]],len(self.batch_list))
        
    # 外部调用，获取batch，
    def getbatch(self):
        # 若读完一个cache，那就再读load一个
        if 0 == len(self.batch_list):
            
            if self.batch_ctrl:
                self.__shuffle_and_batch_multi_batch_ctrl()
            else:
                self.__shuffle_and_batch_multi()
            
            count_dc = {}
            for bl in self.batch_list:
                for seqid, seqlen, dl in bl:
                    if dl not in count_dc.keys():
                        count_dc[dl] = 1
                    else:
                        count_dc[dl] += 1
            print("*"*10)
            print("self.batch_list len count",len(self.batch_list))
            #print("seqlen_list len count",len(self.seqlen_list))
            print("cache len count",sum([v for k,v in count_dc.items()]))
            print("cache count",sorted(count_dc.items(),key=lambda x:x[0]))

            count_dc = {}
            for seqid, seqlen, dl in self.batch_list[0]:
                if dl not in count_dc.keys():
                    count_dc[dl] = 1
                else:
                    count_dc[dl] += 1
            
            print("frist batch count",sorted(count_dc.items(),key=lambda x:x[0]))

            count_dc = {}
            for seqid, seqlen, dl in self.batch_list[-1]:
                if dl not in count_dc.keys():
                    count_dc[dl] = 1
                else:
                    count_dc[dl] += 1
            
            print("last batch count",sorted(count_dc.items(),key=lambda x:x[0]))
        
        batch = self.batch_list.pop(0)
        batch_array = []
        batch_lab_array = []

        # 根据seq_id，组建数据batch，之前获取的是seq_id的batch
        for seqid, seqlen, dl in batch:
            aug_info = self.train_data_infos[dl]["aug_info"]
            if "lmdb" == self.train_data_infos[dl]["data_type"]:
                batch_one = [self.train_data_infos[dl]["lmdb_cache"][seqid],None,aug_info,dl]

            elif "pfile" == self.train_data_infos[dl]["data_type"]:
                pfile_raw_ = "wav" == self.train_data_infos[dl]["feature_type"] 
                if pfile_raw_:
                    if self.have_lab:
                        batch_one = [(self.train_data_infos[dl]["pfile_cache"][seqid],self.train_data_infos[dl]["pfile_lab_cache"][seqid]),None,aug_info,dl]
                    else:
                        batch_one = [(self.train_data_infos[dl]["pfile_cache"][seqid],None),None,aug_info,dl]
                else:
                    if self.have_lab:
                        batch_one = [None,(self.train_data_infos[dl]["pfile_cache"][seqid],self.train_data_infos[dl]["pfile_lab_cache"][seqid]),aug_info,dl]
                    else:
                        batch_one = [None,(self.train_data_infos[dl]["pfile_cache"][seqid],None),aug_info,dl]
                
            else:
                print("type error",dl,self.train_data_infos[dl]["data_type"])
                exit()

            batch_array.append(batch_one)
            
        
        return batch_array
        

########### add by ssyan2 end

if "__main__" == __name__:
    pfile_ = "/yrfs4/asrdictt/ssyan2/jingpinfangyan/sichuan/lib_supervised/newSRF_600/pfile_ce/fea.pfile.0"
    #pfile_ = "/yrfs4/asrdictt/ssyan2/jingpinfangyan/sichuan/lib_supervised/newSRF_600/pfile_ed/fea.pfile.0"
    pinfo1 = PfileInfo(pfile_)
    print(pinfo1)
    print("\n header_version",
    pinfo1.header_version,
    "\n header_size",
    pinfo1.header_size,
    "\n num_sentences",
    pinfo1.num_sentences,
    "\n num_frames",
    pinfo1.num_frames,
    "\n first_feature_column",
    pinfo1.first_feature_column,
    "\n dim_features",
    pinfo1.dim_features,
    "\n first_label_column",
    pinfo1.first_label_column,
    "\n dim_labels",
    pinfo1.dim_labels,
    "\n data_format",
    pinfo1.data_format,
    "\n frame_length",
    pinfo1.frame_length)

    loader1 = PfileChunkLoader(pinfo1,None,100,1000,cachesize=200)
    ret = loader1.get_cache()

    print(len(ret),len(loader1.seq_info))
    print(list(ret.keys())[:5])

    print(ret[list(ret.keys())[0]].shape,ret[list(ret.keys())[0]].reshape((-1,))[1000:1015])
    print()

    lmdb_p = "/yrfs4/asrdictt/ssyan2/jingpinfangyan/sichuan/lib_supervised/newSRF_600/lmdb_datas/lmdb0"
    linfo1 = LmdbInfo(lmdb_p)

    loader2 = LmdbChunkLoader(linfo1,100,1000,cachesize=200)
    ret2 = loader2.get_cache()
    datum = ret2[0][0]
    data1 = np.frombuffer(datum.anc.data, dtype=np.int16)
    print(data1.shape,data1[1000:1015])


