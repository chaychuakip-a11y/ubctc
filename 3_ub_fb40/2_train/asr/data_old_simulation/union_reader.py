# modified from dlp's pfile.py
# by ssyan2

import os
import numpy as np
import random
import copy
import lmdb
import delta
import struct

try:
    from .datum_pb2 import SpeechDatum #Float MC

except:
    from datum_pb2 import SpeechDatum #Float MC

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


class LmdbInfo():

    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.seq_info = self.__read_seq_info__()
        self.lmdb_data = self.__read_lmdb__()
        self.num_sentences = len(self.seq_info)

    def __read_lmdb__(self):
        lmdb_data = lmdb.Environment(self.file_dir, readonly=True, readahead=True, lock=False)
        return lmdb_data
    

    def __read_seq_info__(self):
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
    
    @staticmethod
    def get_lmdb_item(txn,ind_str):
        with txn.cursor() as cursor:
            k = str(ind_str).zfill(12).encode('utf-8')
            cursor.set_key(k)
            datum_ = SpeechDatum()
            datum_.ParseFromString(cursor.value())

        return datum_

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

class PfileInfo():
    def __init__(self, filename):
        assert os.path.isfile(filename), "file not exists: {}".format(filename)
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
        assert self.num_sentences + 1 == len(self.seq_start_pos), "pfile tail is not correct"
        assert self.num_frames == self.seq_start_pos[-1], "frame number {0} mismatch {1} frames in header".format(self.seq_start_pos[-1], self.num_frames)
        self.seq_info = []
        for i in range(self.num_sentences):
            self.seq_info.append([i, self.seq_start_pos[i], self.seq_start_pos[i+1] - self.seq_start_pos[i]])

    def __del__(self):
        self.fp.close()

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

class PfileChunkReader():
    def __init__(self, pfileinfo, pfilelabinfo, start_sent, end_sent, bunchsize, maxsentframe=1000, maxnumsent=1000, nmod_pad=64, cachesize=10000, shuffle=True, random_seed=0):
        # pcli2: assume pfileinfo checking is done
        self.pfileinfo = pfileinfo
        self.pfilelabinfo = pfilelabinfo
        self.batch_list = []
        self.pfile_cache = None
        self.pfile_name = pfileinfo.filename
        self.bunchsize = bunchsize
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.seq_info = pfileinfo.seq_info[start_sent:end_sent]
        self.seq_length = len(self.seq_info)
        self.cachesize = cachesize
        self.cache_start_index = 0
        self.cache_end_index = cachesize if cachesize <= self.seq_length else self.seq_length - 1
        self.fp = open(self.pfile_name, "rb")

        if self.pfilelabinfo is not None:
            self.pfilelab_cache = None
            self.pfilelab_name = pfilelabinfo.filename
            self.labseq_info = pfilelabinfo.seq_info[start_sent:end_sent]
            self.lab_fp = open(self.pfilelab_name, "rb")

        
    def __make_cache(self):
        self.pfile_cache = self.__cache_into_dict(self.pfileinfo, self.seq_info, self.cache_start_index, self.cache_end_index, self.fp)
        if self.pfilelabinfo is not None:
            self.labpfile_cache = self.__cache_into_dict(self.pfilelabinfo, self.labseq_info, self.cache_start_index, self.cache_end_index, self.lab_fp)
        

    def __cache_into_dict(self, pfileinfo, seq_info, cache_start_index, cache_end_index, fp):
        if pfileinfo.data_format[2] == 'f':
            dtype = np.dtype("float32")
        if pfileinfo.data_format[2] == 'i':
            dtype = np.dtype("int32")
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

    
        

    def __shuffle_and_batch(self):
        self.__make_cache()
        self.cache_start_index = self.cache_end_index if self.cache_end_index < self.seq_length - 1 else 0
        self.cache_end_index = min(self.cache_start_index + self.cachesize, self.seq_length - 1)
        seqlen_list = []
        for key in self.pfile_cache:
            seqlen_list.append([key, self.pfile_cache[key].shape[0]])
        if self.maxnumsent > 1:
            ################# add little random sort by ssyan2
            #print("################# add little random sort by ssyan2")
            noisy_num = 3
            seq_len_noisy = [e[1] + random.randint(0,noisy_num) for e in seqlen_list]

            #seqlen_list = sorted(seqlen_list, key=lambda e: e[1])

            #print(seqlen_list[:5],type(seqlen_list),len(seqlen_list))
            seqlen_list = sorted(seqlen_list, key=lambda e: seq_len_noisy[seqlen_list.index(e)])
            #print(seqlen_list[:5],type(seqlen_list),len(seqlen_list))
            ################# add little random sort by ssyan2
            #seqlen_list = sorted(seqlen_list, key=lambda e: e[1])

        current_batch = []
        current_batch_sent = 0
        current_maxsentframe = 0
        for seqid, seqlen in seqlen_list:
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
                current_batch.append(seqid)
            else:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch.append(seqid)
                current_batch_sent = 1
            if self.maxnumsent <= current_batch_sent:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_sent = 0
                current_maxsentframe = 0
        if self.shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.batch_list)

    ########### add by ssyan2 start
    def get_cache(self):
        self.__make_cache()
        if self.cache_start_index >= self.seq_length - 1:
            print("circle now",self.cache_start_index,self.seq_length)
        self.cache_start_index = self.cache_end_index if self.cache_end_index < self.seq_length - 1 else 0
        self.cache_end_index = min(self.cache_start_index + self.cachesize, self.seq_length - 1)
        
        
        if self.pfilelabinfo is not None:
            return self.pfile_cache, self.labpfile_cache 
            
        else:
            return self.pfile_cache
    ########### add by ssyan2 end


    def get_index(self):
        return self.cache_start_index, self.cache_end_index, self.seq_length 

    def getbatch(self):
        if not self.batch_list:
            self.__shuffle_and_batch()
        batch = self.batch_list.pop(0)
        batch_array = []
        lab_batch_array = []
        for seqid in batch:
            batch_array.append(self.pfile_cache[seqid])
            if self.pfilelabinfo is not None:
                lab_batch_array.append(self.labpfile_cache[seqid])
        if self.pfilelabinfo is not None:
            return batch_array, lab_batch_array
        else:
            return batch_array
        



    def __del__(self):
        self.fp.close()
        if self.pfilelabinfo is not None:
            self.lab_fp.close()

class MultiUnionChunkReader():
    def __init__(self, work_data_infos, bunchsize=16000, maxsentframe=1000, maxnumsent=1000, nmod_pad=64, cache_sent_num=10000, shuffle=True, random_seed=0, batch_ctrl=False):
        
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.batch_ctrl = batch_ctrl
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
        


        """
        self.lmdb_infos = lmdb_infos
        self.start_sents = start_sents
        self.end_sents = end_sents
        self.batch_rates = [round(mr/min(mix_rates)) for mr in mix_rates]
        self.mix_rates = [br/self.batch_rates[0] for br in self.batch_rates] if self.batch_ctrl else mix_rates
        """
        

        

        #convt bunchsize to cache_sent_num to control rate By ssyan2
        self.cache_sent_nums = [round(cache_sent_num*mr/sum(self.mix_rates)) for mr in self.mix_rates]
        
        print("multi pfile dataloader info: mix_rates-",self.mix_rates," ;cache_sent_nums-",self.cache_sent_nums)

        for ind, dl in enumerate(self.data_list):
            self.train_data_infos[dl]["cache_sent_num"] = self.cache_sent_nums[ind]
            if "lmdb" == self.train_data_infos[dl]["data_type"]:
                self.train_data_infos[dl]["lmdb_data_info"] = LmdbInfo(self.train_data_infos[dl]["lmdb_file"])
                start_index_, end_index_ = self.train_data_infos[dl]["start_index"],self.train_data_infos[dl]["end_index"]
                split_seq_info_ = self.train_data_infos[dl]["lmdb_data_info"].seq_info[start_index_:end_index_]
                self.split_seq_infos[dl] = split_seq_info_
                self.train_data_infos[dl]["lmdb_chunk_start"] = 0
                self.train_data_infos[dl]["lmdb_cache"] = None

            elif "pfile" == self.train_data_infos[dl]["data_type"]:
                self.train_data_infos[dl]["pfile_fea_info"] = PfileInfo(self.train_data_infos[dl]["pfile_fea"])
                self.train_data_infos[dl]["pfile_lab_info"] = PfileInfo(self.train_data_infos[dl]["pfile_lab"])
                pfileinfo_, pfilelabinfo_ = self.train_data_infos[dl]["pfile_fea_info"], self.train_data_infos[dl]["pfile_lab_info"]
                start_index_, end_index_ = self.train_data_infos[dl]["start_index"],self.train_data_infos[dl]["end_index"]
                cachesize_ = self.train_data_infos[dl]["cache_sent_num"]
                bunchsize_ = self.bunchsize
                # self.train_data_infos[dl]["pfile_fea_info"],self.train_data_infos[dl]["pfile_lab_info"]
                pfile_chunk_reader_ = PfileChunkReader(pfileinfo_, pfilelabinfo_, start_index_, end_index_, bunchsize_, 
                                maxsentframe, maxnumsent, nmod_pad, cachesize_, shuffle, random_seed)
                self.train_data_infos[dl]["pfile_chunk_reader"] = pfile_chunk_reader_

            else:
                print("type error",dl)
                exit()
                
        """
        self.lmdb_chunk_starts = [0]*len(lmdb_infos)
        self.lmdb_caches = [None]*len(lmdb_infos)
        self.seq_lists = [None]*len(lmdb_infos)

        self.split_seq_infos = [self.lmdb_infos[ind].seq_info[self.start_sents[ind]:self.end_sents[ind]] for ind in range(len(lmdb_infos))]
        """
        self.aug_infos = {}
        #print("self.train_data_infos[dl]",self.train_data_infos)
        
        for ind, dl in enumerate(self.data_list):
            if "aug_info_str" in self.train_data_infos[dl].keys():
                aug_info_str_ = self.train_data_infos[dl]["aug_info_str"]
                aug_info = self.__get_aug_info(aug_info_str_)
            else: 
                aug_info = {}
            self.train_data_infos[dl]["aug_info"] = aug_info
            self.aug_infos[dl] = aug_info

        

        self.batch_list = []

    
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

    def __get_lmdb_cache(self,dl):
        start_ = self.train_data_infos[dl]["lmdb_chunk_start"]
        cache_sent_num_ = self.train_data_infos[dl]["cache_sent_num"]
        end_ = min(start_+cache_sent_num_,len(self.split_seq_infos[dl])-1)
        seq_list = self.split_seq_infos[dl][start_:end_]

        if end_ >= len(self.split_seq_infos[dl])-1:
            start_ = 0
            end_ = self.train_data_infos[dl]["cache_sent_num"] - len(seq_list)
            seq_list += self.split_seq_infos[dl][start_:end_]
        
        self.train_data_infos[dl]["lmdb_chunk_start"] = end_

        print(start_,end_,len(self.split_seq_infos[dl])-1)
        
        txn = self.train_data_infos[dl]["lmdb_data_info"].lmdb_data.begin()
        
        datums = [LmdbInfo.get_lmdb_item(txn,sl[1]) for sl in seq_list]

        return datums, seq_list

    
    def __get_cache(self):

        for ind, dl in enumerate(self.data_list):
            
            if "lmdb" == self.train_data_infos[dl]["data_type"]:
                datums_, seq_list_ = self.__get_lmdb_cache(dl)
                self.train_data_infos[dl]["lmdb_cache"] = datums_
                self.train_data_infos[dl]["seq_list"] = seq_list_

            elif "pfile" == self.train_data_infos[dl]["data_type"]:
                ret_cache = self.train_data_infos[dl]["pfile_chunk_reader"].get_cache()
            
                if self.train_data_infos[dl]["pfile_lab_info"]:
                    pfile_cache_, labpfile_cache_ = ret_cache
                else:
                    pfile_cache_ = ret_cache
                    labpfile_cache_ = None
                self.train_data_infos[dl]["pfile_cache"] = pfile_cache_
                self.train_data_infos[dl]["pfile_lab_cache"] = labpfile_cache_
                

            else:
                print("type error",dl)
                exit()


    def __get_seqlen_list(self):

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

    
    def __shuffle_and_batch_multi(self):

        self.__get_cache()

        seqlen_list = self.__get_seqlen_list()

        random.shuffle(seqlen_list)

        if self.maxnumsent > 1:
            #print("################# add little random sort by ssyan2")
            noisy_num = 2
            seq_len_noisy = [e[1] + random.randint(0,noisy_num) for e in seqlen_list]
            seqlen_list = sorted(seqlen_list, key=lambda e: seq_len_noisy[seqlen_list.index(e)])

        #print("multi pfile dataloader info: seqlen_list-",seqlen_list[1000:1020],len(seqlen_list))

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
            random.seed(self.random_seed)
            random.shuffle(self.batch_list)
            random.shuffle(self.batch_list)
        
    def __shuffle_and_batch_multi_batch_ctrl(self):

        self.__get_cache()

        seqlen_list = self.__get_seqlen_list()
        
        random.shuffle(seqlen_list)

        if self.maxnumsent > 1:
            #print("################# add little random sort by ssyan2")
            noisy_num = 2
            seq_len_noisy = [e[1] + random.randint(0,noisy_num) for e in seqlen_list]
            seqlen_list = sorted(seqlen_list, key=lambda e: seq_len_noisy[seqlen_list.index(e)])

        #print("multi pfile dataloader info: seqlen_list-",seqlen_list[1000:1020],len(seqlen_list))

        sample_num = len(list(filter(lambda x:self.data_list[0]==x[2],seqlen_list)))//self.batch_rates[0]
        sample_batch = []
        for inx in range(sample_num):
            sample_batch_min = []
            for ind,dl_name in enumerate(self.data_list):
                batch_num = self.batch_rates[ind]

                try:
                    sample_one = list(filter(lambda x:dl_name==x[2],seqlen_list))[inx*batch_num:inx*batch_num+batch_num]
                except:
                    # 超出了从尾部取，因为一般都是后面不够
                    sample_one = list(filter(lambda x:dl_name==x[2],seqlen_list))[-1*batch_num:]
                sample_batch_min.extend(sample_one)
            
            current_maxsentframe = max(list(map(lambda x: x[1],sample_batch_min)))
            total_frame = current_maxsentframe * len(sample_batch_min)
            if total_frame < self.bunchsize:
                sample_batch.append(sample_batch_min)
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
            #print("befor shuffle",[len(bl) for bl in self.batch_list[-20:]],len(self.batch_list))
            random.seed(self.random_seed)
            random.shuffle(self.batch_list)
            #充分打乱
            random.shuffle(self.batch_list)
            #print("after shuffle",[len(bl) for bl in self.batch_list[-20:]],len(self.batch_list))
        

    def getbatch(self):

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

        for seqid, seqlen, dl in batch:
            aug_info = self.train_data_infos[dl]["aug_info"]
            if "lmdb" == self.train_data_infos[dl]["data_type"]:
                batch_one = [self.train_data_infos[dl]["lmdb_cache"][seqid],None,aug_info,dl]

            elif "pfile" == self.train_data_infos[dl]["data_type"]:
            
                if self.have_lab:
                    batch_one = [
                        self.train_data_infos[dl]["pfile_cache"][seqid],
                        self.train_data_infos[dl]["pfile_lab_cache"][seqid],
                        aug_info,dl]
                else:
                    batch_one = [
                        self.train_data_infos[dl]["pfile_cache"][seqid],None,aug_info,dl]
                

            else:
                print("type error",dl,self.train_data_infos[dl]["data_type"])
                exit()

            batch_array.append(batch_one)
            
            
        
        return batch_array
        

########### add by ssyan2 end



