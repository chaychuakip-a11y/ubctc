# modified from dlp's pfile.py
# by ssyan2

import os
import numpy as np
import random
import lmdb
import delta

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
        self.mean = np.array([float(line[:-1]) for line in content[1:self.mean_dim+1]], dtype=np.float32)
        self.var = np.array([float(line[:-1]) for line in content[self.mean_dim+2:]], dtype=np.float32)



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

def get_lmdb_item(txn,ind_str):
    with txn.cursor() as cursor:
        k = str(ind_str).zfill(12).encode('utf-8')
        cursor.set_key(k)
        datum1 = SpeechDatum()
        datum1.ParseFromString(cursor.value())

    return datum1


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
        with open(get_lmdb_key(self.file_dir),"r") as fr:
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



class MultiLmdbChunkReader():
    def __init__(self, lmdb_infos, start_sents, end_sents, mix_rates, bunchsize=16000, maxsentframe=1000, maxnumsent=1000, nmod_pad=64, cachesize=10000, shuffle=True, random_seed=0, batch_ctrl=False,aug_infos={}):
        
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.batch_ctrl = batch_ctrl
        self.bunchsize = bunchsize
        self.have_lab = True

        
        self.lmdb_infos = lmdb_infos
        self.start_sents = start_sents
        self.end_sents = end_sents
        self.batch_rates = [round(mr/min(mix_rates)) for mr in mix_rates]
        self.mix_rates = [br/self.batch_rates[0] for br in self.batch_rates] if self.batch_ctrl else mix_rates
        

        self.batch_list = []

        #convt bunchsize to cachesize to control rate By ssyan2
        self.cachesizes = [round(cachesize*mr/sum(self.mix_rates)) for mr in self.mix_rates]
        
        print("multi pfile dataloader info: mix_rates-",self.mix_rates," ;cachesizes-",self.cachesizes)

        self.lmdb_chunk_starts = [0]*len(lmdb_infos)
        self.lmdb_caches = [None]*len(lmdb_infos)
        self.seq_lists = [None]*len(lmdb_infos)

        self.split_seq_infos = [self.lmdb_infos[ind].seq_info[self.start_sents[ind]:self.end_sents[ind]] for ind in range(len(lmdb_infos))]

        self.aug_keys = aug_infos


    def __get_cache(self,ind):
        start = self.lmdb_chunk_starts[ind]
        end = min(self.lmdb_chunk_starts[ind]+self.cachesizes[ind],len(self.split_seq_infos[ind])-1)
        seq_list = self.split_seq_infos[ind][start:end]

        if end >= len(self.split_seq_infos[ind])-1:
            start1 = 0
            end = self.cachesizes[ind] - len(seq_list)
            seq_list += self.split_seq_infos[ind][start1:end]
        
        self.lmdb_chunk_starts[ind] = end

        print(start,end,len(self.split_seq_infos[ind])-1)
        
        txn = self.lmdb_infos[ind].lmdb_data.begin()
        
        datums = [get_lmdb_item(txn,sl[1]) for sl in seq_list]

        return datums, seq_list




    
    def __shuffle_and_batch_multi(self):

        for ind, _ in enumerate(self.lmdb_infos):
            self.lmdb_caches[ind], self.seq_lists[ind] = self.__get_cache(ind)

        seqlen_list = []

        for ind, _ in enumerate(self.lmdb_infos):
            for index, key in enumerate(self.seq_lists[ind]):
                seqlen_list.append([index, key[2], ind])
        
        random.shuffle(seqlen_list)

        if self.maxnumsent > 1:
            #print("################# add little random sort by ssyan2")
            noisy_num = 2
            seq_len_noisy = [e[1] + random.randint(0,noisy_num) for e in seqlen_list]
            seqlen_list = sorted(seqlen_list, key=lambda e: seq_len_noisy[seqlen_list.index(e)])

        #print("multi pfile dataloader info: seqlen_list-",seqlen_list[1000:1020],len(seqlen_list))

        self.seqlen_list = seqlen_list

        current_batch = []
        current_batch_sent = 0
        current_maxsentframe = 0
        for seqid, seqlen, ind in self.seqlen_list:
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
                current_batch.append([seqid,seqlen,ind])
            else:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch.append([seqid,seqlen,ind])
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

        for ind, _ in enumerate(self.lmdb_infos):
            self.lmdb_caches[ind], self.seq_lists[ind] = self.__get_cache(ind)

        seqlen_list = []

        for ind, _ in enumerate(self.lmdb_infos):
            for index, key in enumerate(self.seq_lists[ind]):
                seqlen_list.append([index, key[2], ind])
        
        random.shuffle(seqlen_list)

        if self.maxnumsent > 1:
            #print("################# add little random sort by ssyan2")
            noisy_num = 2
            seq_len_noisy = [e[1] + random.randint(0,noisy_num) for e in seqlen_list]
            seqlen_list = sorted(seqlen_list, key=lambda e: seq_len_noisy[seqlen_list.index(e)])

        #print("multi pfile dataloader info: seqlen_list-",seqlen_list[1000:1020],len(seqlen_list))

        self.seqlen_list = seqlen_list

        sample_num = len(list(filter(lambda x:0==x[2],self.seqlen_list)))//self.batch_rates[0]
        sample_batch = []
        for inx in range(sample_num):
            sample_batch_min = []
            for ind,batch_num in enumerate(self.batch_rates):
                try:
                    sample_one = list(filter(lambda x:ind==x[2],self.seqlen_list))[inx*batch_num:inx*batch_num+batch_num]
                except:
                    # 超出了从尾部取，因为一般都是后面不够
                    sample_one = list(filter(lambda x:ind==x[2],self.seqlen_list))[-1*batch_num:]
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
            count_dc = {}
            if self.batch_ctrl:
                self.__shuffle_and_batch_multi_batch_ctrl()
            else:
                self.__shuffle_and_batch_multi()
            for bl in self.batch_list:
                for seqid, seqlen, index in bl:
                    if index not in count_dc.keys():
                        count_dc[index] = 1
                    else:
                        count_dc[index] += 1
            print("*"*10)
            print("self.batch_list len count",len(self.batch_list))
            print("seqlen_list len count",len(self.seqlen_list))
            print("cache len count",sum([v for k,v in count_dc.items()]))
            print("cache count",sorted(count_dc.items(),key=lambda x:x[0]))

            count_dc = {}
            for seqid, seqlen, index in self.batch_list[0]:
                if index not in count_dc.keys():
                    count_dc[index] = 1
                else:
                    count_dc[index] += 1
            
            print("frist batch count",sorted(count_dc.items(),key=lambda x:x[0]))

            count_dc = {}
            for seqid, seqlen, index in self.batch_list[-1]:
                if index not in count_dc.keys():
                    count_dc[index] = 1
                else:
                    count_dc[index] += 1
            
            print("last batch count",sorted(count_dc.items(),key=lambda x:x[0]))
        
        batch = self.batch_list.pop(0)
        batch_array = []

        for seqid, seqlen, index in batch:
            
            aug_info = self.aug_keys[index] if index in self.aug_keys.keys() else {}

            batch_array.append([self.lmdb_caches[index][seqid],aug_info])
            
        if self.have_lab:
            return batch_array
        

########### add by ssyan2 end

if "__main__" == __name__:
    lmdb_dir1 = "/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb_clean/lmdb1/"
    lmdb_dir2 = "/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb_clean/lmdb2/"
    lmdb_dir3 = "/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb_clean/lmdb3/"

    mix_rates = [1,0.5,1.2]
    bunchsize = 40000

    reader = MultiLmdbChunkReader(lmdb_infos = [LmdbInfo(lmdb_dir1),LmdbInfo(lmdb_dir2),LmdbInfo(lmdb_dir3)], start_sents = [0,0,0], end_sents = [40000,10000,20000], mix_rates = mix_rates, bunchsize=bunchsize,batch_ctrl=False)
    
    linfo = LmdbInfo(lmdb_dir1) 
    
    bunchsize1 = int(bunchsize*mix_rates[0]/sum(mix_rates))
    num_batch = linfo.estimate_num_batch(linfo.seq_info,bunchsize=bunchsize1, maxsentframe=1000, maxnumsent=1000, nmod_pad=64,start_end=[0,40000])

    print(linfo.num_sentences,num_batch)
    rets = []
    for i in range(200):
        if i % 100 == 0:
            print("#"*10,i)
        ret = reader.getbatch()
        #print(i,len(ret))
        rets.append(ret)
        
    print(len(rets[-1]))
    data = np.frombuffer(rets[-1][-1][0].anc.data, dtype=np.int16)

    datafb40 = delta.build_fb40(data, num_thread=8)

    print(linfo.num_sentences,num_batch)

    print(data.shape,datafb40.shape)

