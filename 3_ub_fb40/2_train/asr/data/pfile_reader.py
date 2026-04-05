# modified from dlp's pfile.py
# by pcli2 recode by ssyan2

import os
import struct
import numpy as np
import random

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


class Pfileinfo():
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

########### add by ssyan2 start

class MultiPfileChunkReader():
    def __init__(self, pfileinfos, pfilelabinfos, start_sents, end_sents, mix_rates, bunchsize, maxsentframe=1000, maxnumsent=1000, nmod_pad=64, cachesize=20000, shuffle=True, random_seed=0, batch_ctrl=False):
        
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.batch_ctrl = batch_ctrl
        self.bunchsize = bunchsize
        self.have_lab = len(pfilelabinfos) == len(pfileinfos)

        self.pfile_caches = [None]*len(pfileinfos)
        self.labpfile_caches = [None]*len(pfileinfos)
        self.pfileinfos = pfileinfos
        self.pfilelabinfos = pfilelabinfos
        self.start_sents = start_sents
        self.end_sents = end_sents
        self.batch_rates = [round(mr/min(mix_rates)) for mr in mix_rates]
        self.mix_rates = [br/self.batch_rates[0] for br in self.batch_rates] if self.batch_ctrl else mix_rates
        

        self.batch_list = []

        #convt bunchsize to cachesize to control rate By ssyan2
        self.cachesizes = [round(cachesize*mr/sum(self.mix_rates)) for mr in self.mix_rates]
        
        print("multi pfile dataloader info: mix_rates-",self.mix_rates," ;cachesizes-",self.cachesizes)

        self.pfile_chunk_readers = [None]*len(pfileinfos)

        for index, pfileinfo1 in enumerate(pfileinfos):

            if self.have_lab:
                pfilelabinfo1 = pfilelabinfos[index]
            else:
                pfilelabinfo1 = None
            bunchsize1 = self.bunchsize
            cachesize1 = self.cachesizes[index]

            start_sent1, end_sent1 = self.start_sents[index], self.end_sents[index]

            print("multi pfile dataloader info: pfile_index-",index," ;sent_start&end-",start_sent1, end_sent1)

            pfile_chunk_reader = PfileChunkReader(pfileinfo1, pfilelabinfo1, start_sent1, end_sent1, bunchsize1, 
                                maxsentframe, maxnumsent, nmod_pad, cachesize1, shuffle, random_seed+index)
            self.pfile_chunk_readers[index] = pfile_chunk_reader
            
    
    def __shuffle_and_batch_multi(self):

        for ind, _ in enumerate(self.pfile_chunk_readers):
            #print("__shuffle_and_batch_multi",index,len(self.pfile_chunk_readers))
            ret_cache = self.pfile_chunk_readers[ind].get_cache()
            
            if self.have_lab:
                self.pfile_caches[ind],self.labpfile_caches[ind] = ret_cache
            else:
                self.pfile_caches[ind] = ret_cache

        seqlen_list = []

        for ind, pfile_cache in enumerate(self.pfile_caches):
            for key in pfile_cache:
                seqlen_list.append([key, pfile_cache[key].shape[0], ind])
        
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

        for ind, _ in enumerate(self.pfile_chunk_readers):
            #print("__shuffle_and_batch_multi",index,len(self.pfile_chunk_readers))
            ret_cache = self.pfile_chunk_readers[ind].get_cache()
            if self.have_lab:
                self.pfile_caches[ind],self.labpfile_caches[ind] = ret_cache
            else:
                self.pfile_caches[ind] = ret_cache
        
        seqlen_list = []

        for ind, pfile_cache in enumerate(self.pfile_caches):
            for key in pfile_cache:
                seqlen_list.append([key, pfile_cache[key].shape[0], ind])
        
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
        
    def get_all_index(self):

        index_dic = {}
        for ind, _ in enumerate(self.pfile_chunk_readers):
            #print("__shuffle_and_batch_multi",index,len(self.pfile_chunk_readers))
            index_dic[ind] = self.pfile_chunk_readers[ind].get_index()

            print("index count ", ind, index_dic[ind])
            


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
        lab_batch_array = []

        for seqid, seqlen, index in batch:

            batch_array.append(self.pfile_caches[index][seqid])
            if self.have_lab:
                lab_batch_array.append(self.labpfile_caches[index][seqid])
        if self.have_lab:
            return batch_array, lab_batch_array
        else:
            return batch_array

########### add by ssyan2 end



    


