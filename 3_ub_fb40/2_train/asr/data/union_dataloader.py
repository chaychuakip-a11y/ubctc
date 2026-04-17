# by pcli2, innovated by dlp/jieding's source code
import torch
import numpy as np
import random
try:
    import delta
except ImportError:
    delta = None
import copy

from torch.utils.data import DataLoader, Dataset
try:
    from .union_reader import MultiUnionChunkReader, LmdbInfo, PfileInfo, Normfile
except:
    from union_reader import MultiUnionChunkReader, LmdbInfo, PfileInfo, Normfile

# add by mzwang7, for read txt
# txt部分与之前一致，未做改动。
def collate_fn_txt(batch):
    return batch[0]

# add by mzwang7, for read txt
def txtDataLoader(fp_txt, fp_dict, batch_size, start_sent, end_sent, ndivide=1, divide_index=0, shuffle_batch=True, num_workers=1):
    pfile_dataset = txtDataset(fp_txt, fp_dict, batch_size, start_sent, end_sent, ndivide, divide_index, shuffle_batch)
    return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn_txt)

# add by mzwang7, for read txt
class txtDataset(Dataset):
    def __init__(self, fp_txt, fp_dict, batch_size, start_sent, end_sent, ndivide, divide_index, shuffle_batch=True):
        self.batch_size=batch_size
        self.w2i = {}
        
        with open(fp_dict, "r", encoding="GBK") as f:
            for i, l in enumerate(f):
                self.w2i[l.strip()] = i
                
        with open(fp_txt) as f:
            self.data=list(f.readlines())
        
        end_sent = min(len(self.data), end_sent)
        num_sentences = (end_sent - start_sent)// ndivide
        self.batch_num=num_sentences//batch_size
        self.read_startindex = num_sentences  * divide_index + start_sent
        self.read_endindex = self.read_startindex + num_sentences if divide_index < ndivide - 1 else end_sent
        if shuffle_batch:random.shuffle(self.data[self.read_startindex:self.read_endindex])
        
        
    def __getitem__(self, index):
        x=[]
        y=[]
        maxl=0
        sos=self.w2i['<s>']
        eos=self.w2i['</s>']
        unk=self.w2i['<unk>']
        for i in range(self.read_startindex+index*self.batch_size,min(self.read_startindex+(index+1)*self.batch_size,self.read_endindex)):
            wi=[]
            for w in self.data[i].strip().split():
                if w in self.w2i:
                    wi.append(self.w2i[w])
                else:
                    wi.append(unk)
            xi=wi.copy()
            yi=wi.copy()
            xi.insert(0,sos)
            yi.append(eos)
            x.append(xi)
            y.append(yi)
            if len(xi)>maxl:maxl=len(xi)
        x_pad=np.full((self.batch_size,maxl),-1)
        y_pad=np.full((self.batch_size,maxl),-1)
        mask=np.full((self.batch_size,maxl),0)
        for i in range(len(x)):
            l=len(x[i])
            x_pad[i,:l]=x[i]
            mask[i,:l]=1
            y_pad[i,:l]=y[i]
        return torch.from_numpy(x_pad.T).float().contiguous(), torch.from_numpy(mask.T).unsqueeze(2).float().contiguous(), torch.from_numpy(y_pad.T).float().contiguous()

    def __len__(self):
        return self.batch_num

"""
DataLoader配置函数
用以根据ndivide即GPU数目拆分数据集
拆分好的数据传入UnionDataset构建数据集
并传入multi_worker_init_fn来指明数据读取方式
在train.py，数据集定义时调用
"""

def UnionDataLoader(data_infos, batch_num, bunchsize, maxsentframe, maxnumsent, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=True, num_workers=1, cachesize=10000, random_seed=0, batch_ctrl=False, val=False):
    """
    data_infos = {
        "file_norm":"00",
        "data_list":["d0","d1","d2"],
        "d0":{
            "data_type":"pfile",
            "pfile_fea":"d0-0f",
            "pfile_lab":"d0-0l",
            "lmdb_data":"d0-0d",
            "aug_info" : {},
            "start_sent":0,
            "end_sent":10000,
            "mix_rate":1.,
            },
        
        }
    """
    
    split_data_infos = copy.deepcopy(data_infos)
    print("split_data_infos",split_data_infos)
    for dl in data_infos["data_list"]:
        data_info = data_infos[dl]
        #print(dl,data_info["data_type"],data_info)
        if "lmdb" == data_info["data_type"]:
            read_data_info = LmdbInfo(data_info["lmdb_file"])
            #split_data_infos[dl]["lmdb_data_info"] = read_data_info
        elif "pfile" == data_info["data_type"]:
            read_data_info = PfileInfo(data_info["pfile_fea"])
            #split_data_infos[dl]["pfile_fea_info"] = read_data_info
            if "" != data_info["pfile_lab"]:
                read_lab_info = PfileInfo(data_info["pfile_lab"])
                #split_data_infos[dl]["pfile_lab_info"] = read_lab_info
                assert read_data_info.num_sentences == read_data_info.num_sentences, "train pfile {%s} number of sentences in feature and label are not equal"%dl
            else:
                #split_data_infos[dl]["pfile_lab_info"] = None
                pass
        else:
            print("data type error",data_info["data_type"])
            exit()

        start_sent_ = data_info["start_sent"]
        end_sent_ = min(read_data_info.num_sentences, data_info["end_sent"])
        sentence_per_gpu_ =  round((end_sent_ - start_sent_) / ndivide) 
        read_startindex = start_sent_ + sentence_per_gpu_ * divide_index
        read_endindex = min(read_startindex + sentence_per_gpu_ , end_sent_)
        assert start_sent_ < read_data_info.num_sentences , "train data %s start_sent must smaller than total sentences %d"%(dl,read_data_info.num_sentences)
        split_data_infos[dl]["start_index"] = read_startindex
        split_data_infos[dl]["end_index"] = read_endindex
    
    union_dataset = UnionDataset(split_data_infos, batch_num, bunchsize, maxsentframe, maxnumsent, nmod_pad, shuffle_batch, cachesize, random_seed, batch_ctrl, val)
    
    return DataLoader(union_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn, worker_init_fn=multi_worker_init_fn)

"""
在线仿真功能类
定义了在线仿真操作
UnionDataset实例化及执行仿真功能
"""
class AugOnline():

    def __init__(self,seed=314,aug_sets={}):

        lmdb_reverbfile = aug_sets["reverb"]
        # lmdbnoise_paths = aug_sets["reverb_noises"]
        # lmdbreverb_noise_paths = aug_sets["only_noises"]
        lmdbnoise_paths = aug_sets["only_noises"]
        lmdbreverb_noise_paths = aug_sets["reverb_noises"]
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed*3)

        # 初始化
        self.lmdb_reverbfile = ""
        #if lmdb_reverbfile != "" else "/work1/asrdictt/ssyan2/work_dir/multi_input_prjs/0_data_process/reverb_npy/reverb_pak_real_135m.npy"
        self.lmdbnoise_paths = lmdbnoise_paths #if lmdbnoise_paths != "" else "/yrfs4/asrdictt/ssyan2/jingpinfangyan/sichuan/noise/dev.list.reverb_music_noise/lmdb_datas/lmdb0, /yrfs4/asrdictt/ssyan2/jingpinfangyan/sichuan/noise/dev.list.reverb_tv_noise/lmdb_datas/lmdb0"
        self.lmdbnoise_paths = self.lmdbnoise_paths.replace(" ","").split(",")

        self.lmdbreverb_noise_paths = ""
        #if lmdbreverb_noise_paths != "" else "/yrfs4/asrdictt/ssyan2/jingpinfangyan/sichuan/noise/dev.list.reverb_music_noise/lmdb_datas/lmdb0, /yrfs4/asrdictt/ssyan2/jingpinfangyan/sichuan/noise/dev.list.reverb_tv_noise/lmdb_datas/lmdb0"
        self.lmdbreverb_noise_paths = self.lmdbreverb_noise_paths.replace(" ","").split(",")
        
        self.lmdb_noise_infos = None
        self.lmdb_reverb_noise_infos = None
        

        # 单通道仿真参数配置
        self.reverb_data = None
        self.noise_snr = [5,10,15,20]#[3, 3, 5, 5, 5] + list(range(6, 21)) my @snr = (0,5,10,15,20);  #SET
        self.amp_list = list(range(1000, 20000, 200))

        self.rng.shuffle(self.noise_snr)
        #print("noise_snr:",self.noise_snr)
        self.rng.shuffle(self.amp_list)
    
    # 混响
    def __do_reverb(self, wav_data):
        if None == self.reverb_data:
            self.reverb_data = list(np.load(self.lmdb_reverbfile, allow_pickle=True))

        data_list = [wav_data]
        addreverb_list = delta.conv_reverb(data_list, self.reverb_data, random_seed=self.rng.integers(1, 5e2), num_thread=8)
        aug_data = addreverb_list[0]
        data_remove = False
        return aug_data, data_remove
    
    # 混响+加噪
    def __do_reverb_noise(self, wav_data):
        # print("do reverb noise")
        if None == self.reverb_data:
            self.reverb_data = list(np.load(self.lmdb_reverbfile, allow_pickle=True))
        if  None == self.lmdb_reverb_noise_infos:
            self.lmdb_reverb_noise_infos = [LmdbInfo(lmdb_path) for lmdb_path in self.lmdbreverb_noise_paths]

        data_list = [wav_data]
        addreverb_list = delta.conv_reverb(data_list, self.reverb_data, random_seed=self.rng.integers(1, 5e2), num_thread=8)
        
        data_list = addreverb_list
        lmdb_reverb_noise_info_choice = self.rng.choice(self.lmdb_reverb_noise_infos)
        datanoise_index = [ind_str for _,ind_str,_ in self.rng.choice(lmdb_reverb_noise_info_choice.seq_info,len(data_list))]

        datanoise_list = self.__readLmdbnoise__(lmdb_reverb_noise_info_choice.lmdb_data.begin(),datanoise_index)
        addnoise_list = delta.addnoise(data_list, datanoise_list, self.noise_snr, random_seed=self.rng.integers(1, 5e2), num_thread=8)
       # print(addnoise_list)
        aug_data = addnoise_list[0]
        data_remove = False

        return aug_data, data_remove
    
    def __readLmdbnoise__(self, lmdbname, index_list):
        data_out  = []
        for il in index_list:
            datum = LmdbInfo.get_lmdb_item(lmdbname,il)
            data = np.frombuffer(datum.anc.data, dtype = np.int16)
            data_out.append(data)
        return data_out

    # 加噪
    def __do_noise(self, wav_data):
        if None == self.lmdb_noise_infos:
            self.lmdb_noise_infos = [LmdbInfo(lmdb_path) for lmdb_path in self.lmdbnoise_paths]
            
        data_list = [wav_data]
        lmdb_noise_info_choice = self.rng.choice(self.lmdb_noise_infos)
        datanoise_index = [ind_str for _,ind_str,_ in self.rng.choice(lmdb_noise_info_choice.seq_info,len(data_list))]
        datanoise_list = self.__readLmdbnoise__(lmdb_noise_info_choice.lmdb_data.begin(),datanoise_index)
        addnoise_list = delta.addnoise(data_list, datanoise_list, self.noise_snr, random_seed=self.rng.integers(1, 5e2), num_thread=8)
        #print("addnooise_list:",addnoise_list)
        aug_data = addnoise_list[0]
        data_remove = False
        return aug_data, data_remove
    
    
    def __ampChangeSingleCh__(self, data_in, amp):

        data_remove = False
        data_array = np.array(data_in, dtype=np.float32)
        data_sort = np.partition(data_array, -5)
        amp_max = data_sort[-5]
        if amp_max <= 100:
            data_remove = True
        data_out = np.array(data_array/amp_max*amp, dtype=np.int16)

        return data_out, data_remove

    # 幅值变换
    def __do_amp(self, wav_data):
        amp = self.rng.choice(self.amp_list)
        aug_data, data_remove = self.__ampChangeSingleCh__(wav_data, amp)
        
        return aug_data, data_remove

    # 根据概率计算是否做仿真
    def __get_aug_list(self, aug_info):

        aug_list = []
      #  print("aug_info:",aug_info)
        for k,v in aug_info.items():
            if v >= self.rng.random():
                #print(self.rng.random())
                aug_list.append(k)
        
        return aug_list
    # 仿真操作
    def do_aug(self, wav_data, aug_info={}, shuffle=True):
        aug_data = wav_data
        data_remove = False
        aug_list = self.__get_aug_list(aug_info)
        #if len(aug_list) > 0 : print(aug_list)
        if shuffle:
            random.shuffle(aug_list)
        else:
            aug_list = sorted(aug_list)

        #print("aug_list:",aug_list)
        for al in aug_list:
            if "reverb" == al:
                aug_data, data_remove = self.__do_reverb_noise(aug_data)
            elif "noise" == al:
                aug_data, data_remove = self.__do_noise(aug_data)
            elif "amp" == al:
                aug_data, data_remove = self.__do_amp(aug_data)
            else:
                pass
            if data_remove:
                break

        return aug_data, data_remove

"""
Dataset类
用以从chunk中读入一个batch原始数据

"""
class UnionDataset(Dataset):
    def __init__(self, split_data_infos, batch_num, bunchsize, maxsentframe, maxnumsent, nmod_pad=64, shuffle_batch=True, cachesize=10000, random_seed=0, batch_ctrl=False,val=False):
        self.split_data_infos = split_data_infos

        aug_infos = {}

        for dl in self.split_data_infos["data_list"]:
            data_info = self.split_data_infos[dl]
            if "lmdb" == data_info["data_type"]:
                if data_info["aug_info_str"]:
                    aug_infos[dl] = data_info["aug_info_str"]
            #
            elif "pfile" == data_info["data_type"] and "wav" == data_info["feature_type"]:
                if data_info["aug_info_str"]:
                    aug_infos[dl] = data_info["aug_info_str"]
        
        self.aug_infos = aug_infos
        if len(self.aug_infos.keys()) > 0 and "aug_sets" in self.split_data_infos.keys():
            self.aug_online = AugOnline(aug_sets=self.split_data_infos["aug_sets"])

        norm = Normfile(self.split_data_infos["file_norm"])
        self.mean = norm.mean
        self.var = norm.var

        self.bunchsize = bunchsize
        self.batch_num = batch_num
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle_batch = shuffle_batch
        self.cachesize = cachesize
        self.lmdb_chunk_reader = None
        self.random_seed = random_seed
        self.batch_ctrl = batch_ctrl
        self.val = val

    # 处理lmdb数据，输入音频，进行仿真及提取fb40特征，在__get_samples中调用
    def __deal_lmdb_data__(self,datum,aug={}):
        
        #print("datum",datum)
        wav_data     = np.frombuffer(datum.anc.data, dtype=np.int16)
        celabel    = np.frombuffer(datum.anc.state_data, dtype=np.int16).reshape(-1, 1)
        edlabel = np.frombuffer(datum.anc.ed_data, dtype=np.int16).reshape(-1, 1)
        
        if len(self.aug_infos.keys()) > 0 and not self.val:
            aug_data, data_remove = self.aug_online.do_aug(wav_data,aug_info=aug)
        else:
            aug_data, data_remove = wav_data, False
        
        data = delta.build_fb40(aug_data, num_thread=8)

        nframes = min(celabel.shape[0],data.shape[0])              
        data = data[:nframes, :]
        celabel = celabel[:nframes, :]
        
        
        return data, celabel, edlabel, data_remove
    
    # 处理lmdb数据，输入音频，进行仿真及提取fb40特征，在__get_samples中调用
    def __deal_pfile_data__(self,data_lab_raw,data_lab_fb40,aug={}):

        if data_lab_raw:
            wav_data, lab_data = data_lab_raw
            wav_data = wav_data.reshape((-1,)).astype(np.int16)
            if len(self.aug_infos.keys()) > 0 and not self.val:
                aug_data, data_remove = self.aug_online.do_aug(wav_data,aug_info=aug)
            else:
                aug_data, data_remove = wav_data, False
            data = delta.build_fb40(aug_data, num_thread=8)
            if 2 == lab_data.shape[1]:
                edlabel, celabel = np.split(lab_data, 2, axis=1)
            else:
                print("label error")
                exit()
            
            nframes = min(celabel.shape[0],data.shape[0])              
            data = data[:nframes, :]
            celabel = celabel[:nframes, :]
            edlabel = edlabel[:nframes, :]

            return data, celabel, edlabel, data_remove
        else:
            data, lab_data = data_lab_fb40
            if 2 == lab_data.shape[1]:
                edlabel, celabel = np.split(lab_data, 2, axis=1)
            elif 1 == lab_data.shape[1]:
                edlabel = None
                celabel = lab_data
            else:
                print("label error, lab_data.shape[1]!=2, =", lab_data.shape[1])
                exit()
            data_remove = False
            return data, celabel, edlabel, data_remove
            
    
    # 根据数据类别得到处理好的数据和标签
    def __get_samples(self,samples):
        datas = []
        labels = []
        label_ctcs = []
        
        for sample in samples:
            data_lab_raw, data_lab_fb40, aug_info, dl = sample
            data_type = self.split_data_infos[dl]["data_type"]
            # 若lmdb数据，通过__deal_lmdb_data__函数仿真及提取fb40特征，后归一化
            if "lmdb" == data_type :
                data_one, label_ctc_, label_ed_, data_remove_ = self.__deal_lmdb_data__(data_lab_raw,aug=aug_info)

            elif "pfile" == data_type:
                data_one, label_ctc_, label_ed_, data_remove_ = self.__deal_pfile_data__(data_lab_raw,data_lab_fb40,aug=aug_info)
                
            if not data_remove_:
                data_one = (data_one - self.mean) * self.var
                datas.append(data_one)
                label_ctcs.append(label_ctc_)
                labels.append(label_ed_)    
            
        return datas, labels, label_ctcs


    # 数据迭代函数，
    def __getitem__(self, index):
        # 读取原始数据
        read_samples = self.union_chunk_reader.getbatch()
        # 处理为特征数据
        datas, labels, label_ctcs = self.__get_samples(read_samples)
        # 将数据填充为self.nmod_pad倍数数据
        datas, data_mask = self.__pad_nmod(datas, self.nmod_pad, 0)

        if labels[0] is None:
            labels=[]
            for idx, element in enumerate(label_ctcs):
                e1 = element
                e1 = e1.squeeze()
                e1 = e1[np.insert(np.diff(e1).astype(np.bool), 0, True)]
                e1 = np.expand_dims(e1, 1)
                labels.append(e1)
            # print('labels: ', labels)
            # print('labels_ctc:', label_ctcs)
        labels, _ = self.__pad_nmod(labels, None, -1)
        
        meta = {}
        if label_ctcs:
            label_ctcs, label_ctc_mask = self.__pad_nmod(label_ctcs, self.nmod_pad, -1)
            # label_ctcs = label_ctcs.squeeze(1).squeeze(1)
            # label_ctcs = label_ctcs.transpose(1, 0)
            # label_ctc_lengths = label_ctc_mask.sum(1)
            meta["label_ce"] = label_ctcs
            meta["data_mask"] = data_mask
            meta["label_mask"] = label_ctc_mask

        # print(labels.shape, label_ctcs.shape);exit()    #torch.Size([50, 1, 1, 39]) torch.Size([50, 1, 1, 164])

        label_mask = labels.clone()
        label_mask[label_mask >= 0] = 1
        label_mask[label_mask < 0] = 0
        maxlen = label_mask.sum(3).max()
        labels = labels[:, :, :, :maxlen]
        label_mask = label_mask[:, :, :, :maxlen]
        labels = labels.squeeze(1).squeeze(1)
        labels = labels.transpose(1, 0)
        label_mask = label_mask.squeeze(1).squeeze(1)
        label_mask = label_mask.transpose(1, 0)
        meta["mask"] = data_mask.contiguous()
        meta["att_label"] = labels.float().contiguous()
        meta["att_mask"] = label_mask.float().contiguous()
        meta['w'] = torch.Tensor([datas.shape[3]])
        meta["rnn_mask"] = data_mask.transpose(1, 0).unsqueeze(2).contiguous()
        
        datas = datas.float()
        return datas, meta
    # 数据补齐为nmod整数倍，保证下采样时不会丢失信息，补充部分计算loss时会被mask掉。
    def __pad_nmod(self, sequence, nmod, val):
        maxlen = 0
        pad_list = []
        mask_list = []
        for nparray in sequence:
            if nparray.shape[0] > maxlen:
                maxlen = nparray.shape[0]
        if nmod is not None:
            maxlen = maxlen if maxlen % nmod == 0 else maxlen + nmod - maxlen % nmod
        for nparray in sequence:
            padlen = maxlen - nparray.shape[0]
            nparray = np.pad(nparray, ((0, padlen), (0, 0)), mode="constant", constant_values=(val,))
            nparray = nparray.transpose(1, 0)
            torcharray = torch.from_numpy(nparray)
            torchmask = torch.ones(1, torcharray.size()[1])
            if padlen > 0:
                torchmask[:, -padlen:] = 0
            torcharray = torcharray.reshape(1, 1, torcharray.size()[0], torcharray.size()[1])
            pad_list.append(torcharray)
            mask_list.append(torchmask)

        batch_array = torch.cat(pad_list, dim=0)
        batch_mask = torch.cat(mask_list, dim=0)
        return batch_array, batch_mask

    def __len__(self):
        return self.batch_num

"""
定义数据chunk_reader，根据num_workers拆分数据
"""
def multi_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    worker_idx = worker_info.id
    dataset = worker_info.dataset
    dataset.work_data_infos = copy.deepcopy(dataset.split_data_infos)

    for dl in dataset.split_data_infos["data_list"]:
        sentence_per_worker_ = round((dataset.split_data_infos[dl]["end_index"] - dataset.split_data_infos[dl]["start_index"]) / num_workers)
        dataset.work_data_infos[dl]["start_index"] = dataset.split_data_infos[dl]["start_index"] + sentence_per_worker_ * worker_idx
        dataset.work_data_infos[dl]["end_index"] = min(dataset.work_data_infos[dl]["start_index"] + sentence_per_worker_, dataset.split_data_infos[dl]["end_index"])
    
    dataset.union_chunk_reader = MultiUnionChunkReader(dataset.work_data_infos, dataset.bunchsize, dataset.maxsentframe, dataset.maxnumsent, 
                                                  dataset.nmod_pad, dataset.cachesize, dataset.shuffle_batch, dataset.random_seed, dataset.batch_ctrl,dataset.val)


def collate_fn(batch):
    data, meta = batch[0]
    return data, meta


if "__main__" == __name__:
    pass
    """
    UnionDataLoader(data_infos, batch_num, bunchsize, maxsentframe, maxnumsent, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=True, num_workers=2, cachesize=10000, random_seed=0, batch_ctrl=False, val=False)
    """