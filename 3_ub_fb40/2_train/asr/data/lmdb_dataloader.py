# by pcli2, innovated by dlp/jieding's source code
import torch
import numpy as np
import random
import delta

from torch.utils.data import DataLoader, Dataset
from .lmdb_reader import LmdbInfo, MultiLmdbChunkReader, get_lmdb_item


def LmdbDataLoader(lmdb_files, file_norm, batch_num, bunchsize, maxsentframe, maxnumsent, start_sents, end_sents, mix_rates, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=True, num_workers=2, cachesize=10000, random_seed=0, batch_ctrl=False,aug_infos={}):

    lmdb_infos = [LmdbInfo(lmdb_files[ind]) for ind,lf in enumerate(lmdb_files)]
    

    
    
    num_sentences_s = [lmdb_infos[ind].num_sentences for ind,lf in enumerate(lmdb_files)]
    read_startindexs = [None for ind,lf in enumerate(lmdb_files)]
    read_endindexs = [None for ind,lf in enumerate(lmdb_files)]

    for ind,lf in enumerate(lmdb_files):
        
        assert start_sents[ind] < num_sentences_s[ind] , "lmdb index %d start_sent must smaller than total sentences %d"%(ind,num_sentences_s[ind])

        end_sents[ind] = min(num_sentences_s[ind], end_sents[ind])
        num_sentences_s[ind] = end_sents[ind] - start_sents[ind]
        read_startindexs[ind] = int(num_sentences_s[ind] / ndivide * divide_index) + start_sents[ind]
        read_endindexs[ind] = read_startindexs[ind] + int(num_sentences_s[ind] / ndivide) if divide_index < ndivide - 1 else end_sents[ind]

    
    
    lmdb_dataset = LmdbDataset(lmdb_files, file_norm, read_startindexs, read_endindexs, mix_rates, batch_num, bunchsize, maxsentframe, 
                                 maxnumsent, nmod_pad, shuffle_batch, cachesize, random_seed, batch_ctrl,aug_infos)
    
    return DataLoader(lmdb_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn, worker_init_fn=multi_worker_init_fn)


class AugOnline():

    def __init__(self,seed=314,lmdb_reverbfile="",lmdbnoise_paths=""):
        
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed*3)

        # 初始化
        self.lmdb_reverbfile     = lmdb_reverbfile if lmdb_reverbfile != "" else "/train8/asrkws/kaishen2/Data_kws/Reverb/reverb.npy"
        self.lmdbnoise_paths = lmdbnoise_paths if lmdbnoise_paths != "" else "/train8/asrkws/kaishen2/Data_kws/Universal_MC_interNoise/lmdb/lmdb0,/train8/asrkws/kaishen2/Data_kws/Universal_MC_diffuseNoise/lmdb/lmdb0"
        self.lmdbnoise_paths = self.lmdbnoise_paths.replace(" ","").split(",")
        
        self.lmdb_noise_infos = None
        

        # 单通道仿真参数配置
        self.reverb_data = None
        self.noise_snr = [3, 3, 5, 5, 5] + list(range(6, 21))
        self.amp_list = list(range(1000, 20000, 200))

        self.rng.shuffle(self.noise_snr)
        self.rng.shuffle(self.amp_list)
    
    def __do_reverb(self, wav_data):
        if self.reverb_data == None:
            self.reverb_data = list(np.load(self.lmdb_reverbfile, allow_pickle=True))

        data_list = [wav_data]
        addreverb_list = delta.conv_reverb(data_list, self.reverb_data, random_seed=self.rng.integers(1, 5e2), num_thread=8)
        aug_data = addreverb_list[0]
        data_remove = False
        return aug_data, data_remove

    def __readLmdbnoise__(self, lmdbname, index_list):
        data_out  = []
        for il in index_list:
            datum = get_lmdb_item(lmdbname,il)
            data = np.frombuffer(datum.anc.data, dtype = np.int16)
            data_out.append(data)
        return data_out

    def __do_noise(self, wav_data):
        if self.lmdb_noise_infos == None:
            self.lmdb_noise_infos = [LmdbInfo(lmdb_path) for lmdb_path in self.lmdbnoise_paths]
            
        data_list = [wav_data]
        lmdb_noise_info_choice = self.rng.choice(self.lmdb_noise_infos)
        datanoise_index = [ind_str for _,ind_str,_ in self.rng.choice(lmdb_noise_info_choice.seq_info,len(data_list))]
        #print("self.lmdb_noise_infos.index(lmdb_noise_info_choice)",self.lmdb_noise_infos.index(lmdb_noise_info_choice),datanoise_index)
        #print("lmdb_noise_info_choice.seq_info",lmdb_noise_info_choice.seq_info.index(datanoise_index[0]))
        datanoise_list = self.__readLmdbnoise__(lmdb_noise_info_choice.lmdb_data.begin(),datanoise_index)
        addnoise_list = delta.addnoise(data_list, datanoise_list, self.noise_snr, random_seed=self.rng.integers(1, 5e2), num_thread=8)
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

    def __do_amp(self, wav_data):
        amp = self.rng.choice(self.amp_list)
        aug_data, data_remove = self.__ampChangeSingleCh__(wav_data, amp)
        
        return aug_data, data_remove

    def __get_aug_list(self, aug_info):

        aug_list = []
        for k,v in aug_info.items():
            if v >= self.rng.random():
                aug_list.append(k)
        
        return aug_list

    def do_aug(self, wav_data, aug_info={}, shuffle=True):
        aug_data = wav_data
        data_remove = False
        aug_list = self.__get_aug_list(aug_info)
        #if len(aug_list) > 0 : print(aug_list)
        if shuffle:
            random.shuffle(aug_list)
        else:
            aug_list = sorted(aug_list)

        for al in aug_list:
            if "reverb" == al:
                aug_data, data_remove = self.__do_reverb(aug_data)
            elif "noise" == al:
                aug_data, data_remove = self.__do_noise(aug_data)
            elif "amp" == al:
                aug_data, data_remove = self.__do_amp(aug_data)
            else:
                pass
            if data_remove:
                break

        return aug_data, data_remove


class LmdbDataset(Dataset):
    def __init__(self, lmdb_files, lmdb_norm, start_indexs, end_indexs, mix_rates, batch_num, bunchsize, maxsentframe, maxnumsent, 
                 nmod_pad=64, shuffle_batch=True, cachesize=10000, random_seed=0, batch_ctrl=False, aug_infos={},):
        self.lmdb_files = lmdb_files
        with open(lmdb_norm, 'r') as nf:
            normfile_lines = nf.read().splitlines()
        self.mel_spec_mean = np.array([float(item) for item in normfile_lines[1:41]])
        self.mel_spec_std  = np.array([float(item) for item in normfile_lines[42:82]])

        self.bunchsize = bunchsize
        self.start_indexs = start_indexs
        self.end_indexs = end_indexs
        self.mix_rates = mix_rates
        self.batch_num = batch_num
        self.maxsentframe = maxsentframe
        self.maxnumsent = maxnumsent
        self.nmod_pad = nmod_pad
        self.shuffle_batch = shuffle_batch
        self.cachesize = cachesize
        self.lmdb_chunk_reader = None
        self.random_seed = random_seed
        self.batch_ctrl = batch_ctrl
        self.aug_infos = aug_infos
        if len(self.aug_infos.keys()) > 0:
            self.aug_online = AugOnline()

    def fb40_norm(self,wav_data):
        data = delta.build_fb40(wav_data, num_thread=8)
        data = (data - self.mel_spec_mean[None,:]) * self.mel_spec_std
        return data
        
    
    def __get_data__(self,datum,aug={}):
        
        #print("datum",datum)
        wav_data     = np.frombuffer(datum.anc.data, dtype=np.int16)
        celabel    = np.frombuffer(datum.anc.state_data, dtype=np.int16).reshape(-1, 1)
        edlable = np.frombuffer(datum.anc.ed_data, dtype=np.int16).reshape(-1, 1)

        

        #exit()
        
        if len(self.aug_infos.keys()) > 0:
            aug_data, data_remove = self.aug_online.do_aug(wav_data,aug_info=aug)
        else:
            aug_data, data_remove = wav_data, False
        
        
        data = self.fb40_norm(aug_data)
        #print("wav_data.shape",wav_data.shape,aug_data.shape,data.shape,celabel.shape,edlable.shape,edlable)
        #exit()
        nframes = min(celabel.shape[0],data.shape[0])              
        data = data[:nframes,:]
        #padzeros_mch = np.tile(self.padzeros_mch, (nframes, 1))
        #data = np.concatenate( (data, padzeros_mch), axis=1)
        label_out = celabel[:nframes, :]
        edlable_out = edlable
        
        return data, label_out, edlable_out, data_remove


    def __getitem__(self, index):
        
        datums = self.lmdb_chunk_reader.getbatch()
        #print("datums len",len(datums))
        data = []
        label = []
        label_ctc = []
        count = 0
        for datum, aug_info in datums:
            
            count += 1
            data1, label_ctc1,label_syl1, data_remove = self.__get_data__(datum,aug=aug_info)
            
            if not data_remove:
                data.append(data1)
                label_ctc.append(label_ctc1)
                label.append(label_syl1)
        
        
        data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)

        label, _ = self.__pad_nmod(label, None, -1)
        meta = {}
        if label_ctc:
            label_ctc, label_ctc_mask = self.__pad_nmod(label_ctc, self.nmod_pad, -1)
            label_ctc = label_ctc.squeeze(1).squeeze(1)
            label_ctc = label_ctc.transpose(1, 0)
            label_ctc_lengths = label_ctc_mask.sum(1)
            meta["label_ctc"] = label_ctc
            meta["data_mask"] = data_mask
            meta["label_mask"] = label_ctc_mask

        label_mask = label.clone()
        label_mask[label_mask >= 0] = 1
        label_mask[label_mask < 0] = 0
        maxlen = label_mask.sum(3).max()
        label = label[:, :, :, :maxlen]
        label_mask = label_mask[:, :, :, :maxlen]
        label = label.squeeze(1).squeeze(1)
        label = label.transpose(1, 0)
        label_mask = label_mask.squeeze(1).squeeze(1)
        label_mask = label_mask.transpose(1, 0)
        meta["mask"] = data_mask.contiguous()
        meta["att_label"] = label.float().contiguous()
        meta["att_mask"] = label_mask.float().contiguous()
        meta['w'] = torch.Tensor([data.shape[3]])
        meta["rnn_mask"] = data_mask.transpose(1, 0).unsqueeze(2).contiguous()
        """
        torch.Size([54, 1, 40, 88])
        label_ctc torch.Size([88, 54])
        data_mask torch.Size([54, 88])
        label_mask torch.Size([54, 88])
        mask torch.Size([54, 88])
        att_label torch.Size([5, 54])
        att_mask torch.Size([5, 54])
        w torch.Size([1])
        rnn_mask torch.Size([88, 54, 1])

        torch.Size([50, 1, 40, 96])
        label_ctc torch.Size([96, 50])
        data_mask torch.Size([50, 96])
        label_mask torch.Size([50, 96])
        mask torch.Size([50, 96])
        att_label torch.Size([9, 50])
        att_mask torch.Size([9, 50])
        w torch.Size([1])
        rnn_mask torch.Size([96, 50, 1])
        """
        data = data.float()
        return data, meta

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


def multi_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    worker_idx = worker_info.id
    dataset = worker_info.dataset
    dataset.lmdb_files = [LmdbInfo(dlf) for dlf in dataset.lmdb_files]
    start_indexs = dataset.start_indexs
    end_indexs = dataset.end_indexs
    sentence_per_workers = [int((end_indexs[ind] - start_indexs[ind]) / num_workers) for ind,dlf in enumerate(dataset.lmdb_files)] #int((end_index - start_index) / num_workers)
    worker_starts = [start_indexs[ind] + sentence_per_workers[ind] * worker_idx for ind,dlf in enumerate(dataset.lmdb_files)] #start_index + sentence_per_worker * worker_idx
    worker_ends = [min(worker_starts[ind] + sentence_per_workers[ind], end_indexs[ind]) for ind,dlf in enumerate(dataset.lmdb_files)] #min(worker_start + sentence_per_worker, end_index)
    dataset.lmdb_chunk_reader = MultiLmdbChunkReader(dataset.lmdb_files, worker_starts, worker_ends,
                                                  dataset.mix_rates, dataset.bunchsize, dataset.maxsentframe, dataset.maxnumsent, 
                                                  dataset.nmod_pad, dataset.cachesize, dataset.shuffle_batch, dataset.random_seed, dataset.batch_ctrl, dataset.aug_infos)

def collate_fn(batch):
    data, meta = batch[0]
    return data, meta
