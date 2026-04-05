# by pcli2, innovated by dlp/jieding's source code
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from .pfile_reader import Normfile, Pfileinfo, MultiPfileChunkReader

# add by mzwang7, for read txt
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

def PfileDataLoader(file_feas, file_labs, file_norm, batch_num, bunchsize, maxsentframe, maxnumsent, start_sents, end_sents, mix_rates, ndivide=1, divide_index=0, nmod_pad=64, 
                    shuffle_batch=True, num_workers=2, cachesize=10000, random_seed=0, batch_ctrl=False):

    fea_pfileinfos = [Pfileinfo(file_feas[ind]) for ind,ff in enumerate(file_feas)]
    lab_pfileinfos = [Pfileinfo(file_labs[ind]) if file_labs[ind] is not None else None for ind,fl in enumerate(file_labs)] 

    for ind,fl in enumerate(file_labs):
        if file_labs[ind] is not None:
            assert fea_pfileinfos[ind].num_sentences == lab_pfileinfos[ind].num_sentences, "pfile index %d number of sentences in feature and label are not equal"%ind
    
    num_sentences_s = [fea_pfileinfos[ind].num_sentences for ind,ff in enumerate(file_feas)]
    read_startindexs = [None for ind,ff in enumerate(file_feas)]
    read_endindexs = [None for ind,ff in enumerate(file_feas)]

    for ind,ff in enumerate(file_feas):
        
        assert start_sents[ind] < num_sentences_s[ind], "pfile index %d start_sent must smaller than total sentences %d"%(ind,num_sentences_s[ind])

        end_sents[ind] = min(num_sentences_s[ind], end_sents[ind])
        num_sentences_s[ind] = end_sents[ind] - start_sents[ind]
        read_startindexs[ind] = int(num_sentences_s[ind] / ndivide * divide_index) + start_sents[ind]
        read_endindexs[ind] = read_startindexs[ind] + int(num_sentences_s[ind] / ndivide) if divide_index < ndivide - 1 else end_sents[ind]

    
    
    pfile_dataset = PfileDataset(file_feas, file_labs, file_norm, read_startindexs, read_endindexs, mix_rates, batch_num, bunchsize, maxsentframe, 
                                 maxnumsent, nmod_pad, shuffle_batch, cachesize, random_seed, batch_ctrl)
    
    return DataLoader(pfile_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn", collate_fn=collate_fn, worker_init_fn=multi_worker_init_fn)






class PfileDataset(Dataset):
    def __init__(self, file_feas, file_labs, file_norm, start_indexs, end_indexs, mix_rates, batch_num, bunchsize, maxsentframe, maxnumsent, 
                 nmod_pad=64, shuffle_batch=True, cachesize=10000, random_seed=0, batch_ctrl=False):
        self.file_feas = file_feas
        self.file_labs = file_labs
        norm = Normfile(file_norm)
        mean = torch.from_numpy(norm.mean)
        mean = mean.reshape(1, 1, -1, 1)
        var = torch.from_numpy(norm.var)
        var = var.reshape(1, 1, -1, 1)
        self.mean = mean
        self.var = var
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
        self.pfile_chunk_reader = None
        self.random_seed = random_seed
        self.batch_ctrl = batch_ctrl

    def __getitem__(self, index):
        if self.lab_pfileinfos is None:
            data = self.pfile_chunk_reader.getbatch()
            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            meta = {}
            meta["mask"] = data_mask
            return data, meta
        else:
            data, label = self.pfile_chunk_reader.getbatch()
            label_ctc = []
            if label[0].shape[1] == 2:
                for idx, element in enumerate(label):
                    e1, e2 = np.split(element, 2, axis=1)
                    label_ctc.append(e2)
                    label[idx] = e1 # xiaobao
            
            data, data_mask = self.__pad_nmod(data, self.nmod_pad, 0)
            data = data - self.mean
            data = data * self.var
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
    dataset.fea_pfileinfos = [Pfileinfo(dff) for dff in dataset.file_feas]
    dataset.lab_pfileinfos = [Pfileinfo(dfl) if dfl is not None else None for dfl in dataset.file_labs]
    start_indexs = dataset.start_indexs
    end_indexs = dataset.end_indexs
    sentence_per_workers = [int((end_indexs[ind] - start_indexs[ind]) / num_workers) for ind,dff in enumerate(dataset.file_feas)] #int((end_index - start_index) / num_workers)
    worker_starts = [start_indexs[ind] + sentence_per_workers[ind] * worker_idx for ind,dff in enumerate(dataset.file_feas)] #start_index + sentence_per_worker * worker_idx
    worker_ends = [min(worker_starts[ind] + sentence_per_workers[ind], end_indexs[ind]) for ind,dff in enumerate(dataset.file_feas)] #min(worker_start + sentence_per_worker, end_index)
    dataset.pfile_chunk_reader = MultiPfileChunkReader(dataset.fea_pfileinfos, dataset.lab_pfileinfos, worker_starts, worker_ends,
                                                  dataset.mix_rates, dataset.bunchsize, dataset.maxsentframe, dataset.maxnumsent, 
                                                  dataset.nmod_pad, dataset.cachesize, dataset.shuffle_batch, dataset.random_seed,dataset.batch_ctrl)

def collate_fn(batch):
    data, meta = batch[0]
    return data, meta
