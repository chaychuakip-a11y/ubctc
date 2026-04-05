# -*- coding: utf-8 -*-
#python=3.6
import os
import lmdb
import tqdm
import numpy as np

try:
    from .datum_pb2 import SpeechDatum #Float MC
    from .datum1_pb2 import Datum1 #diffu rir
    from .datum2_pb2 import Datum2 # czzhu2
except:
    from datum_pb2 import SpeechDatum #Float MC
    from datum1_pb2 import Datum1 #diffu rir
    from datum2_pb2 import Datum2 # czzhu2

import delta

def lmdb_create():
    # 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖
    # map_size定义最大储存容量，单位是kb，以下定义1TB容量
    env = lmdb.open("./train",map_size=1000)
    env.close()

def lmdb_using():
    env = lmdb.open("./train", map_size=int(1e9)) 

    # 参数write设置为True才可以写入
    txn = env.begin(write=True)  

    # 添加数据和键值 
    txn.put(key = '1'.encode(), value = 'aaa'.encode())
    txn.put(key = '2'.encode(), value = 'bbb'.encode()) 
    txn.put(key = '3'.encode(), value = 'ccc'.encode()) 

    # 通过键值删除数据 
    txn.delete(key = '1'.encode()) 

    # 修改数据 
    txn.put(key = '3'.encode(), value = 'ddd'.encode()) 

    # 通过commit()函数提交更改 
    txn.commit() 
    env.close()

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

np.random.seed(314)
voc_size      = 3261

self_padzeros_mch  = np.zeros((1, 40))

def lmdb_read(lmdb_path='./train'):
    lmdb_key = get_lmdb_key(lmdb_path)
    
    with open(lmdb_key,"r") as fr:
        lines = fr.readlines()
    print(lines[:10],len(lines))
    np.random.shuffle(lines)
    print(lines[:10],len(lines))
    clines = lines[:20000]
    sclines = sorted(clines,key=lambda x:int(x.strip().split(" ")[0]))
    print(sclines[:10],len(sclines))
    env = lmdb.Environment(lmdb_path, readonly=True, readahead=True, lock=False) 
    #env = lmdb.open("./train")   # or
    txn = env.begin()
    for sl in tqdm.tqdm(sclines[:]):
        label_out    = []
        syllable_out = []
        wordid_out   = []
        with txn.cursor() as cursor:
            k = sl.strip().split(" ")[0].encode('utf-8')
            #print(k)
            #print(list(cursor.get_keys())[:5])
            cursor.set_key(k)
            datum    = Datum2()
            datum.ParseFromString(cursor.value())

            data     = np.frombuffer(datum.anc.data, dtype=np.int16)
            label    = np.frombuffer(datum.anc.state_data, dtype=np.int16).reshape(-1, 1)
            syllable = np.fromstring(datum.anc.syllable_data, dtype=np.int16).reshape(-1, 1)

            #print(sl,data.shape,label.shape,syllable.shape)

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
            #data = (data - self.mel_spec_mean[None,:]) * self.mel_spec_std
            nframes = min(data.shape[0], label.shape[0], syllable.shape[0])
            data = data[:nframes,:]
            padzeros_mch = np.tile(self_padzeros_mch, (nframes, 1))
            data = np.concatenate((data, padzeros_mch), axis=1)
            label_out.append(label[:nframes, :])
            syllable_out.append(syllable[:nframes, :])
            wordid_out.append(wordid)

            #print(nframes,data.shape,label.shape,syllable.shape)
            """
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
            """

    
    # close
    env.close()



def main():
    #lmdb_create()
    #lmdb_using()
    #lmdb_read()
    lmdb_read("/yrfs4/asrdictt/czzhu2/rnnt/S1_CN/lmdb_clean/lmdb2/")

# errors:
# 1. lmdb.MapFullError: mdb_put: MDB_MAP_FULL: Environment mapsize limit reached
# 解决方法： lmdb.open("./train", map_size=int(1e9)

# 2. TypeError: Won't implicitly convert Unicode to bytes; use .encode()
# 解决方法： TypeError:不会隐式地将Unicode转换为字节,对字符串部分，进行.encode()

if __name__ == '__main__':
    main()
