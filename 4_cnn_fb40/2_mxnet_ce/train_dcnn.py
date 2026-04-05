# pylint:skip-file
import os, sys
import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model
import numpy as np


"""
    --network:
    --clip-gradient:
    --gpus:
    --batch-size:
    --lr:
    --lr-factor:
    --lr-factor-epoch:
    --wd:
    --momentum:
    --model-prefix:
    --load-epoch:
    --kv-store:
    --num-epochs:
    --epoch-size:
    --eval-size:
    --display-freq:
"""
parser = argparse.ArgumentParser(description='train an image classifer')
parser.add_argument('--network', type=str, default='dilate',
                    help = 'the cnn to use')
parser.add_argument('--multi-node', type=bool, default=False,
                    help='use multiple machine to training')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used when multi-node training mode is False, e.g "0,1,2,3"')
parser.add_argument('--ngpu-per-worker', type=int, default=4,
                    help='ngpu per worker when multi-node training mode is True')
parser.add_argument('--batch-size', type=int, default=1,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.005,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='the weight decay value')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='the momentum value')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the momentum value')
parser.add_argument('--model-prefix', type=str, default="mxmodel0",
                    help='the prefix of the model to load/save')
parser.add_argument('--num-epochs' , type=int, default=10,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--epoch-size', type=int, default='33750',
                    help ='the epoch size')
parser.add_argument('--eval-size', type=int, default='1000',
                    help='the eval size')
parser.add_argument('--train-metric', type=str, default='acc',
                    help='calc function of train metric')
parser.add_argument('--eval-metric', type=str, default='acc',
                    help='calc function of eval metric')
parser.add_argument('--display-freq', type=int, default='100',
                    help='the display frequent')
parser.add_argument('--mutable-data', type=bool, default='True',
                    help='is mutable data shape')
parser.add_argument('--frame-num', type=int, default=2048,
                    help='frame num of multi sentences')
parser.add_argument('--sentence-num', type=int, default=2048,
                    help='max sentence num')
parser.add_argument('--eval-total-size', type=int,
                    help='total eval size')
parser.add_argument('--use-bmuf', type=bool, default=False,
                    help='bmuf flag')
parser.add_argument('--sync-freq', type=int, default=10,
                    help='synchornize freqence')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha for easgd')
parser.add_argument('--blr', type=float, default=1.0,
                    help='block lerning rate')
parser.add_argument('--bm', type=float, default=0.75,
                    help='block momentum')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for data reader')
parser.add_argument('--coff', type=float, default=1.0,
                    help='coffine value for resnet')
parser.add_argument('--lmdbdir', type=str, default='',
                    help='lmdbdir for data')
parser.add_argument('--part-id', type=int, default=0,
                    help='part index for data')
parser.add_argument('--pad-num', type=int, default=32,
                    help='pad-num around sentences')
parser.add_argument('--pad-between', type=int, default=8,
                    help='pad-num between sentences')
parser.add_argument('--frame-predict', type=int, default=4,
                    help='frame predict for time pooling')
parser.add_argument('--window-width', type=int, default=15,
                    help='frame predict for time pooling')
parser.add_argument('--num-classes', type=int, default=9004,
                    help='size of classification layer')
args = parser.parse_args()


# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(num_classes=args.num_classes, coff=args.coff)

class FullyChunkIter(mx.io.DataIter):
    def __init__(self, path_data, data_shape, batch_size, sampler_type,chunk_size,
        path_keylist="", label_name="label", pad_num=32, frame_predict=2, chunk_num=1, frame_num=0,\
        seed=1111, shuffle=False, num_parts=1, part_index=0, label_width=1,sentence_num=1024, window_width=15, pad_between=8):

        self.chunk_iter = mx.io.FullyChunkIter(
                label_name    =  label_name,
                path_data     =  path_data,
                path_keylist  =  path_keylist,
                data_shape    =  data_shape,
                batch_size    =  batch_size,
                frame_num     =  frame_num,
                label_width   =  label_width,
                sampler_type  =  sampler_type,
                chunk_num     =  chunk_num,
                sentence_num  =  sentence_num,
                pad_num       =  pad_num,
                pad_between   =  pad_between,
                frame_predict =  frame_predict,
                fixed_length  =  False,
                seed          =  seed,
                shuffle       =  shuffle,
                num_parts     =  num_parts,
                part_index    =  part_index,
                chunk_size    =  chunk_size)

        self.batch_size     = batch_size
        self.frame_predict  = frame_predict
        self.window_width   = window_width
        self.provide_data   = [("data", (batch_size, 1, 40, frame_num)),
                ("mask", (batch_size, frame_num/4, frame_num/4))]
        self.provide_label  = [("label",(batch_size, frame_num))]
        print self.provide_data, self.provide_label

    def next(self):
        data_batch = self.chunk_iter.next()

        mask_shape = (1, data_batch.data[0].shape[3]/4,
                         data_batch.data[0].shape[3]/4)
        mask_data  = np.zeros(mask_shape)
        label_data = data_batch.label[0].asnumpy()

        #TODO: init mask according to local window width
        for i in range(mask_shape[1]):
            # left scan
            for j in range(self.window_width + 1):
                id = i - j
                if id >=0 and id < mask_shape[1]:
                    if label_data[0][id*4] == -2:
                        break
                    mask_data[0][i][id] = 1

            # right scan
            for j in range(self.window_width + 1):
                id = i + j
                if id >=0 and id < mask_shape[1]:
                    if label_data[0][id*4] == -2:
                        break
                    mask_data[0][i][id] = 1

        data_batch.data.append(mx.nd.array(mask_data))
        data_batch.provide_data  = [("data", data_batch.data[0].shape), ("mask", mask_shape)]
        data_batch.provide_label = [("label", data_batch.label[0].shape)]

        return data_batch

# data
def get_iterator(args, kv):
    data_shape = (1, 40, args.frame_num)
    path = "%s/train_part%d"%(args.lmdbdir,args.part_id)

    train = FullyChunkIter(
        label_name    = "label",
        path_data     = path,
        path_keylist  = path+"/keys.txt",
        data_shape    = data_shape,
        batch_size    = args.batch_size,
        frame_num     = args.frame_num,
        label_width   = args.frame_num,
        sampler_type  = "RAND_SAMPLER",
        num_parts     = kv.num_workers,
        part_index    = kv.rank,
        seed          = args.seed,
        shuffle       = True,
        pad_num       = args.pad_num,
        pad_between   = args.pad_between,
        frame_predict = args.frame_predict,
        window_width  = args.window_width,
        chunk_size    = 128,
        sentence_num  = args.sentence_num,
        chunk_num     = 200)

    val = FullyChunkIter(
        label_name    = "label",
        path_data     = "%s/test"%(args.lmdbdir),
        path_keylist  = "%s/test/keys.txt"%(args.lmdbdir),
        data_shape    = data_shape,
        frame_num     = args.frame_num,
        label_width   = args.frame_num,
        batch_size    = args.batch_size,
        frame_predict = args.frame_predict,
        window_width  = args.window_width,
        sentence_num  = 1,
        sampler_type  = "SEQUENCE_SAMPLER",
        chunk_size    = args.batch_size)

    return (train, val)

# train
train_model.fit(args, net, get_iterator)
