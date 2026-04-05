import find_mxnet
import mxnet as mx
import numpy as np
from attention import multi_head_att

def ConvFactory(data, label, num_filter, kernel, stride=(1,1), pad=(0, 0), dilate=(1, 1), act_type=None,  name=None, suffix='', cudnn_tune="off", no_bias=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, cudnn_tune=cudnn_tune, cudnn_off=True, no_bias=no_bias)
    #conv = mx.sym.MaskZero(data=conv, label=label)
    if act_type != None:
        conv = mx.sym.LeakyReLU(data=conv, slope=0.1)

    return conv

def DeconvFactory(data, label, num_filter, kernel, stride=(1,1), pad=(0, 0), dilate=(1, 1), act_type=None,  name=None, suffix=''):
    conv = mx.sym.Deconvolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    #conv = mx.sym.MaskZero(data=conv, label=label)
    if act_type != None:
        conv = mx.sym.LeakyReLU(data=conv, slope=0.1)

    return conv

def ResdModule(data, label, state, num_filter, coff=1.0, kernel=(1,3), pad=(0,1), stride=(1,1), drop=0.1):
    conv = ConvFactory(data=data, label=label, num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, act_type='relu')
    #if drop != 0:
    conv = mx.sym.Dropout(data=conv, p=0.1, flag=True)
    conv = ConvFactory(data=conv, label=label, num_filter=num_filter, kernel=kernel, pad=pad, stride=stride)
    res  = (state + conv)
    act  = mx.sym.LeakyReLU(data=res, slope=0.1)

    return res, act

def CompositeModuleAtt(data, label, mask, state, num_layer, num_filter, kernel=(1, 3), pad=(0, 1)):
    res, act = state, data

    for i in range(num_layer):
        self_att = multi_head_att(act, mask, 512, num_filter, 8, 0.1)
        #self_att = mx.sym.MaskZero(self_att, label)
        res      = res + self_att
        act      = mx.sym.LeakyReLU(data=res, slope=0.1)
        res, act = ResdModule(data=act, label=label, state=res, num_filter=num_filter, drop=0.0) 

    return res, act

def CompositeModule(data, label, mask, state, num_layer, num_filter, kernel=(1, 3), pad=(0, 1)):
    res, act = state, data

    for i in range(num_layer):
        res, act = ResdModule(data=act, label=label, state=res, num_filter=num_filter, drop=0.0) 

    return res, act

def Conv2dModule(data, label):
    # layer 0
    conv  =	ConvFactory(data=data, label=label, num_filter=64, kernel=(3,3), pad=(1,1), stride=(1,1), act_type='relu')
    pool  = mx.sym.Pooling(data=conv,kernel=(1,2),stride=(1,2), pool_type='max')
    proj  = ConvFactory(data=conv, label=label, num_filter=64, kernel=(1,2), stride=(1,2), no_bias=True)

    # res layer 1
    res, act = ResdModule(pool, label, proj, kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=64, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=64, drop=0)
    pool     = mx.sym.Pooling(data=act, kernel=(2,2),stride=(2,2), pool_type='max')
    proj     = ConvFactory(data=res, label=label, num_filter=96, kernel=(2,2), stride=(2,2), no_bias=True) 

    # res layer 2
    res, act = ResdModule(pool, label, proj, kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=96, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=96, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=96, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=96, drop=0)
    pool     = mx.sym.Pooling(data=act, kernel=(2,2),stride=(2,2), pool_type='max')
    proj     = ConvFactory(data=res, label=label, num_filter=160, kernel=(2,2), stride=(2,2), no_bias=True) 

    # res layer 3
    res, act = ResdModule(pool, label, proj, kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=160, drop=0)
    pool     = mx.sym.Pooling(data=act, kernel=(2,1),stride=(2,1), pool_type='max')
    proj     = ConvFactory(data=res, label=label, num_filter=256, kernel=(2,1), stride=(2,1), no_bias=True) 

    # res layer 4
    res, act = ResdModule(pool, label, proj, kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=256, drop=0)
    res, act = ResdModule(act,  label, res,  kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=256, drop=0)
    pool     = act

    dconv = DeconvFactory(data=pool,  label=label, num_filter=256,  kernel=(1,2), stride=(1,2))

    # final 1d feature
    rep      = mx.sym.Reshape(data=dconv, shape=(1, -1, 1, 0))
    conv     = ConvFactory(data=rep, label=label, num_filter=512, kernel=(1,1), stride=(1,1)) 

    #_, shape, _ = conv.infer_shape(data=(1,1,40,2048), label=(1,2048))
    #print "xxxx: shape: ", shape

    return conv

def get_symbol(seq_len=1024,fea_dim=40, num_classes=9004, batch_num=1, module_num=10, coff=1):
    # data
    data  = mx.sym.Variable(name="data")
    label = mx.sym.Variable(name="label")
    mask  = mx.sym.Variable(name="mask")
    data  = mx.sym.Reshape(data=data, shape=(batch_num, 1, fea_dim, -1))

    # 2d to 1d
    conv  = Conv2dModule(data, label)
    res = conv
    act = conv

    # layer 1~12
    res, act = CompositeModuleAtt(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModule(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModuleAtt(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModule(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModuleAtt(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModule(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModuleAtt(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModule(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModuleAtt(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=512, kernel=(1, 3), pad=(0, 1))

    # layer 13-44
    res    = ConvFactory(data=res, label=label, num_filter=768, kernel=(1,1), stride=(1,1), no_bias=True)
    res, act = CompositeModule(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=768, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModuleAtt(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=768, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModule(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=768, kernel=(1, 3), pad=(0, 1))

    # layer 45~48
    res    = ConvFactory(data=res, label=label, num_filter=1280, kernel=(1,1), stride=(1,1), no_bias=True)
    res, act = CompositeModuleAtt(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=1280, kernel=(1, 3), pad=(0, 1))
    res, act = CompositeModule(data=act, label=label, mask=mask, state=res, num_layer=1, num_filter=1280, kernel=(1, 3), pad=(0, 1))

    # layer 49~51
    conv    = ConvFactory(data=act, label=label, num_filter=2048, kernel=(1,3), pad=(0,1), stride=(1,1), act_type='relu')
    convlow = DeconvFactory(data=conv,  label=label, num_filter=512,  kernel=(1,4), stride=(1,4))

    convout = ConvFactory(data=convlow, label=label, num_filter=num_classes, kernel=(1,1),stride=(1,1))
    convout = mx.sym.SwapAxis(convout, dim1=1, dim2=3)
    convout = mx.sym.Reshape(data=convout, shape=(-1,num_classes))
    label   = mx.sym.Reshape(data=label, shape=(-1,))
    label   = (label < -1) + label
    softmax = mx.sym.SoftmaxOutput(data=convout, label=label, use_ignore=True, normalization='valid')
    return softmax
