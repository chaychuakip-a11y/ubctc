import mxnet as mx
from collections import namedtuple

class GradMul(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], 'inplace', in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], 'inplace', out_grad[0]*28.0)

@mx.operator.register('GradMul')
class GradMulProp(mx.operator.CustomOpProp):
    def __init__(self):
        #self.name = name
        super(GradMulProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        out_shape  = in_shape[0]
        return  [data_shape], [out_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return GradMul()

class GradDiv(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], 'inplace', in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], 'inplace', out_grad[0]*(1.0/28.0))

@mx.operator.register('GradDiv')
class GradDivProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(GradDivProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        out_shape  = in_shape[0]
        return  [data_shape], [out_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return GradDiv()

import numpy as np
class Debug(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], 'inplace', in_data[0])
        x = in_data[0].asnumpy()[0]
        print 'forward: ', x.shape 
        print x
        import sys
        sys.stdout.flush()

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], 'inplace', out_grad[0])
        x = in_grad[0].asnumpy()[0]
        print 'backward: ', x.shape 
        print x
        import sys
        sys.stdout.flush()

@mx.operator.register('debug')
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DebugProp, self).__init__(need_top_grad=True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        out_shape  = in_shape[0]
        return  [data_shape], [out_shape], []
    def create_operator(self, ctx, shapes, dtypes):
        return Debug()

def weight_norm(num_filter, num_channel, kernel=(1,1), is_fc=False, name=None):
    if is_fc is False:
        shape = (num_filter, num_channel, kernel[0], kernel[1])
        g = mx.sym.Variable(shape=shape, name='%s_g_weight'%name)
        v = mx.sym.Variable(shape=(num_filter,), name='%s_gamma'%name)
        w = mx.sym.broadcast_mul(mx.sym.L2Normalization(data=g), mx.sym.Reshape(data=v, shape=(num_filter, 1, 1, 1)))
    else:
        shape = (num_filter, num_channel)
        g = mx.sym.Variable(shape=shape, name='%s_g_weight'%name)
        v = mx.sym.Variable(shape=(num_filter,), name='%s_gamma'%name)
        w = mx.sym.broadcast_mul(mx.sym.L2Normalization(data=g), mx.sym.Reshape(data=v, shape=(num_filter, 1)))

    return w

def layer_norm(data, num_hidden, eps=1e-5, is_fc=False, name=None):
    """
    layer normalization
    """

    scale = mx.sym.Variable('%s_gamma'%name, shape=(num_hidden,))
    shift = mx.sym.Variable('%s_beta'%name, shape=(num_hidden,))

    mean = mx.sym.mean(data=data, axis=1, keepdims=True)
    var  = mx.sym.mean(mx.sym.square(mx.sym.broadcast_minus(data, mean)),
                       axis=1, keepdims=True)

    input_norm  = mx.sym.broadcast_minus(data, mean)
    input_norm  = mx.sym.broadcast_mul(input_norm, mx.sym.rsqrt(var+eps))

    shape = (1, num_hidden, 1, 1) if is_fc is False else (1, num_hidden)
    input_norm  = mx.sym.broadcast_mul(input_norm, mx.sym.Reshape(scale, shape=shape))
    input_norm  = mx.sym.broadcast_add(input_norm, mx.sym.Reshape(shift, shape=shape))

    return input_norm

def conv_factory(data, num_filter, num_channel, kernel, stride=(1,1), pad=(0, 0), act_type=True, no_bias=False, suffix='', cudnn_tune="off", use_bn=False, name=None):
    #w = weight_norm(num_filter, num_channel, kernel, False, name="%s_conv"%name)
    w  = mx.sym.Variable(name="%s_weight"%name)
    conv = mx.sym.Convolution(data=data, weight=w, num_filter=num_filter, kernel=kernel, stride=stride,
                              pad=pad, cudnn_tune=cudnn_tune, no_bias=no_bias, name=name)
    #conv = layer_norm(conv, num_filter, is_fc=False, name=name)

    return conv

def fc_linear(data, num_hidden, num_input, no_bias=False, name=None):
    #w = weight_norm(num_hidden, num_input, None, True, name="%s_conv"%name)
    w  = mx.sym.Variable(name="%s_weight"%name)
    fc = mx.sym.FullyConnected(data=data, weight=w, num_hidden=num_hidden, no_bias=no_bias, name=name)
    #fc = layer_norm(fc, num_hidden, is_fc=True, name=name)

    return fc
