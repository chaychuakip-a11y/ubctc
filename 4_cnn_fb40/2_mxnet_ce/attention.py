import logging
import math

import mxnet as mx
import numpy as np

class Debug(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], 'inplace', in_data[0])
        x = in_data[0].asnumpy()[0]
        print 'forward: ', x.shape 
        print x[0][0:33]
        print x[1][0:33]
        import sys
        sys.stdout.flush()

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], 'inplace', out_grad[0])
        x = in_grad[0].asnumpy()[0]
        #print 'backward: ', x.shape 
        #print x
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


def split_heads(x, depth_per_head, heads):
    """
    Returns a symbol with head dimension folded into batch and depth divided by the number of heads.

    :param x: Symbol of shape (batch, length, depth).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch * heads, length, depth_per_heads).
    """
    # (batch, length, heads, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(0, -1, heads, depth_per_head))
    # (batch, heads, length, depth/heads)
    x = mx.sym.transpose(data=x, axes=(0, 2, 1, 3))
    # (batch * heads, length, depth/heads)
    return mx.sym.reshape(data=x, shape=(-3, -1, depth_per_head))

def combine_heads(x, depth_per_head, heads):
    """
    Returns a symbol with both batch & length, and head & depth dimensions combined.

    :param x: Symbol of shape (batch * heads, length, depth_per_head).
    :param depth_per_head: Depth per head.
    :param heads: Number of heads.
    :return: Symbol of shape (batch, length, depth).
    """
    # (batch, heads, length, depth_per_head)
    x = mx.sym.reshape(data=x, shape=(-4, -1, heads, 0, depth_per_head))
    # (batch, length, heads, depth_per_head)
    x = mx.sym.transpose(x, axes=(0, 2, 1, 3))
    # (batch, length, depth)
    return mx.sym.reshape(x, shape=(-1, 0, depth_per_head * heads))


def broadcast_to_heads(x, num_heads, ndim, fold_heads=True):
    """
    Broadcasts batch-major input of shape (batch, d1 ... dn-1) to (batch*heads, d1 ... dn-1).

    :param x: Batch-major input. Shape: (batch, d1 ... dn-1).
    :param num_heads: Number of heads.
    :param ndim: Number of dimensions in x.
    :param fold_heads: Whether to fold heads dimension into batch dimension.
    :return: Tensor with each sample repeated heads-many times.
             Shape: (batch * heads, d1 ... dn-1) if fold_heads == True, (batch, heads, d1 ... dn-1) else.
    """
    dims = [0] * (ndim - 1)
    # x: (batch, 1)
    x = mx.sym.expand_dims(x, axis=1)
    # x: (batch, heads, dims...)
    x = mx.sym.broadcast_to(x, shape=[0, num_heads] + dims)
    if fold_heads:
        # (batch * heads, dims...)
        return mx.sym.reshape(x, shape=[-3] + dims)
    else:
        # x: (batch, heads, dims...)
        return x


def dot_attention(queries, keys, values, mask=None, dropout=0.0, prefix=''):
    """
    Computes dot attention for a set of queries, keys, and values.

    :param queries: Attention queries. Shape: (n, lq, d).
    :param keys: Attention keys. Shape: (n, lk, d).
    :param values: Attention values. Shape: (n, lk, dv).
    :param lengths: Optional sequence lengths of the keys. Shape: (n,).
    :param dropout: Dropout probability.
    :param prefix: Optional prefix
    :return: 'Context' vectors for each query. Shape: (n, lq, dv).
    """

    # (n, lq, lk)
    logits = mx.sym.batch_dot(lhs=queries, rhs=keys, transpose_b=True)#, name='%sdot' % prefix)
    #logits = mx.sym.Custom(data=logits, op_type='debug')
    logits = mx.sym.broadcast_add(logits, (mask-1)*1e10)#, name='%mask_mul' % prefix)

    probs = mx.sym.softmax(logits, axis=-1)
    probs = mx.sym.Dropout(probs, p=dropout, flag=True) if dropout > 0.0 else probs

    # (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
    return mx.sym.batch_dot(lhs=probs, rhs=values)#, name='%scontexts' % prefix)

def  multi_head_att(data, mask, channel, channel_out, heads, dropout, prefix=""):
    # swap axis: (batch, channel, 1, length) -> (batch, length, channel)
    inputs = mx.sym.Reshape(mx.sym.swapaxes(data, 1, 3), shape=(0, 0, -1))

    # combined: (batch, max_length, depth * 3)
    combined = mx.sym.FullyConnected(data=inputs,
            num_hidden=channel * 3,
            flatten=False)

    # split into query, keys and values
    # (batch, max_length, depth)
    # pylint: disable=unbalanced-tuple-unpacking
    queries, keys, values = mx.sym.split(data=combined, num_outputs=3, axis=2)

    # scale by sqrt(depth_per_head)
    channel_per_head = channel / heads
    queries = queries * (channel_per_head ** -1.0)

    # (batch*heads, length, depth/heads)
    queries = split_heads(queries, channel_per_head, heads)
    keys = split_heads(keys, channel_per_head, heads)
    values = split_heads(values, channel_per_head, heads)

    # (batch*heads, query_max_length, depth_per_head)
    contexts = dot_attention(queries, keys, values,
            mask=mask, dropout=dropout, prefix=prefix)

    # (batch, query_max_length, depth)
    contexts = combine_heads(contexts, channel_per_head, heads)

    # contexts: (batch, query_max_length, output_depth)
    contexts = mx.sym.FullyConnected(data=contexts,
                                    num_hidden=channel_out,
                                    flatten=False)
    contexts = mx.sym.expand_dims(mx.sym.swapaxes(contexts, dim1=1, dim2=2), 2)

    return contexts
