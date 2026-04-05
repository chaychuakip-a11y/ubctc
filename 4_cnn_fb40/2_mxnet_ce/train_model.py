import find_mxnet
import mxnet as mx
import logging

def fit(args, network, data_loader):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # load model?
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-0"
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model?
    checkpoint = None if model_prefix is None or kv.rank is not 0 else mx.callback.do_checkpoint(model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    if 'multi_node' in args and args.multi_node == True and len(args.gpus.split(',')) is 1:
        devs = mx.gpu(kv.rank % args.ngpu_per_worker)

    epoch_size = args.epoch_size
    if args.kv_store.startswith('dist'):
        epoch_size /= kv.num_workers

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    if 'mutable_data' in args and args.mutable_data is True:
        model_args['mutable_data'] = args.mutable_data

    if 'use_bmuf' in args and args.use_bmuf is True:
        model_args['use_bmuf'] = True
        model_args['sync_freq'] = args.sync_freq
        model_args['alpha'] = args.alpha
        model_args['blr']   = args.blr
        model_args['bm']    = args.bm

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    # metric
    #metrics = mx.metric.create_from_list([args.train_metric, args.eval_metric])
    #if 'eval_total_size' in args and args.eval_total_size is not None:
    #    metrics['eval'].set_total(args.eval_total_size)
    metrics = mx.metric.create('acc')

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = args.momentum,
        wd                 = args.wd,
        epoch_size         = epoch_size,
        eval_size          = args.eval_size,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2),
        optimizer          = args.optimizer,
        **model_args)

    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = kv,
        eval_metric        = metrics,
        #batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.display_freq),mx.callback.log_train_metric(args.display_freq)],
        batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.display_freq)],
        epoch_end_callback = checkpoint)
