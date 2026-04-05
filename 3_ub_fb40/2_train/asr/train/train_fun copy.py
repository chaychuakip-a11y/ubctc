# pcli2 2019 Nov.
# bmuf logic is from caffe
from ..data import PfileInfo, LmdbInfo, txtDataLoader, UnionDataLoader 
from ..utils import clip_grad_norm
from ..optim import Lookahead
from ..optim import SGD
from ..utils.message import *
from ..utils.train_helper import *
import copy
import torch.jit as jit
import torch.distributed as dist
import torch.optim as optim
import torch
import random


class BMUF():
    def __init__(self, model, config, dist):
        momentums = []
        global_models = []
        for param in model.parameters():
            temp = torch.zeros_like(param, requires_grad=False)
            temp.copy_(param.data)
            global_models.append(temp)
            momentums.append(torch.zeros_like(param, requires_grad=False))
        momentums = torch.nn.utils.parameters_to_vector(momentums)
        global_models = torch.nn.utils.parameters_to_vector(global_models)
        self.momentums = momentums
        self.global_models = global_models
        ### copy from /work/asrprg/hcwang17/tmp/tmp_for_xiaobao/train_init.py
        if os.path.exists("{}/_momentums.pth".format(config["TrainSetting"]["OutDir"])):
            momentums_dict = torch.load("{}/_momentums.pth".format(config["TrainSetting"]["OutDir"]),map_location=torch.device('cpu'))
            # print('===',momentums_dict['momentums'])
            # print('===',momentums_dict['epoch'])
            # print('===',momentums_dict['gpu_num'])
            if os.path.exists("{}/model.init".format(config["TrainSetting"]["OutDir"])):
                momentums = momentums_dict['momentums'].to(momentums)
                # momentums = momentums_dict['momentums']
            else:
                os.system('rm -rf {}/_momentums.pth'.format(config["TrainSetting"]["OutDir"]))
        ### copy from /work/asrprg/hcwang17/tmp/tmp_for_xiaobao/train_init.py
        self.bmuf_alpha = config["TrainSetting"].getfloat("BMUF_ALPHA")
        # rewrite by ssyan2 auto BMUF_BM
        self.bmuf_bm = config["TrainSetting"].getfloat("BMUF_BM") if config["TrainSetting"].getfloat("BMUF_BM") else  1. - 1./config["TrainSetting"].getint("NGpu") 
        self.bmuf_blr = config["TrainSetting"].getfloat("BMUF_BLR")
        self.dist = dist

    def update(self, model):
        self.__update_param(model)

    def __update_param(self, model):
        # for v, momentums, global_models in zip(model.parameters(), self.momentums, self.global_models):
        #     size = float(self.dist.get_world_size())
        #     avg = v.detach().clone()
        #     self.dist.all_reduce(avg.data, op=dist.ReduceOp.SUM)
        #     avg.data /= size
        #     update = self.bmuf_bm * momentums + global_models
        #     grad = avg - update
        #     momentums.copy_(self.bmuf_blr * grad +
        #                         self.bmuf_bm * momentums)
        #     global_models.copy_(global_models + momentums)
        #     update = self.bmuf_bm * momentums + global_models
        #     v.data.copy_(v.detach() - self.bmuf_alpha * (v.detach() - update))
        size = float(self.dist.get_world_size())
        v = torch.nn.utils.parameters_to_vector(model.parameters())
        avg = v.detach().clone()
        self.dist.all_reduce(avg.data, op=dist.ReduceOp.SUM)
        avg.data /= size
        update = self.bmuf_bm * self.momentums + self.global_models
        grad = avg - update
        self.momentums.copy_(self.bmuf_blr * grad +
                             self.bmuf_bm * self.momentums)
        self.global_models.copy_(self.global_models + self.momentums)
        update = self.bmuf_bm * self.momentums + self.global_models
        v.data.copy_(v.detach() - self.bmuf_alpha * (v.detach() - update))
        torch.nn.utils.vector_to_parameters(v, model.parameters())

def run_multi_gpu(config, LOG):
    dist.init_process_group("nccl", init_method=config["TrainSetting"]["InitMethod"],
                            world_size=config["TrainSetting"].getint("NGpu"),
                            rank=config["TrainSetting"].getint("GlobalRank"))
    if not torch.cuda.is_available():
        LOG.log("no gpu device is available")
        raise Exception("no gpu device is available")
    ##xxtong
    # np.random.seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.cuda.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    torch.set_printoptions(10)

    model = get_module(config["Model"]["ModelName"])()
    
    finetune_lm=False
    try:
        finetune_lm=config['NNLMSetting'].getboolean('FinetuneLM')
    except Exception as e:
        finetune_lm=False
    # init
    init_param(model, config["TrainSetting"]["PreviousModel"], finetune_lm)

    # train
    with torch.cuda.device(config["TrainSetting"].getint("LocalRank")):
            
        if config['TrainSetting'].get('TrainType') == "LM":
            d=txtDataLoader(
                config["NNLMSetting"]["TrainTextfile"],
                config["NNLMSetting"]["DictFile"],
                config["NNLMSetting"].getint("BatchSize"),
                config["NNLMSetting"].getint("TrainStartSent"),
                config["NNLMSetting"].getint("TrainEndSent"),
                ndivide=config["TrainSetting"].getint("NGpu"),
                divide_index=config["TrainSetting"].getint("GlobalRank"),
                shuffle_batch=config["NNLMSetting"].getboolean("Shuffle"),
                num_workers=1)

        if config['TrainSetting'].get('TrainType') == "ASR":
           
            ########### rewrite by ssyan2 start
            data_infos = eval(config["TrainDataSetting"]["data_infos"])
            
            d = UnionDataLoader(
                data_infos=data_infos,
                bunchsize=config["TrainDataSetting"].getint("Bunchsize"),
                batch_num=config["DataSetting"].getint("TrainIterNum")-10,
                #batch_num=110,
                maxsentframe=config["TrainDataSetting"].getint("MaxSentFrame"),
                maxnumsent=config["TrainDataSetting"].getint("MaxNumSent"),
                ndivide=config["TrainSetting"].getint("NGpu"),
                divide_index=config["TrainSetting"].getint("GlobalRank"),
                nmod_pad=config["TrainDataSetting"].getint("PadNum"),
                shuffle_batch=config["DataSetting"].getboolean("ShuffleBatch"),
                random_seed=config["TrainSetting"].getint("RandomSeed"),
                batch_ctrl=config["TrainDataSetting"].getboolean("BatchCtrl"))
            ########### rewrite by ssyan2 end

        model = model.cuda()
        if config["TrainSetting"].getboolean("JIT") == True:
            model = jit.script(model)
        # optimizer
        if config["TrainSetting"]["Optimizer"] == "SGD":
            print('SGD lr modified')
            # 遍历所有参数，根据前缀名给不同层设置不同的学习率
            params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name.startswith('encoder.densenet'):
                        params.append({'params': param, 'lr': config["TrainSetting"].getfloat("CurrentLearningRate")/100})
                    elif name.startswith('encoder.proj'):
                        params.append({'params': param, 'lr': config["TrainSetting"].getfloat("CurrentLearningRate")/100})
                    elif name.startswith('encoder.net'):
                        params.append({'params': param, 'lr': config["TrainSetting"].getfloat("CurrentLearningRate")/10})
                    elif name.startswith('encoder.enc_lstm'):
                        params.append({'params': param, 'lr': config["TrainSetting"].getfloat("CurrentLearningRate")/10})
                    else:
                        params.append({'params': param, 'lr': config["TrainSetting"].getfloat("CurrentLearningRate")})
            optimizer = SGD(
                params,
                lr=config["TrainSetting"].getfloat("CurrentLearningRate"),
                momentum=0.9,
                weight_decay=0,
                nesterov=True)
            # print(params)
        #     optimizer = SGD(
        #         model.parameters(),
        #         lr=config["TrainSetting"].getfloat("CurrentLearningRate"),
        #         momentum=0.9,
        #         weight_decay=0,
        #         nesterov=True)
        if config["TrainSetting"]["Optimizer"] == "ADAM":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate")
            )
        if config["TrainSetting"].getboolean("LookAhead") == True:
            optimizer = Lookahead(optimizer, k=config["TrainSetting"].getint("LookAhead_K"),
                                  alpha=config["TrainSetting"].getfloat("LookAhead_Alpha"))

        # bmuf global states
        bmuf = BMUF(model, config, dist)

        # train
        bmuf_train(model, bmuf, optimizer, config, d, LOG)
        if (config["TrainSetting"].getint("GlobalRank") == 0):
            LOG.log("train accomplished")
            save_model(model.cpu(), config["TrainSetting"]["CurrentModel"])
    #        train_iter.reset()

    dist.barrier()


def run_single_gpu(config, LOG):
    if not torch.cuda.is_available():
        LOG.log("no gpu device is available")
        raise Exception("no gpu device is available")

    # np.random.seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    # torch.cuda.manual_seed(config["TrainSetting"].getint("RandomSeed"))
    torch.set_printoptions(10)

    with torch.cuda.device(config["TrainSetting"].getint("LocalRank")):
        model = get_module(config["Model"]["ModelName"])()
            
        # train
        if config['TrainSetting'].get('TrainType') == "LM":
            d=txtDataLoader(config["NNLMSetting"]["TrainTextfile"],
                            config["NNLMSetting"]["DictFile"],
                            config["NNLMSetting"].getint("BatchSize"),
                            config["NNLMSetting"].getint("TrainStartSent"),
                            config["NNLMSetting"].getint("TrainEndSent"),
                            ndivide=1,
                            divide_index=0,
                            shuffle_batch=config["NNLMSetting"].getboolean("Shuffle"),
                            num_workers=1)
        
        if config['TrainSetting'].get('TrainType') == "ASR":
            ########### rewrite by ssyan2 start
            data_infos = eval(config["TrainDataSetting"]["data_infos"])
            
            d = UnionDataLoader(
                data_infos=data_infos,
                bunchsize=config["TrainDataSetting"].getint("Bunchsize"),
                batch_num=config["DataSetting"].getint("TrainIterNum")-10,
                #batch_num=400,
                maxsentframe=config["TrainDataSetting"].getint("MaxSentFrame"),
                maxnumsent=config["TrainDataSetting"].getint("MaxNumSent"),
                ndivide=config["TrainSetting"].getint("NGpu"),
                divide_index=config["TrainSetting"].getint("GlobalRank"),
                nmod_pad=config["TrainDataSetting"].getint("PadNum"),
                shuffle_batch=config["DataSetting"].getboolean("ShuffleBatch"),
                random_seed=config["TrainSetting"].getint("RandomSeed"),
                batch_ctrl=config["TrainDataSetting"].getboolean("BatchCtrl"))
            ########### rewrite by ssyan2 end
            

        finetune_lm=False
        try:
            finetune_lm=config['NNLMSetting'].getboolean('FinetuneLM')
        except Exception as e:
            finetune_lm=False
        
        model = model.cuda()
        if config["TrainSetting"].getboolean("JIT") == True:
            model = jit.script(model)
        if (os.path.split(config["TrainSetting"]["PreviousModel"])[1] != "None"):
            init_param(model, config["TrainSetting"]["PreviousModel"], finetune_lm)
        if config["TrainSetting"]["Optimizer"] == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate"),
                momentum=0.9,
                weight_decay=0,
                nesterov=True)
            # for name, param in model.named_parameters():
            #     print(name)
            # exit()
        if config["TrainSetting"]["Optimizer"] == "ADAM":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate")
            )
        if config["TrainSetting"].getboolean("LookAhead") == True:
            optimizer = Lookahead(optimizer, k=config["TrainSetting"].getint("LookAhead_K"),
                                  alpha=config["TrainSetting"].getfloat("LookAhead_Alpha"))

        simple_train(model, optimizer, config, d, LOG)
        LOG.log("train accomplished")
        save_model(model.cpu(), config["TrainSetting"]["CurrentModel"])


def run_validation(config, model, LOG):
            
    if config['TrainSetting'].get('TrainType') == "LM":
        d=txtDataLoader(
            config["NNLMSetting"]["ValidationTextfile"],
            config["NNLMSetting"]["DictFile"],
            1,
            config["NNLMSetting"].getint("ValidationStartSent"),
            config["NNLMSetting"].getint("ValidationEndSent"),
            ndivide=1,
            divide_index=config["TrainSetting"].getint("GlobalRank"),
            shuffle_batch=False,
            num_workers=1)
        validation(model, config, d, LOG)
    if config['TrainSetting'].get('TrainType') == "ASR":

        ########### rewrite by ssyan2 start
        dsets = []
        # 加载添加的数据集
        for ind,data_add in enumerate(eval(config["TrainDataSetting"]["List"])):
            if config[data_add].getint("ValidationIterNum") > 0:
                dsets.append(data_add)
        print("val datasetings",dsets)
        for datasetting in dsets:
            LOG.log("\nvalidated data {0}".format(datasetting))

            get_data_infos = eval(config["TrainDataSetting"]["data_infos"])
            val_data_infos = {
                "file_norm":get_data_infos["file_norm"],
                "data_list":[datasetting],
                }
            
            val_data_infos[datasetting] = copy.deepcopy(get_data_infos[datasetting])
            for val_k in get_data_infos[datasetting].keys():
                if "val_" == val_k[:4]:
                    val_data_infos[datasetting][val_k[4:]] = get_data_infos[datasetting][val_k]
            val_data_infos[datasetting]["mix_rate"] = 1.0

            d = UnionDataLoader(
                data_infos=val_data_infos,
                batch_num=config[datasetting].getint("ValidationIterNum")-10,
                bunchsize=config["TrainDataSetting"].getint("Bunchsize"),
                maxsentframe=4096,
                maxnumsent=1,
                ndivide=1,
                divide_index=0,
                num_workers=1,
                nmod_pad=config["TrainDataSetting"].getint("PadNum"),
                shuffle_batch=False,
                val=True)
            
            

            validation(model, config, d, LOG)
        ########### rewrite by ssyan2 end
    
    

class Trainer():
    def __init__(self, train_type):
        self.train_type = train_type

    def lm_forward(self, index, data_element, model):
        x, mask, y= data_element
        x = x.cuda()
        mask=mask.cuda()
        y = y.cuda()
        celoss = model(x, mask, y)
        return celoss
    def asr_forward(self, index, data_element, model):
        data, meta = data_element
        data = data.cuda()
        for key in meta:
            meta[key] = meta[key].cuda()
        # celoss = model(data, meta)
        totalloss, edloss,celoss = model(data, meta) # xiaobao
        return totalloss,edloss, celoss
    def forward(self, index, data_element, model):
        if self.train_type == "ASR":
            return self.asr_forward(index, data_element, model)
        if self.train_type == "LM":
            return self.lm_forward(index, data_element, model)
    

def bmuf_train(model, bmuf, optimizer, config, data_loader, LOG):


    model.train()
    sum_of_last_displayed_loss = 0
    sum_of_last_displayed_celoss = 0
    last_t = time.time()

    cur_model = config["TrainSetting"]["CurrentModel"]
    list_model_name = cur_model.split(".")
    print(list_model_name,len(data_loader))
    cur_iter = list_model_name[-2].replace("iter","")
    cur_part = list_model_name[-1].replace("part","")
    
    # if not os.path.exists("train_contiguous_new") and config["TrainSetting"].getint("GlobalRank") == 0:
    #     os.mkdir("train_contiguous_new") 

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
    print("################################")
    print("it is using the right dataloader")
    # print("data_loader=", len(data_loader))
    print("################################")

    for i, data_element in enumerate(data_loader):

         
        optimizer.zero_grad()

        loss, edloss, celoss = trainer.forward(i, data_element, model) # xiaobao
        
        loss.backward()

        
        sum_of_last_displayed_loss += edloss.item()
        sum_of_last_displayed_celoss += celoss.item()

        clip_grad_norm(model.parameters(), config["TrainSetting"].getfloat("ClipGradient"),
                       config["TrainSetting"].getfloat("ClipGradient2"), config["TrainSetting"].getfloat("Discount"))
        optimizer.step()

        if i % config["TrainSetting"].getint("Display") == 0 and i > 0:
            considered_iteration = config["TrainSetting"].getint("Display")
            displayed_loss = sum_of_last_displayed_loss / considered_iteration
            displayed_loss_ce = sum_of_last_displayed_celoss / considered_iteration
            LOG.log(
                "gpu: {} loss: {} {} {} / {}, {:.2f} seconds/iteration".format(config["TrainSetting"].getint("GlobalRank"),
                                                                            displayed_loss, displayed_loss_ce, i,
                                                                            config["DataSetting"].getint(
                                                                                "TrainIterNum"),
                                                                            (time.time() - last_t) / config[
                                                                                "TrainSetting"].getint("Display")),
                rank=config["TrainSetting"].getint("GlobalRank"),
                logtype=LogType.SYNC, syncid=i, syncheader="iteration{0}".format(i))
            sum_of_last_displayed_loss = 0
            sum_of_last_displayed_celoss = 0
            last_t = time.time()
        if i % config["TrainSetting"].getint("BMUF_SYNC") == 0 and i > 0:
            bmuf.update(model)
        if config["TrainSetting"].getint("CVInterval") > 0 and i % config["TrainSetting"].getint("CVInterval") == 0 and \
                config["TrainSetting"].getint("GlobalRank") == 0 and i > 0:
            print(f'{config["TrainSetting"]["CurrentModel"]}.{i} saved')
            save_model(model.cpu(), config["TrainSetting"]["CurrentModel"]+f'.{i}')
            model.cuda()
    #         # run_validation(config, model, LOG)
    #         # model.train()
    # if config["TrainSetting"].getint("CVInterval") == 0 and config["TrainSetting"].getint("GlobalRank") == 0:
    #     run_validation(config, model, LOG)
    torch.save({'momentums':bmuf.momentums, 'epoch':config["TrainSetting"]["CurrentModel"], 'gpu_num':int(config["TrainSetting"]["NGpu"])}, \
            "{}/_momentums.pth".format(config["TrainSetting"]["OutDir"]))


def simple_train(model, optimizer, config, data_loader, LOG):
    model.train()
    sum_of_last_displayed_loss = 0
    sum_of_last_displayed_celoss = 0
    last_t = time.time()

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
    
    for i, data_element in enumerate(data_loader):
    
        optimizer.zero_grad()

        loss, edloss, celoss = trainer.forward(i, data_element, model) # xiaobao
        
        loss.backward()
        sum_of_last_displayed_loss += edloss.item() 
        sum_of_last_displayed_celoss += celoss.item() 
        clip_grad_norm(model.parameters(), config["TrainSetting"].getfloat("ClipGradient"),
                       config["TrainSetting"].getfloat("ClipGradient2"), config["TrainSetting"].getfloat("Discount"))
        optimizer.step()

        if i % config["TrainSetting"].getint("Display") == 0 and i > 0:
            considered_iteration = config["TrainSetting"].getint("Display")
            displayed_loss = sum_of_last_displayed_loss / considered_iteration
            displayed_loss_ce = sum_of_last_displayed_celoss / considered_iteration
            LOG.log(
                "gpu: {} loss: {} {} {} / {}, {:.2f} seconds/iteration".format(config["TrainSetting"].getint("GlobalRank"),
                                                                            displayed_loss, displayed_loss_ce, i,
                                                                            config["DataSetting"].getint(
                                                                                "TrainIterNum"),
                                                                            (time.time() - last_t) / config[
                                                                                "TrainSetting"].getint("Display")))
            sum_of_last_displayed_loss = 0
            sum_of_last_displayed_celoss = 0
            last_t = time.time()
        if config["TrainSetting"].getint("CVInterval") > 0 and i % config["TrainSetting"].getint(
                "CVInterval") == 0 and i > 0:
            run_validation(config, model, LOG)
            model.train()
    if config["TrainSetting"].getint("CVInterval") == 0:
        run_validation(config, model, LOG)


def validation(model, config, data_loader, LOG):
    model.eval()
    criterion_holder = []

    trainer_class = get_module(config['TrainSetting'].get("Trainer")) if config['TrainSetting'].get("Trainer") is not None else Trainer
    trainer = trainer_class(config['TrainSetting'].get('TrainType'))
            
    for i, data_element in enumerate(data_loader):
        criterion,ce,pad = trainer.forward(i, data_element, model)
        #print(criterion)
        criterion = criterion.item()
        if i % config["TrainSetting"].getint("Display") == 0:
            LOG.log("validated {0} / {1}".format(i,
                                                 config["TrainSetting"].getint("CVSentNum")))
        criterion_holder.append(criterion)

    criterion_holder = np.array(criterion_holder)
    LOG.log("modelname.. {0}".format(config["TrainSetting"]["CurrentModel"]))
    LOG.log("acc: {0}".format(np.mean(criterion_holder)))

def single_gpu_warper(config, message_queue, LOG):
    status = printer(run_single_gpu)(config, LOG)
    message_queue.put(status)


def multi_gpu_warper(config, message_queue, LOG):
    status = printer(run_multi_gpu)(config, LOG)
    message_queue.put(status)


class Trainer():
    def __init__(self, train_type):
        self.train_type = train_type

    def lm_forward(self, index, data_element, model):
        x, mask, y= data_element
        x = x.cuda()
        mask=mask.cuda()
        y = y.cuda()
        celoss = model(x, mask, y)
        return celoss
    def asr_forward(self, index, data_element, model):
        data, meta = data_element
        data = data.cuda()
        for key in meta:
            meta[key] = meta[key].cuda()
        # celoss = model(data, meta)
        totalloss, edloss,celoss = model(data, meta) # xiaobao
        return totalloss,edloss, celoss
    def forward(self, index, data_element, model):
        if self.train_type == "ASR":
            return self.asr_forward(index, data_element, model)
        if self.train_type == "LM":
            return self.lm_forward(index, data_element, model)