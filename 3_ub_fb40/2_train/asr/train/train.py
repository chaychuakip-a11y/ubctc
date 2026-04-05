# pcli2 2019 Nov.
# bmuf logic is from caffe
from ..data import PfileInfo, LmdbInfo, txtDataLoader, LmdbDataLoader, UnionDataLoader 
from ..utils import clip_grad_norm
from ..optim import Lookahead
from ..optim import SGD
from ..utils.message import *
from torch.backends import cudnn
from ..utils.train_helper import *
import configparser
import copy
import torch.jit as jit
import torch.distributed as dist
import random
import torch.optim as optim
import torch
import multiprocessing


context = multiprocessing.get_context("spawn")

cudnn.benchmark = False
cudnn.enabled = True


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
                shuffle_batch=config["TrainSetting"].getboolean("ShuffleBatch"),
                random_seed=config["TrainSetting"].getint("RandomSeed"),
                batch_ctrl=config["TrainDataSetting"].getboolean("BatchCtrl"))
            ########### rewrite by ssyan2 end

        model = model.cuda()
        if config["TrainSetting"].getboolean("JIT") == True:
            model = jit.script(model)
        # optimizer
        if config["TrainSetting"]["Optimizer"] == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=config["TrainSetting"].getfloat("CurrentLearningRate"),
                momentum=0.9,
                weight_decay=0,
                nesterov=True)
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
                shuffle_batch=config["TrainSetting"].getboolean("ShuffleBatch"),
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
        # jiazai tianjia de shujuji
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
    # print("################################")
    # print("it is using the right dataloader")
    # # print("data_loader=", len(data_loader))
    # print("################################")

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
            # run_validation(config, model, LOG)
            # model.train()
    if config["TrainSetting"].getint("CVInterval") == 0 and config["TrainSetting"].getint("GlobalRank") == 0:
        run_validation(config, model, LOG)
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



class train():
    def __init__(self):
        self.__done = False
        self.__LOG = None
        self.__message = None
        self.__config = None

    def start_train(self):
        if self.__config is None:
            self.__raise_err("missing config file")
        self.__check_config()
        self.__init_env()
        self.__loop()

    def load_config(self, configPath):
        config = configparser.ConfigParser()
        config.read(configPath)
        self.__config = config 
        if not os.path.exists(configPath) : self.__config = None #add py ssyan2

    def __init_env(self):
        if "MASTER_ADDR" in os.environ:
            self.__distributed = True
        else:
            self.__distributed = False

        if self.__distributed:
            #self.__master_addr = socket.gethostbyname(os.environ["MASTER_ADDR"])
            self.__master_addr = os.environ["MASTER_ADDR"]
            self.__master_port = int(os.environ["MASTER_PORT"])
            self.__log_port = get_port_id(self.__master_port)
            self.__num_worker = int(os.environ["WORLD_SIZE"])
            self.__rank = int(os.environ["RANK"])
        else:
            self.__master_addr = socket.gethostbyname("localhost")
            self.__master_port = random.randint(20000, 30000)
            self.__log_port = get_port_id(self.__master_port)
            self.__num_worker = 1
            self.__rank = 0
        if "NGPU_PER_WORKER" in os.environ:  #### modify by taoyu
            ngpu = int(os.environ["NGPU_PER_WORKER"]) * self.__num_worker
            self.__config["TrainSetting"]["NGpu"] = str(ngpu)
            self.__config["TrainSetting"]["BMUF_BLR"] = "1.0"
            self.__config["TrainSetting"]["BMUF_ALPHA"] = "0.75" if ngpu == 4 else "1.0"
            if ngpu == 1:
                self.__config["TrainSetting"]["BMUF_BM"] = "1.0"
            elif ngpu == 4:
                self.__config["TrainSetting"]["BMUF_BM"] = "0.75"
            elif ngpu == 8:
                self.__config["TrainSetting"]["BMUF_BM"] = "0.9"
            elif ngpu == 12:
                self.__config["TrainSetting"]["BMUF_BM"] = "0.92"
            elif ngpu == 16:
                self.__config["TrainSetting"]["BMUF_BM"] = "0.94"
            elif ngpu == 32:
                self.__config["TrainSetting"]["BMUF_BM"] = "0.972"
            else:
                self.__config["TrainSetting"]["BMUF_BM"] = str(1-1/ngpu)
        self.__message = Message(
            self.__master_addr, self.__log_port, self.__num_worker)
        if self.__rank == 0:
            self.__message.start_server()
        else:
            self.__message.start_client()
        if self.__rank == 0:
            self.__reset_current_log(delete=True)

        self.__LOG = self.__message.getlog()

        self.__config["TrainSetting"]["MasterAddr"] = self.__master_addr
        self.__config["TrainSetting"]["MasterPort"] = str(self.__master_port)
        log = ''
        log += get_log_header("Network Summary")
        log += "Master Addr.. {0}\n".format(
            self.__config["TrainSetting"]["MasterAddr"])
        log += "Master Port.. {0}\n".format(
            self.__config["TrainSetting"]["MasterPort"])
        if self.__distributed:
            log += "Log Port.. {0}\n".format(self.__log_port)
        log += "Num Workers.. {0}".format(self.__num_worker)
        if self.__rank == 0:
            self.__LOG.log(log)

    def __reset_current_log(self, delete=True):
        if self.__rank != 0:
            return
        train_dir = self.__config["TrainSetting"]["OutDir"]
        r = get_current_part_index_and_sent(self.__config)
        if not r:
            current_log_name = "default.log"
        else:
            iteration, dist_ind, iter_idx, part_idx, part_numx, sent = r # rewrite by ssyan
            if iter_idx == "init":
                current_log_name = "init.log"
            else:
                current_log_name = "iter{0}.part{1}.log".format(
                    iter_idx, part_idx)
        current_log_name = os.path.join(train_dir, current_log_name)
        self.__log_name = current_log_name
        if not os.path.exists(self.__config["TrainSetting"]["OutDir"]):
            os.mkdir(self.__config["TrainSetting"]["OutDir"])
        if delete:
            if os.path.exists(self.__log_name):
                os.remove(self.__log_name)
        self.__message.set_logpath(
            self.__log_name, self.__config["TrainSetting"].getint("NGpu"))

    def __init_train(self):
        # get last model name
        previous_model_name = get_previous_model_name(self.__config)
        previous_model_name = str(previous_model_name)

        # get output dir
        train_dir = self.__config["TrainSetting"]["OutDir"]
        # get current module name
        self.__config["TrainSetting"]["PreviousModel"] = previous_model_name

        r = get_current_part_index_and_sent(self.__config)
        print("datasetting r::{0}".format(r))
        if not r:
            self.__done = True
            return 0
        self.__done = False
        iteration, dist_ind, iter_idx, part_idx, part_numx, sent = r

        self.__LOG.log("datasetting r::{0}".format(r)) #pengqi4
         
        if iter_idx == "init":
            current_model_name = "model.init"
            self.__is_inited = False
        else:
            current_model_name = "model.iter{0}.part{1}".format(
                iter_idx, part_idx)
            self.__is_inited = True
        self.__config["TrainSetting"]["CurrentModel"] = os.path.join(
            train_dir, current_model_name)

        # get training type
        train_type = self.__config['TrainSetting'].get('TrainType')
        train_type = "ASR" if train_type is None else train_type
        self.__config['TrainSetting']["TrainType"] = train_type

        
        # set data
        if self.__config['TrainSetting'].get('TrainType') == "LM":
            data_dir = self.__config["NNLMSetting"]["DataDir"]
            current_text_file = self.__config["NNLMSetting"]["TextPrefix"].replace('$', str(part_idx))
            current_text_file = os.path.join(data_dir, current_text_file)
            if not os.path.exists(current_text_file):
                self.__raise_err("{0} doesn't exist".format(current_text_file))
        
        # 使用dict管理数据信息
        self.data_infos = {
            "file_norm":"",
            "data_list":[]
        }
        # 根据DataType获取主数据(DataSetting)集路径
        
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            self.data_infos["data_list"].append("DataSetting")
            self.data_infos["DataSetting"] = {}
            self.data_infos["DataSetting"]["data_type"] = self.__config['DataSetting'].get('DataType')

            if "pfile" == self.__config["DataSetting"]["DataType"]:
                data_dir = self.__config["DataSetting"]["DataDir"]
                data_dir_lab = self.__config["DataSetting"]["DataDirLab"] # xiaobao
                current_label_pfile = self.__config["DataSetting"]["LabelPrefix"].replace('$', str(part_numx))
                current_feature_pfile = self.__config["DataSetting"]["FeaturePrefix"].replace('$', str(part_numx))
                current_label_pfile = os.path.join(data_dir_lab, current_label_pfile) # xiaobao
                current_feature_pfile = os.path.join(data_dir, current_feature_pfile)

                self.data_infos["DataSetting"]["pfile_fea"] = current_feature_pfile
                self.data_infos["DataSetting"]["pfile_lab"] = current_label_pfile
                if not os.path.exists(current_label_pfile):
                    self.__raise_err("{0} doesn't exist".format(current_label_pfile))
                if not os.path.exists(current_feature_pfile):
                    self.__raise_err("{0} doesn't exist".format(current_feature_pfile))
            
            elif "lmdb" == self.__config["DataSetting"]["DataType"]:
                Lmdb_dir = self.__config["DataSetting"]["LmdbDir"]
                current_lmdb = self.__config["DataSetting"]["LmdbPrefix"].replace('$', str(part_numx))
                current_lmdb = os.path.join(Lmdb_dir, current_lmdb) # ssyan2

                self.data_infos["DataSetting"]["lmdb_file"] = current_lmdb
                if not os.path.exists(current_lmdb):
                    self.__raise_err("{0} doesn't exist".format(current_lmdb))
            
            norm_file = self.__config["TrainDataSetting"]["NormFile"]
            self.data_infos["file_norm"] = norm_file
            if not os.path.exists(norm_file):
                self.__raise_err("{0} doesn't exist".format(norm_file))
            
            ########### add by ssyan2 start
            # get add lists and rates
            if "TrainDataSetting" not in self.__config or self.__config["TrainDataSetting"].get("TrainDatas") == None or self.__config["TrainDataSetting"].get("TrainDatas") == "":
                self.__config["TrainDataSetting"]["List"] = str(["TrainSetting"])
                self.__config["TrainDataSetting"]["MixRates"] = str([1.0])
                self.data_infos["DataSetting"]["mix_rate"] = 1.0
            else:
                if "init" == iter_idx :
                    if None == self.__config["TrainDataSetting"].get("MixFlag"):
                        self.__config["TrainDataSetting"]["List"] = str([pfile.strip(" ") for pfile in self.__config["TrainDataSetting"].get("TrainDatas").split(",") if pfile != ""])
                        #assert self.__config["TrainDataSetting"].get("Rates") != None , "Rate not set please check cfg!!!"
                        
                        self.__LOG.log("ori set TrainDatas is %s"%self.__config["TrainDataSetting"]["List"])

                        # check_list and check_rates
                        check_list = []
                        for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])):
                            if data_add in self.__config:
                                check_list.append(data_add)
                            else:
                                self.__LOG.log("TrainDatas %s is not set,will be removed"%data_add)
                        
                        # remove zeros
                        dataset_init_num = self.__config["DataSetting"].getint("InitSent")
                        check_list_rmzero = []
                        check_rates_rmzero = []

                        for ind, data_add in enumerate(check_list):
                            self.__config[data_add]["InitSent"] = str(self.__config[data_add].getint("InitSent"))  if self.__config[data_add].getint("InitSent") else "0"
                            dataset_init_num_ = self.__config[data_add].getint("InitSent")
                            
                            print(ind,data_add,dataset_init_num_)
                            rate_add = dataset_init_num_ / dataset_init_num
                            if rate_add != 0:
                                check_list_rmzero.append(data_add)
                                check_rates_rmzero.append(rate_add)
                            else:
                                self.__LOG.log("This rate is %s pfile:%s will be removed"%(rate_add,data_add))
                    
                        self.__config["TrainDataSetting"]["List"] = str(check_list_rmzero)
                        self.__config["TrainDataSetting"]["MixRates"] = str(check_rates_rmzero)
                        for ind, dl in enumerate(check_list_rmzero):

                            if dl not in self.data_infos.keys():
                                self.data_infos[dl] = {}
                            self.data_infos[dl]["mix_rate"] = check_rates_rmzero[ind]
                            if "lmdb" == self.__config[dl].get('DataType'): 
                                aug_info_str_ = self.__config[dl].get("Aug")

                                print("aug_info_str_",dl,aug_info_str_)
                                self.data_infos[dl]["aug_info_str"] = aug_info_str_ if aug_info_str_ else {}
                        
                    
                else:
                    if None == self.__config["TrainDataSetting"].get("MixFlag"):
                        self.__config["TrainDataSetting"]["List"] = str([pfile.strip(" ") for pfile in self.__config["TrainDataSetting"].get("TrainDatas").split(",") if pfile != ""])
                        assert self.__config["TrainDataSetting"].get("Rates") != None , "Rate not set please check cfg!!!"
                        self.__config["TrainDataSetting"]["Rates"] = str([rate.strip(" ") for rate in self.__config["TrainDataSetting"].get("Rates").split(",") if rate != ""])
                        self.__LOG.log("ori set TrainDatas is %s"%self.__config["TrainDataSetting"]["List"])
                        self.__LOG.log("ori set MixRates is %s"%self.__config["TrainDataSetting"]["Rates"])

                        check_list = []
                        for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])):
                            if data_add in self.__config:
                                check_list.append(data_add)
                            else:
                                self.__LOG.log("TrainDatas %s is not set,will be removed"%data_add)
                        
                        check_rates = []
                        for ind, rate_add in enumerate(eval(self.__config["TrainDataSetting"]["Rates"])):
                            if 0 == ind:
                                if 1.0 != eval(rate_add):
                                    self.__LOG.log("MixRates[0] %s most be 1, please check"%rate_add)
                                    print("MixRates[0] %s most be 1, please check"%rate_add)
                            if type(eval(rate_add)) != type("ssyan2") and (eval(rate_add) == -1 or eval(rate_add) >=0):
                                check_rates.append(eval(rate_add))
                            else:
                                self.__LOG.log("MixRates %s most be number -1 or >=0,please check cfg"%rate_add)
                                print("MixRates %s most be number -1 or >=0,please check cfg"%rate_add)
                                exit()
                        
                        assert len(check_list) == len(check_rates),"please make sure check_list:%s and check_rates:%s have same len"%(check_list,check_rates)
                        
                        # remove zeros
                        check_list_rmzero = []
                        check_rates_rmzero = []
                        for ind, data_add in enumerate(check_list):
                            rate_add = check_rates[ind]
                            if rate_add != 0:
                                check_list_rmzero.append(data_add)
                                check_rates_rmzero.append(rate_add)
                            else:
                                self.__LOG.log("This rate is %s pfile:%s will be removed"%(rate_add,data_add))
                    
                        self.__config["TrainDataSetting"]["List"] = str(check_list_rmzero)
                        self.__config["TrainDataSetting"]["MixRates"] = str(check_rates_rmzero)
                        self.__config["TrainDataSetting"]["MixFlag"] = "ok"

                        
                        for ind, dl in enumerate(check_list_rmzero):
                            if dl not in self.data_infos.keys():
                                self.data_infos[dl] = {}
                            self.data_infos[dl]["mix_rate"] = check_rates_rmzero[ind]
                            if "lmdb" == self.__config[dl].get('DataType'): 
                                aug_info_str_ = self.__config[dl].get("Aug")
                                self.data_infos[dl]["aug_info_str"] = aug_info_str_ if aug_info_str_ else ""
                    else:
                        for ind, dl in enumerate(eval(self.__config["TrainDataSetting"]["List"])):
                            if dl not in self.data_infos.keys():
                                self.data_infos[dl] = {}
                            self.data_infos[dl]["mix_rate"] = float(eval(self.__config["TrainDataSetting"]["MixRates"])[ind])
                            if "lmdb" == self.__config[dl].get('DataType'): 
                                aug_info_str_ = self.__config[dl].get("Aug")
                                self.data_infos[dl]["aug_info_str"] = aug_info_str_ if aug_info_str_ else ""
                        

                
                self.__LOG.log("After check TrainDatas is %s, MixRates is %s (-1 use all)"%(self.__config["TrainDataSetting"]["List"],self.__config["TrainDataSetting"]["MixRates"]))

                print("After check TrainDatas is %s, MixRates is %s (-1 use all)"%(self.__config["TrainDataSetting"]["List"],self.__config["TrainDataSetting"]["MixRates"]))
                self.data_infos["data_list"] = eval(self.__config["TrainDataSetting"]["List"])
                if "aug_sets" not in self.data_infos:
                    self.data_infos["aug_sets"] = {}
                
                self.data_infos["aug_sets"]["reverb"] = self.__config["TrainDataSetting"].get("ReverbFile")
                self.data_infos["aug_sets"]["reverb_noises"] = self.__config["TrainDataSetting"].get("ReverbNoisePaths")
                self.data_infos["aug_sets"]["only_noises"] = self.__config["TrainDataSetting"].get("NoisePaths")

                # get BatchCtrl flag
                if self.__config["TrainDataSetting"].get("BatchCtrl") != None and  self.__config["TrainDataSetting"].getboolean("BatchCtrl"):
                    self.__config["TrainDataSetting"]["BatchCtrl"] = str(True)
                else:
                    self.__config["TrainDataSetting"]["BatchCtrl"] = str(False)
                self.__LOG.log("After check BatchCtrl is %s "%self.__config["TrainDataSetting"]["BatchCtrl"])

                print("After check BatchCtrl is %s"%self.__config["TrainDataSetting"]["BatchCtrl"])
            
            # get dist setting
            for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                self.data_infos[data_add]["data_type"] = self.__config[data_add].get('DataType')

                r_ = get_current_part_index_and_sent_add(self.__config,data_add,dist_ind,iteration)
                self.data_infos[data_add]["r"] = r_
    
                print("TrainDataSetting r",data_add,r_)

            
            
            # get add train pfile part info
            for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                #r1 = get_current_part_index_and_sent_add(self.__config,data_add)
                
                r_ = self.data_infos[data_add]["r"]
                part_numx_ = r_[2]

                if "pfile" == self.__config[data_add].get('DataType'):
                    data_dir_ = self.__config[data_add]["DataDir"]
                    data_dir_lab_ = self.__config[data_add]["DataDirLab"] # xiaobao
                    current_label_pfile_ = self.__config[data_add]["LabelPrefix"].replace('$', str(part_numx_))
                    current_feature_pfile_ = self.__config[data_add]["FeaturePrefix"].replace('$', str(part_numx_))
                        
                    current_label_pfile_ = os.path.join(data_dir_lab_, current_label_pfile_) # xiaobao
                    current_feature_pfile_ = os.path.join(data_dir_, current_feature_pfile_)
                    
                    if not os.path.exists(current_label_pfile_):
                        self.__raise_err("{0} doesn't exist".format(current_label_pfile_))
                    if not os.path.exists(current_feature_pfile_):
                        self.__raise_err("{0} doesn't exist".format(current_feature_pfile_))
                
                    #current_label_pfile_adds.append(current_label_pfile1)
                    #current_feature_pfile_adds.append(current_feature_pfile1)
                    self.data_infos[data_add]["pfile_fea"] = current_feature_pfile_
                    self.data_infos[data_add]["pfile_lab"] = current_label_pfile_
                
                elif "lmdb" == self.__config[data_add].get('DataType'):
                    Lmdb_dir_ = self.__config[data_add]["LmdbDir"]
                    current_lmdb_ = self.__config[data_add]["LmdbPrefix"].replace('$', str(part_numx_))
                    current_lmdb_ = os.path.join(Lmdb_dir_, current_lmdb_) # ssyan2

                    if not os.path.exists(current_lmdb_):
                        self.__raise_err("{0} doesn't exist".format(current_lmdb_))
                    #current_lmdb_adds.append(current_lmdb1)
                    self.data_infos[data_add]["lmdb_file"] = current_lmdb_
            ########### add by ssyan2 end

        if sent == "all":
            start_sent = 0
            
            
            if self.__config['TrainSetting'].get('TrainType') == "LM":
                end_sent=0
                with open(current_text_file) as f:
                    for l in f:
                        end_sent+=1
            if self.__config['TrainSetting'].get('TrainType') == "ASR":
                if "pfile" == self.__config["DataSetting"].get('DataType'):
                    end_sent = int(PfileInfo(current_feature_pfile).num_sentences)
                elif "lmdb" == self.__config["DataSetting"].get('DataType'):
                    end_sent = int(LmdbInfo(current_lmdb).num_sentences)

        else:
            start_sent, end_sent = sent.split('-')
            start_sent = int(start_sent)
            end_sent = int(end_sent)
        
        self.data_infos["DataSetting"]["start_sent"] = start_sent
        self.data_infos["DataSetting"]["end_sent"] = end_sent
        ########### add by ssyan2 start
        print("self.data_infos",self.data_infos)
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            for ind,data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                r_ = self.data_infos[data_add]["r"]
                start_sent_ = 0 if "all" == r_[3] else int(r_[3].split("-")[0])
                if "pfile" == self.__config[data_add].get('DataType'):
                    end_sent_ = int(PfileInfo(self.data_infos[data_add]["pfile_fea"]).num_sentences)  if "all" == r_[3] else int(r_[3].split("-")[1])
                elif "lmdb" == self.__config[data_add].get('DataType'):
                    end_sent_ = int(LmdbInfo(self.data_infos[data_add]["lmdb_file"]).num_sentences)  if "all" == r_[3] else int(r_[3].split("-")[1])
                
                self.data_infos[data_add]["start_sent"] = start_sent_
                self.data_infos[data_add]["end_sent"] = end_sent_
        
        validation_sentnum = self.__config["DataSetting"].getint("CVSentNum")
        self.data_infos["DataSetting"]["CVSentNum"] = validation_sentnum

        if not validation_sentnum > 0:
            self.__raise_err("CVSentNum must be a positive value")
        
        ########### add by ssyan2 start
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            # only asr multi input support
            
            for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                validation_sentnum_ = self.__config[data_add].getint("CVSentNum")
                
                if validation_sentnum_:
                    if validation_sentnum_ < 0:
                        self.__raise_err("%s CVSentNum must be a positive value"%data_add)
                else:
                    validation_sentnum_ = 0
                
                self.data_infos[data_add]["CVSentNum"] = validation_sentnum_
        ########### add by ssyan2 end

        
        if self.__config['TrainSetting'].get('TrainType') == "LM":
            validation_datadir = self.__config["NNLMSetting"]["ValidationDatadir"]
            validation_text_file = self.__config["NNLMSetting"]["ValidationText"]
            validation_text_file = os.path.join(validation_datadir, validation_text_file)
            if not os.path.exists(validation_text_file):
                self.__raise_err("{0} doesn't exist".format(validation_text_file))
            validation_start_sent = 0
            validation_end_sent = validation_sentnum

        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            if self.__config["DataSetting"].get("ValidationDatadir"):
                validation_datadir = self.__config["DataSetting"]["ValidationDatadir"]
                if not (self.__config["DataSetting"].get("ValidationLabel")):
                    self.__raise_err("validation label isn't specified")
                if not (self.__config["DataSetting"].get("ValidationFeature")):
                    self.__raise_err("validation feature isn't specified")
                validation_label_pfile = self.__config["DataSetting"]["ValidationLabel"]
                validation_feature_pfile = self.__config["DataSetting"]["ValidationFeature"]
                validation_label_pfile = os.path.join(validation_datadir, validation_label_pfile)
                validation_feature_pfile = os.path.join(validation_datadir, validation_feature_pfile)
                if not os.path.exists(validation_label_pfile):
                    self.__raise_err("{0} doesn't exist".format(
                        validation_label_pfile))
                if not os.path.exists(validation_feature_pfile):
                    self.__raise_err("{0} doesn't exist".format(
                        validation_feature_pfile))
                validation_start_sent = 0
                validation_end_sent = validation_sentnum
            else:
                if "pfile" == self.__config["DataSetting"].get('DataType'):
                    validation_label_pfile = current_label_pfile
                    validation_feature_pfile = current_feature_pfile
                    self.data_infos["DataSetting"]["val_pfile_fea"] = validation_feature_pfile
                    self.data_infos["DataSetting"]["val_pfile_lab"] = validation_label_pfile
                elif "lmdb" == self.__config["DataSetting"].get('DataType'):
                    validation_lmdb = current_lmdb
                    self.data_infos["DataSetting"]["val_lmdb_file"] = validation_lmdb
                
                if not (validation_sentnum <= end_sent):
                    self.__raise_err(
                        "CVSentNum must be smaller than train sentnum")
                end_sent = end_sent - validation_sentnum
                validation_start_sent = end_sent
                validation_end_sent = validation_start_sent + validation_sentnum

                self.data_infos["DataSetting"]["end_sent"] = end_sent
                self.data_infos["DataSetting"]["val_start_sent"] = validation_start_sent
                self.data_infos["DataSetting"]["val_end_sent"] = validation_end_sent


                ########### add by ssyan2 start

                for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                    if "pfile" == self.__config[data_add].get('DataType'):
                        validation_feature_pfile_ = self.data_infos[data_add]["pfile_fea"]
                        validation_label_pfile_ = self.data_infos[data_add]["pfile_lab"]
                        self.data_infos[data_add]["val_pfile_fea"] = validation_feature_pfile_
                        self.data_infos[data_add]["val_pfile_lab"] = validation_label_pfile_
                    elif "lmdb" == self.__config[data_add].get('DataType'):
                        validation_lmdb_ = self.data_infos[data_add]["lmdb_file"]
                        self.data_infos[data_add]["val_lmdb_file"] = validation_lmdb_
                    
                    end_sent_ = self.data_infos[data_add]["end_sent"] - self.data_infos[data_add]["CVSentNum"]
                    validation_start_sent_ = end_sent_
                    validation_end_sent_ = validation_start_sent_ + self.data_infos[data_add]["CVSentNum"]
                    self.data_infos[data_add]["end_sent"] = end_sent_
                    self.data_infos[data_add]["val_start_sent"] = validation_start_sent_
                    self.data_infos[data_add]["val_end_sent"] = validation_end_sent_
                
                ########### add by ssyan2 end

            bunchsize = self.__config["TrainDataSetting"].getint("Bunchsize")
            padnum = self.__config["TrainDataSetting"].getint("PadNum")
            maxsentframe = self.__config["TrainDataSetting"].getint("MaxSentFrame")
            maxnumsent = self.__config["TrainDataSetting"].getint("MaxNumSent")

            

        if self.__config['TrainSetting'].get('TrainType') == "LM":
            cnt=0
            with open(current_text_file) as f:
                for l in f:
                    cnt+=1
            train_iternum = cnt // self.__config["NNLMSetting"].getint("BatchSize")
        
            
                
        validation_iternum = int(validation_end_sent - validation_start_sent)

        self.data_infos["DataSetting"]["validation_iternum"] = validation_iternum

        ########### add by ssyan2 start
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            # only asr multi input support
            for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                self.data_infos[data_add]["validation_iternum"] = self.data_infos[data_add]["val_end_sent"] - self.data_infos[data_add]["val_start_sent"]
                    
        ########### add by ssyan2 end
        
        if self.__config['TrainSetting'].get('TrainType') == "LM":
            self.__config["NNLMSetting"]["TrainTextfile"] = current_text_file
            self.__config["NNLMSetting"]["ValidationTextfile"] = validation_text_file
            self.__config["NNLMSetting"]["ValidationStartSent"] = str(validation_start_sent)
            self.__config["NNLMSetting"]["ValidationEndSent"] = str(validation_end_sent)
            self.__config["NNLMSetting"]["TrainStartSent"] = str(start_sent)
            self.__config["NNLMSetting"]["TrainEndSent"] = str(end_sent)
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            
            
            mix_rates = [1.]
            for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                # remove first
                full_use_flag = True if -1 == self.data_infos[data_add]["mix_rate"] else False
                mix_sent_num = self.data_infos[data_add]["end_sent"] - self.data_infos[data_add]["start_sent"]
                main_sent_num = self.data_infos["DataSetting"]["end_sent"] - self.data_infos["DataSetting"]["start_sent"]
                if True == full_use_flag:
                    mix_rate = mix_sent_num / main_sent_num
                else:
                    mix_rate = self.data_infos[data_add]["mix_rate"]
                    assert mix_rate > 0,"pfile %s mix_rate <= 0 please check"%data_add
                
                mix_rates.append(round(mix_rate,3))
                self.data_infos[data_add]["mix_rate"] = round(mix_rate,3)
    
            self.__config["TrainDataSetting"]["Rate"] = str(mix_rates)
            self.__config["TrainDataSetting"]["BatchRate"] = str([round(mr/min(mix_rates)) for mr in  mix_rates])
            

            ########### add by ssyan2 end
        

        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            train_iternums = []
            if self.__config["DataSetting"].get('DataType') == "pfile":
                train_iternum = PfileInfo.estimate_num_batch(PfileInfo(current_feature_pfile).seq_info[start_sent: end_sent], round(bunchsize*1/(sum(mix_rates))), maxsentframe, maxnumsent, padnum)
            elif self.__config["DataSetting"].get('DataType') == "lmdb":
                train_iternum = LmdbInfo.estimate_num_batch(LmdbInfo(current_lmdb).seq_info[start_sent: end_sent], round(bunchsize*1/(sum(mix_rates))), maxsentframe, maxnumsent, padnum)
            train_iternums.append(train_iternum)
            self.data_infos["DataSetting"]["train_iternum"] = train_iternum

            for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                start_sent_ = self.data_infos[data_add]["start_sent"]
                end_sent_ = self.data_infos[data_add]["end_sent"]
                mix_rate_ = self.data_infos[data_add]["mix_rate"]
                
                if self.__config[data_add].get('DataType') == "pfile":
                    current_feature_pfile_ =  self.data_infos[data_add]["pfile_fea"]
                    train_iternum_ = PfileInfo.estimate_num_batch(PfileInfo(current_feature_pfile_).seq_info[start_sent_: end_sent_], round(bunchsize*mix_rate_/(sum(mix_rates))), maxsentframe, maxnumsent, padnum)
                elif self.__config[data_add].get('DataType') == "lmdb":
                    current_lmdb_ =  self.data_infos[data_add]["lmdb_file"]
                    train_iternum_ = LmdbInfo.estimate_num_batch(LmdbInfo(current_lmdb_).seq_info[start_sent_: end_sent_], round(bunchsize*mix_rate_/(sum(mix_rates))), maxsentframe, maxnumsent, padnum)
                train_iternums.append(train_iternum_)
                self.data_infos[data_add]["train_iternum"] = train_iternum_


        # train_iternum = max(train_iternums)
        if self.__is_inited:
            train_iternum = train_iternum // self.__config["TrainSetting"].getint("NGpu")

        self.__config["DataSetting"]["TrainIterNum"] = str(train_iternum)
        self.__config["DataSetting"]["ValidationIterNum"] = str(validation_iternum)
        
        ########### add by ssyan2 start
        if self.__config['TrainSetting'].get('TrainType') == "ASR":
            # only asr multi input support
            for ind,data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                self.__config[data_add]["ValidationIterNum"] = str(self.data_infos[data_add]["validation_iternum"])
        ########### add by ssyan2 end
        

        if self.__rank == 0:
            log = ''
            log += get_log_header("Model and Data Summary")
            log += "previous model.. {0}\n".format(
                self.__config["TrainSetting"]["PreviousModel"])
            log += "current model.. {0}\n".format(
                self.__config["TrainSetting"]["CurrentModel"])
            if self.__config['TrainSetting'].get('TrainType') == "LM":
                log += "train data.. {0}, {1}-{2}\n".format(self.__config["NNLMSetting"]["TrainTextfile"],
                                                        self.__config["NNLMSetting"]["TrainStartSent"],
                                                        self.__config["NNLMSetting"]["TrainEndSent"])
                log += "validation data.. {0}, {1}-{2}\n".format(self.__config["NNLMSetting"]["ValidationTextfile"],
                                                             self.__config["NNLMSetting"]["ValidationStartSent"],
                                                             self.__config["NNLMSetting"]["ValidationEndSent"])
            if self.__config['TrainSetting'].get('TrainType') == "ASR":

                for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])):

                    if self.__config[data_add].get('DataType') == "pfile":
                        log += "train data.. {0}, {1}-{2}\n".format(self.data_infos[data_add]["pfile_fea"],
                                                                self.data_infos[data_add]["start_sent"],
                                                                self.data_infos[data_add]["end_sent"])
                        log += "train lab.. {0}, {1}-{2}\n".format(self.data_infos[data_add]["pfile_lab"],
                                                                self.data_infos[data_add]["start_sent"],
                                                                self.data_infos[data_add]["end_sent"]) 
                    if self.__config[data_add].get('DataType') == "lmdb":
                        log += "train lmdb.. {0}, {1}-{2}\n".format(self.data_infos[data_add]["lmdb_file"],
                                                                self.data_infos[data_add]["start_sent"],
                                                                self.data_infos[data_add]["end_sent"])
                
                for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])):

                    if self.__config[data_add].get('DataType') == "pfile":
                        log += "val data.. {0}, {1}-{2}\n".format(self.data_infos[data_add]["val_pfile_fea"],
                                                                self.data_infos[data_add]["val_start_sent"],
                                                                self.data_infos[data_add]["val_end_sent"])
                        
                    if self.__config[data_add].get('DataType') == "lmdb":
                        log += "val lmdb.. {0}, {1}-{2}\n".format(self.data_infos[data_add]["val_lmdb_file"],
                                                                self.data_infos[data_add]["val_start_sent"],
                                                                self.data_infos[data_add]["val_end_sent"])
                    
                log += "norm file.. {0}\n".format(
                self.data_infos["file_norm"])
            log += "estimate train iternum : "
            for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])):

                log += "{0}-{1}\t ".format(
                    data_add,self.data_infos[data_add]["train_iternum"]//self.__config["TrainSetting"].getint("NGpu"))
            log += "\ntrain iternum.. {0}\n".format(
                self.__config["DataSetting"].getint("TrainIterNum"))

            
            log += "validation iternum.. {0}\n".format(
                self.__config["DataSetting"].getint("ValidationIterNum"))
            

            ########### add by ssyan2 start
            print("self.data_infos[data_add]",self.data_infos)
            if self.__config['TrainSetting'].get('TrainType') == "ASR":
                # only asr multi input support
                for ind, data_add in enumerate(eval(self.__config["TrainDataSetting"]["List"])[1:]):
                    if self.data_infos[data_add]["validation_iternum"] > 0:
                        log += "validation add "+ data_add + " iternum.. {0}\n".format(
                            self.data_infos[data_add]["validation_iternum"])
                log += "train data rate ...%s\n"%{self.__config["TrainDataSetting"]["Rate"]}
                if self.__config["TrainDataSetting"].getboolean("BatchCtrl"):
                    log += "train data batch rate ...%s\n"%{self.__config["TrainDataSetting"]["BatchRate"]}
            ########### add by ssyan2 end
            
            # self.__config["TrainDataSetting"]["data_infos"] = str(self.data_infos)

            self.__LOG.log(log)
        self.__config["TrainDataSetting"]["data_infos"] = str(self.data_infos)

        half_iter_idx = self.__config["TrainSetting"].getint("Half")
        if iter_idx == "init":
            current_lr = self.__config["TrainSetting"].getfloat(
                "InitLearningRate")
        else:
            if iter_idx < half_iter_idx:
                current_lr = self.__config["TrainSetting"].getfloat(
                    "LearningRate")
            else:
                current_lr = self.__config["TrainSetting"].getfloat("LearningRate") / (
                    2 ** (iter_idx - half_iter_idx + 1))
        self.__config["TrainSetting"]["CurrentLearningRate"] = str(current_lr)

        if self.__rank == 0:
            log = ''
            log += get_log_header("Train Setting Summary")
            log += "optimizer.. {0}\n".format(
                self.__config["TrainSetting"]["Optimizer"])
            log += "use lookahead.. {0}\n".format(
                self.__config["TrainSetting"].getboolean("LookAhead"))
            if self.__config["TrainSetting"].getboolean("LookAhead"):
                log += "lookahead parameters.. K={0} alpha = {1}\n".format(
                    self.__config["TrainSetting"].getint("LookAhead_K"),
                    self.__config["TrainSetting"].getfloat("LookAhead_Alpha"))
            log += "learning rate.. {0}".format(
                self.__config["TrainSetting"].getfloat("CurrentLearningRate"))
            self.__LOG.log(log)

    def __loop(self):
        while True:
            self.__reset_current_log(delete=False)
            self.__init_train()
            if self.__done:
                break
            self.__train_next_part()
        self.__clean_activites()

    def __clean_activites(self):
        if self.__message is not None:
            self.__message.shutdown()

    def __train_next_part(self):
        if not self.__is_inited:
            if (self.__rank == 0 and self.__distributed):
                self.__config["TrainSetting"]["GlobalRank"] = str(0)
                self.__config["TrainSetting"]["LocalRank"] = str(0)
                message_queue = context.Queue()
                p = context.Process(target=single_gpu_warper, args=(
                    self.__config, message_queue, self.__LOG,))
                p.start()
                status = message_queue.get()
                p.join()
                if status == 0:
                    self.__LOG.log("train process failed")
                    self.__raise_err("train process failed")
                self.__message.wait()
            if (self.__rank != 0 and self.__distributed):
                # pcli2: other workers will wait main worker done.
                self.__message.wait()
                self.__LOG.log(
                    "worker {0} received model initializing finished signal from master worker".format(self.__rank))
            if (not self.__distributed):
                self.__config["TrainSetting"]["GlobalRank"] = str(0)
                self.__config["TrainSetting"]["LocalRank"] = str(0)
                message_queue = context.Queue()
                p = context.Process(target=single_gpu_warper, args=(
                    self.__config, message_queue, self.__LOG))
                p.start()
                status = message_queue.get()
                p.join()
                if status == 0:
                    self.__LOG.log("train process failed")
                    self.__raise_err("train process failed")
        else:
            if self.__distributed:
                self.__config["TrainSetting"]["InitMethod"] = "tcp://{0}:{1}".format(
                    self.__config["TrainSetting"]["MasterAddr"], self.__config["TrainSetting"]["MasterPort"])
                num_gpu_per_node = int(
                    self.__config["TrainSetting"].getint("NGpu") / self.__num_worker)
                process = []
                message_queue = context.Queue()
                for gpu in range(num_gpu_per_node):
                    self.__config["TrainSetting"]["GlobalRank"] = str(
                        num_gpu_per_node * self.__rank + gpu)
                    self.__config["TrainSetting"]["LocalRank"] = str(gpu)
                    p = context.Process(target=multi_gpu_warper, args=(
                        self.__config, message_queue, self.__LOG))
                    p.start()
                    process.append(p)

                status = 1
                for i in range(num_gpu_per_node):
                    status = status * message_queue.get()
                for p in process:
                    p.join()
                if status == 0:
                    self.__LOG.log("train process failed")
                    self.__raise_err("train process failed")

                self.__message.wait()
            else:
                if self.__config["TrainSetting"].getint("NGpu") > 1:
                    self.__config["TrainSetting"]["InitMethod"] = "tcp://{0}:{1}".format(
                        self.__config["TrainSetting"]["MasterAddr"], self.__config["TrainSetting"]["MasterPort"])
                    num_gpu_per_node = int(
                        self.__config["TrainSetting"].getint("NGpu") / self.__num_worker)
                    process = []
                    message_queue = context.Queue()
                    for gpu in range(num_gpu_per_node):
                        self.__config["TrainSetting"]["GlobalRank"] = str(
                            num_gpu_per_node * self.__rank + gpu)
                        self.__config["TrainSetting"]["LocalRank"] = str(gpu)
                        p = context.Process(target=multi_gpu_warper, args=(
                            self.__config, message_queue, self.__LOG))
                        p.start()
                        process.append(p)

                    status = 1
                    for i in range(num_gpu_per_node):
                        status = status * message_queue.get()
                    for p in process:
                        p.join()
                    if status == 0:
                        self.__LOG.log("train process failed")
                        self.__raise_err("train process failed")
                if self.__config["TrainSetting"].getint("NGpu") == 1:
                    self.__config["TrainSetting"]["GlobalRank"] = str(0)
                    self.__config["TrainSetting"]["LocalRank"] = str(0)
                    message_queue = context.Queue()
                    p = context.Process(target=single_gpu_warper, args=(
                        self.__config, message_queue, self.__LOG))
                    p.start()
                    status = message_queue.get()
                    p.join()
                    if status == 0:
                        self.__LOG.log("train process failed")
                        self.__raise_err("train process falied")

    def __raise_err(self, info):
        self.__clean_activites()
        raise Exception(info)

    def __check_config(self):
        status = gen_part_list(self.__config)
        if isinstance(status, str):
            self.__raise_err(status)

        try:
            NGpu = self.__config["TrainSetting"].getint("NGpu")
            assert NGpu > 0
        except Exception:
            self.__raise_err("NGpu must be a integers")

        try:
            Iter = self.__config["TrainSetting"].getint("Iter")
            assert Iter > 0
            Half = self.__config["TrainSetting"].getint("Half")
            assert Half > 0
        except Exception:
            self.__raise_err("Iter and Half must be integers")

        try:
            InitLearningRate = self.__config["TrainSetting"].getfloat(
                "InitLearningRate")
            assert InitLearningRate > 0
            LearningRate = self.__config["TrainSetting"].getfloat(
                "LearningRate")
            assert LearningRate > 0
        except Exception:
            self.__raise_err(
                "InitLearningRate and LearningRate must be floats")

        try:
            Optimizer = self.__config["TrainSetting"]["Optimizer"]
            assert Optimizer == "SGD" or Optimizer == "ADAM"
        except Exception:
            self.__raise_err("Optimizer must be SGD or ADAM")

