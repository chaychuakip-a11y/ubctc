# pcli2 2019 Nov.
import os
import importlib
import torch
import sys
import multiprocessing
import traceback
import hashlib
import numpy as np
from ..layers import SelfAttention, AddSOS


def gen_part_list(config):
    DistributionDict = {}
    print("Part in","PartDistribution" in config)
    assert "PartDistribution" in config, "PartDistribution is not set"
    initnum = config["DataSetting"].getint("InitSent")
    DistributionDict["Init"] = {}
    DistributionDict["Init"]['0'] = "{2}:{0}-{1}".format(0, initnum,0)
    PartDistribution = config["PartDistribution"]
    iternum = config["TrainSetting"].getint("Iter")
    for i in range(iternum):
        key = "Iteration" + str(i)
        value = PartDistribution.get(key)
        part_info = {}
        if value:
            value = value.replace(' ', '')
            part_list = value.split(',')
            step = 0
            for e in part_list:
                sent = "all"
                if ':' in e:
                    e, sent = e.split(':')
                    if '-' not in sent:
                        return "you should set start and end sent in part {0}, {1}".format(e, key)
                if '-' in e:
                    part_start, part_end = e.split('-')
                    for partidx in range(int(part_start), int(part_end) + 1):
                        if str(step) in part_info:
                            pass
                            #return "{0} appears more than one time in {1}".format(str(partidx), key)
                        part_info[str(step)] = str(partidx)+":"+sent
                        step += 1
                else:
                    if e in part_info:
                        pass
                        #return "part {0} appears more than one time in {1}".format(e, key)
                    part_info[str(step)] = str(e)+":"+sent
                    step += 1
        else:
            partnum = config["DataSetting"].getint("Npart")
            for i in range(partnum):
                part_info[str(i)] = str(i)+":"+"all"

        all_keys = part_info.keys()
        sorted_all_keys = sorted(all_keys, key=lambda partkey: int(partkey))
        sorted_dict = {}
        for partkey in sorted_all_keys:
            sorted_dict[partkey] = part_info[partkey]
        DistributionDict[key] = sorted_dict
    return DistributionDict

def gen_part_list_add(config,datasetting=""):
    MainDistributionDict = gen_part_list(config)
    DistributionDict = {}
    PartDistName = "PartDistribution" if "" == datasetting else "PartDistribution_"+datasetting
    print("Part in",PartDistName in config)
    assert PartDistName in config, "%s is not set"%PartDistName
    initnum = config[datasetting].getint("InitSent")
    DistributionDict["Init"] = {}
    DistributionDict["Init"]['0'] = "{2}:{0}-{1}".format(0, initnum,0)
    PartDistribution = config[PartDistName]
    iternum = config["TrainSetting"].getint("Iter")
    for i in range(iternum):
        key = "Iteration" + str(i)
        value = PartDistribution.get(key)
        part_info = {}
        if value:
            value = value.replace(' ', '')
            part_list = value.split(',')
            step = 0
            for e in part_list:
                sent = "all"
                if ':' in e:
                    e, sent = e.split(':')
                    if '-' not in sent:
                        return "you should set start and end sent in part {0}, {1}".format(e, key)
                if '-' in e:
                    part_start, part_end = e.split('-')
                    for partidx in range(int(part_start), int(part_end) + 1):
                        if str(step) in part_info:
                            pass
                            #return "{0} appears more than one time in {1}".format(str(partidx), key)
                        part_info[str(step)] = str(partidx)+":"+sent
                        step += 1
                else:
                    if e in part_info:
                        pass
                        #return "part {0} appears more than one time in {1}".format(e, key)
                    part_info[str(step)] = str(e)+":"+sent
                    step += 1
        else:
            partnum = config["DataSetting"].getint("Npart") if "" == datasetting else config[datasetting].getint("Npart")
            for i in range(partnum):
                part_info[str(i)] = str(i)+":"+"all"

        all_keys = part_info.keys()
        sorted_all_keys = sorted(all_keys, key=lambda partkey: int(partkey))
        main_sorted_all_keys = sorted(MainDistributionDict[key].keys(), key=lambda partkey: int(partkey))
        sorted_dict = {}
        for step, partkey in enumerate(main_sorted_all_keys):
            sorted_key = sorted_all_keys[step % len(sorted_all_keys)]
            sorted_dict[partkey] = part_info[sorted_key]
        DistributionDict[key] = sorted_dict
    return DistributionDict


def get_current_part_index_and_sent(config):
    outdir = config["TrainSetting"].get("OutDir")
    DistributionDict = gen_part_list(config)
    print("TrainSetting dist",DistributionDict)
    
    for iteration in DistributionDict:
        if iteration == "Init":
            iter_idx = "init"
            part_idx = 0
            modelname = os.path.join(outdir, "model.init")
            sent = DistributionDict[iteration]['0']
            part_numx, sent = sent.split(":")
            if not os.path.exists(modelname):
                return iteration,0,iter_idx, part_idx, part_numx, sent
        else:
            iter_idx = int(iteration.replace("Iteration", ''))
            for dist_ind,e in enumerate(DistributionDict[iteration]):
                modelname = os.path.join(
                    outdir, "model.iter{0}.part{1}".format(iter_idx, e))
                part_idx = int(e)
                sent = DistributionDict[iteration][e]
                part_numx, sent = sent.split(":")
                if not os.path.exists(modelname):
                    return iteration,dist_ind, iter_idx, part_idx, part_numx, sent #pengqi4
    return None

def get_current_part_index_and_sent_add(config,datasetting="",dist_ind=0,iteration="Init"):
    outdir = config["TrainSetting"].get("OutDir")
    DistributionDict = gen_part_list_add(config,datasetting)
    print("TrainSetting dist",datasetting,DistributionDict)
    
    if iteration == "Init":
        iter_idx = "init"
        part_idx = 0
        modelname = os.path.join(outdir, "model.init")
        sent = DistributionDict[iteration]['0']
        part_numx, sent = sent.split(":")
        if not os.path.exists(modelname):
            return iter_idx, part_idx, part_numx, sent
    else:
        iter_idx = int(iteration.replace("Iteration", ''))
        dist_ind %= len(DistributionDict[iteration].keys())
        e = list(DistributionDict[iteration].keys())[dist_ind]
        part_idx = int(e)
        sent = DistributionDict[iteration][e]
        part_numx, sent = sent.split(":")
        return iter_idx, part_idx, part_numx, sent #pengqi4
    return None


def get_previous_model_name(config):
    outdir = config["TrainSetting"].get("OutDir")
    DistributionDict = gen_part_list(config)
    modelname_last = None
    for iteration in DistributionDict:
        if iteration == "Init":
            iter_idx = "init"
            part_idx = 0
            modelname_current = os.path.join(outdir, "model.init")
            sent = DistributionDict[iteration]['0']
            if not os.path.exists(modelname_current):
                return modelname_last
            modelname_last = modelname_current
        else:
            iter_idx = int(iteration.replace("Iteration", ''))
            for e in DistributionDict[iteration]:
                modelname_current = os.path.join(
                    outdir, "model.iter{0}.part{1}".format(iter_idx, e))
                part_idx = int(e)
                sent = DistributionDict[iteration][e]
                if not os.path.exists(modelname_current):
                    return modelname_last
                modelname_last = modelname_current
    return modelname_last


def get_module(module_name):
    m_name, classname = module_name.split('.')
    m = importlib.import_module(m_name)
    module = getattr(m, classname)
    return module


def init_param(model, path, finetune_lm=False):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict["state_dict"],strict=False)
    #modified by mzwang7, for finetune ilm internal lm
    if finetune_lm:
        model.decoder.mlp_attention.embedding_lm.load_state_dict(model.decoder.mlp_attention.embedding.state_dict())
        model.decoder.mlp_attention.lm_lstm.load_state_dict(model.decoder.mlp_attention.att_lstm.state_dict())


def get_average_sentnum(bunchsize, padnum, labelpfile):
    cmd = "/home/asr/pcli2/scripts/pfile_info -p -q {0}".format(labelpfile)
    result = os.popen(cmd)
    sentences_length = []
    pfilelen = result.readlines()
    for line in pfilelen:
        line = line[0:-1]
        length = int(line.split()[1])
        sentences_length.append(length)
    result.close()

    totalsent = len(sentences_length)
    totalbunch = 0
    i = 0
    while i < totalsent:
        totalframe = 0
        readfinal = 0
        while sentences_length[i] + 2 * padnum > bunchsize:
            i += 1
            if i >= totalsent:
                readfinal = 1
                break
        if readfinal == 1:
            break
        while totalframe < bunchsize:
            totalframe += sentences_length[i] + 2 * padnum
            i += 1
            if i >= totalsent:
                readfinal = 1
                break
        if readfinal == 1:
            break
        if totalframe > bunchsize:
            i -= 1
        totalbunch += 1

    return float(totalsent) / float(totalbunch)


def get_total_sentnum(pfilefeature):
    cmd = "/home/asr/pcli2/scripts/pfile_info {0} | tail -n 1 | sed \"s: .*::\"".format(
        pfilefeature)
    result = os.popen(cmd)
    total_sent = float(result.read().replace('\n', ''))
    return total_sent


def save_model(model, modelpath):
    with open(modelpath, "wb") as f:
        torch.save({"state_dict": model.state_dict()}, f)


def printer(func):
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return 1
        except Exception:
            print(traceback.format_exc())
            return 0

    return handler


def get_port_id(master_port):
    idx = int(master_port) + 1
    return idx


def set_decode_mode(model, mode):
    for module in model.modules():
        if isinstance(module, SelfAttention):
            module.decode = bool(mode)
        if isinstance(module, AddSOS):
            module.decode = bool(mode)


