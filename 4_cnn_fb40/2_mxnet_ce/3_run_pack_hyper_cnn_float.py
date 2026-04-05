#!/usr/local/bin/python
#coding=utf-8
import os,sys
import shutil
import re

os.environ['PYTHONPATH'] = ''  #### 取消python3的环境变量,以免对后续调用python2脚本产生影响；

for i in [22]:
    jsonfile     = r'\\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\json\hyper_cnn_float.s9004_4i4o.json'
    dfAcResPara  = r'\\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\dfAcResPara\dfAcResPara.fb40_4i4o_s9004.cfg'
    fea_norm     = r'\\smb.dlp.com\work1\asrdictt\taoyu\mlg\slovenian\am\4_cnn_fb40\1_down_pfile\lib_fb40\fea.norm'
    states_count = r'\\smb.dlp.com\work1\asrdictt\taoyu\mlg\slovenian\am\4_cnn_fb40\1_down_pfile\lib_fb40\states.count.low100.txt'
    mlp_weight   = r'\\smb.dlp.com\work1\asrdictt\taoyu\mlg\slovenian\am\4_cnn_fb40\2_mxnet_ce\mlp\dcnn-0-{:04d}.params'.format(i)
    dir_out      = r'\\smb.dlp.com\work1\asrdictt\taoyu\mlg\slovenian\am\4_cnn_fb40\2_mxnet_ce\mlp\out'
    res_name     = r'ac_car_sl_taoyu_cnn_1.3kh_aug_e{}.bin'.format(i)

    assert os.path.isfile(jsonfile), 'No exist file: '+jsonfile
    assert os.path.isfile(dfAcResPara), 'No exist file: '+dfAcResPara
    assert os.path.isfile(fea_norm), 'No exist file: '+fea_norm
    assert os.path.isfile(states_count), 'No exist file: '+states_count
    assert os.path.isfile(mlp_weight), 'No exist file: '+mlp_weight

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    os.system(r'C:\Anaconda\python \\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\MeanVarPriBuilder.py norm {} {}\norm.params'.format(fea_norm, dir_out))
    os.system(r'C:\Anaconda\python \\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\MeanVarPriBuilder.py pri {0} {1}\pri.params 4 >{1}\MeanVarPriBuilder.pri.log'.format(states_count, dir_out))
    os.system(r'C:\Anaconda\python \\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\mxnet_merge.py {0}\norm.params,{0}\pri.params,{1} {0}\hypercnn.params'.format(dir_out, mlp_weight))
    os.system(r'\\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\pack_res_to_bin.exe {0}\hypercnn.params {0}\hypercnn.params.bin -t MXNET_PARAM -z false -e true'.format(dir_out))
    os.system(r'\\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\pack_res_to_bin.exe {1} {0}\hypercnn.json.bin -t MXNET_JSON -z true -e true'.format(dir_out, jsonfile))
    os.system(r'\\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\pack_res_concat.exe {0}\hypercnn.params.bin,{0}\hypercnn.json.bin {0}\hypercnn_mxnet_res.bin -t MXNET_RES -e true'.format(dir_out))
    os.system(r'\\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\pack_res_to_bin.exe {1} {0}\dfAcResPara.bin -t MXNET_CFG -z false -e false'.format(dir_out, dfAcResPara))
    os.system(r'\\smb.dlp.com\asrsmb\asr\taoyu\ACPack\ACPackCNN\bin\pack_res_concat.exe {0}\dfAcResPara.bin,{0}\hypercnn_mxnet_res.bin {0}\{1} -t MXNET_RES_WITH_CFG -e false'.format(dir_out, res_name))
