# 多机多卡

ky exp submit PtJob --modelName train --modelPath /work1/asrdictt/taoyu/tmp -m 300 -n train-ko -d ko_ce_ctc -e 'train.sh config_ce.ini config_ctc_init_ce.ini' -l train_ctc_init_ce.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -w 2 -g 4 --useGpu -k TeslaV100-PCIE-16GB -r dlp3-asrdictt-car-reserved
