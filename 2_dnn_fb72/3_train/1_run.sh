####注意：ring_layer工具有ring_layer-{M,P,V,A}等版本，在M40机器上用工具ring_layer-M，以此类推；crossValid工具一样；

ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d train-ko -e 'train_ring_layer_lowrank_b1024_jump2-2-2-4.pl' -l train_ring_layer_lowrank_b1024_jump2-2-2-4.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -w 1 -g 4 --useDist --useGpu -k TeslaP40 -r dlp3-asrdictt-car-reserved
ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d train-ko -e 'train_ring_layer_lowrank_b1024_jump2-2-2-4.pl' -l train_ring_layer_lowrank_b1024_jump2-2-2-4.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -w 1 -g 4 --useDist --useGpu -k TeslaM4024GB -r dlp3-asrdictt-car-reserved
ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d train-ko -e 'train_ring_layer_lowrank_b1024_jump2-2-2-4.pl' -l train_ring_layer_lowrank_b1024_jump2-2-2-4.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -w 1 -g 4 --useDist --useGpu -k TeslaM40 -r dlp3-asrdictt-car-reserved


ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d train-ko -e 'train_ring_layer_lowrank_b1024_jump2-2-2-4.pl' -l train_ring_layer_lowrank_b1024_jump2-2-2-4.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -w 1 -g 8 --useDist --useGpu -k TeslaV100-PCIE-24GB -r dlp3-asrdictt-car-reserved

ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d train-ko -e 'train_ring_layer_lowrank_b1024_jump2-2-2-4.pl' -l train_ring_layer_lowrank_b1024_jump2-2-2-4.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -w 1 -g 4 --useDist --useGpu -k TeslaV100-PCIE-12GB
