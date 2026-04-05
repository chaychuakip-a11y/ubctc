####注意：Pretrain工具有Pretrain-{M,P,V,A}等版本，在M40机器上用工具Pretrain-M，以此类推；

ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d pretrain-ko -e '2_C_Pretrain_jump8.cw11_h2048.pl' -l 2_C_Pretrain_jump8.cw11_h2048.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -g 1 -w 1 --useDist --useGpu -k TeslaA100-PCIE-24GB -r dlp3-asrdictt-reserved

ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d pretrain-ko -e '2_C_Pretrain_jump8.cw11_h2048.pl' -l 2_C_Pretrain_jump8.cw11_h2048.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -g 1 -w 1 --useDist --useGpu -k TeslaV100-PCIE-24GB -r dlp3-asrdictt-reserved

ky exp submit PtJob --isModelTest --modelPath /work1/asrdictt/taoyu/tmp -n train -d pretrain-ko -e '2_C_Pretrain_jump8.cw11_h2048.pl' -l 2_C_Pretrain_jump8.cw11_h2048.log -i reg.deeplearning.cn/ayers/nvidia-cuda:9.2-cudnn7-devel-centos7-py2 -g 1 -w 1 --useDist --useGpu -k TeslaP40 -r dlp3-asrdictt-reserved

