ky exp submit MpiJob --modelName train --modelPath /work1/asrdictt/taoyu/tmp -n train-cnn -d cnn-ko-ini -e 'train_init.pl -1' -l train_init.log -i reg.deeplearning.cn/ayers/sp_dist:test1.3  --useGpu --useDist -w 1 -g 1 -k TeslaV100-PCIE-16GB -r dlp3-asrdictt-car-reserved 

ky exp submit MpiJob --modelName train --modelPath /work1/asrdictt/taoyu/tmp -n train-cnn -d cnn-ko -e 'mpi_run.pl -1' -l mpi_run.log -i reg.deeplearning.cn/ayers/sp_dist:test1.3 --useDist --useGpu -w 4 -g 4 -k TeslaV100-PCIE-32GB -r dlp3-asrdictt-car-reserved -s 'dlp2-99-220 dlp2-99-221 dlp2-99-232 dlp2-99-250'

ky exp submit MpiJob --modelName train --modelPath /work1/asrdictt/taoyu/tmp -n train-cnn -d cnn-ko -e 'train.sh -1' -l mpi_run.log -i reg.deeplearning.cn/ayers/sp_dist:test1.3 --useDist --useGpu -w 2 -g 8 -k TeslaV100-PCIE-48GB -r dlp3-asrdictt-car-reserved -s 'dlp2-98-170 dlp2-98-171'

