gpu_type=$(nvidia-smi -L | head -n 1 | cut -d'(' -f 1 | cut -d':' -f 2 | sed -e 's/^\s\+//' -e 's/\s\+$//')
if [ "$gpu_type" = 'NVIDIA GeForce RTX 3090' -o "$gpu_type" = 'NVIDIA A40' ]; then
    ln -s -f c-a.so asr/c.so
    source /home3/asrdictt/taoyu/conda_zhyou2_pth39cu111tch191.bashrc ####A100
else
    ln -s -f c-v.so asr/c.so
    source /home3/asrdictt/taoyu/conda_zhyou2_pth39cu102tch191.bashrc ####V100
fi

if [ ! $NGPU_PER_WORKER ]; then export CUDA_VISIBLE_DEVICES=0 ; fi  #### if running in local machine, then set gpu id;

echo $CUDA_VISIBLE_DEVICES
echo `which python`
echo $LD_LIBRARY_PATH

/work1/asrdictt/taoyu/sbin/wait_file.pl /work1/asrdictt/taoyu/mlg/hindi/am/3_ub_fb40/1_down_pfile/lib_fb40/fea.norm.done
/work1/asrdictt/taoyu/sbin/wait_time.pl 1

for cfg in $@
do
echo "training for config $cfg"
python train.py $cfg
done
