source /home/asr/kaishen2/bashrc-pytorch1.7.1-cuda10.2

#[4gbu]:alpha/bm=0.75/0.75||[8gpu]:alpha/bm=1.0/0.9||[12gpu]:alpha/bm=1.0/0.92||[16gpu]:alpha/bm=1.0/0.94
#[20gpu]:alpha/bm=1.0/0.954||[24gpu]]:alpha/bm=1.0/0.965||[28gpu]:alpha/bm=1.0/0.97||[32gpu]:alpha/bm=1.0/0.972;

python train.py \
    --lmdb_path '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/lmdb/lmdb0' \
    --lmdb_key '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/lmdb/lmdb0/keys_lens.txt' \
	--lmdbpb_path '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/lmdb_huifang/lmdb0'  \
	--lmdbpb_key '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/lmdb_huifang/lmdb0/keys_lens.txt'  \
    --lmdbuni_path '/train8/asrkws/liangdai/data_dir/universal_srf/lmdb0'  \
	--lmdbuni_key '/train8/asrkws/liangdai/data_dir/universal_srf/lmdb0/keys_lens.txt'  \
	--lmdbunipb_path '/train8/asrkws/liangdai/data_dir/universal_srf_pb/lmdb0'  \
	--lmdbunipb_key '/train8/asrkws/liangdai/data_dir/universal_srf_pb/lmdb0/keys_lens.txt'  \
    --lmdbnoise_path '/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb0,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb1,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb2,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb3,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb4,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb5,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb6,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb7'  \
	--lmdbnoise_key '/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb0/keys_lens.txt,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb1/keys_lens.txt,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb2/keys_lens.txt,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb3/keys_lens.txt,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb4/keys_lens.txt,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb5/keys_lens.txt,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb6/keys_lens.txt,/train8/asrkws/kaishen2/Data_kws/Universal_CHS_Noise/lmdb7/keys_lens.txt'  \
    --lmdbnone_path '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/fanli/lmdb0' \
    --lmdbnone_key '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/fanli/lmdb0/keys_lens.txt' \
    --cv_lmdb_path '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/lmdb/lmdb0' \
	--cv_lmdb_key '/work1/asrkws/liangdai/ivw_online_train/sample_data_from_hxxj/lmdb/lmdb0/keys_lens.txt' \
    --lmdb_reverbfile '/train8/asrkws/kaishen2/Data_kws/Reverb/reverb.npy' \
    --lmdb_normfile './res/feanorm/fea_fb40.norm' \
    --lmdb_cmdlist './res/cmdlist/cmdlist_hxxj_incSil.txt' \
    --num_cmds 17 \
    --train_sent_main 100 \
    --train_sent_pb 0 \
    --train_ratio_uni 2 \
    --train_ratio_unipb 1 \
    --train_ratio_none 1 \
    --train_mainkwsid '1:3:10' \
    --train_padhead_frame 256 \
    --train_padtail_frame 24 \
    --train_e2e_winsize 64 \
    --val_sent_num 3000 \
    --init_sent_num 250000 \
    --max_sent_frame 30000 \
    --fintune_model './res/model/rename_init_ShuffleNetV2.pth' \
    --network 'network.ivw_ShuffleNetV2_VersionStable.KWSnnE2EKld' \
    --out_dir './train/branch_ivw' \
    --log_path './train/branch_ivw/0_train.log' \
    --batch_size 2048 \
    --bunch_size 2048 \
    --init_optimizer 'SGD' \
    --init_lr 0.0005 \
    --init_lr_2nd 0.001 \
    --optimizer 'SGD' \
    --lr 0.002 \
    --lr_2nd 0.004 \
    --epochs 8 \
    --lr_half_epochs 2,3,4,5,6,7,8,9 \
    --discount 0.5 \
    --use_bmuf \
    --bmuf_sync 50 \
    --display 100 \
    --seed 27863875 \
    --is_pad \
    --is_ivw \
    --is_reverb \
    --is_addnoise \
    --is_addnoisepb \
    --quant \
    --clamp \
    --gpu_num 1 \

    ### [used cfg] ###
    # --quant \
    # --clamp \
    # --is_fp16 \
    # --gpu_num 1 \

