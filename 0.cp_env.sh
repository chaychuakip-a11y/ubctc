dir_src=/work1/asrdictt/taoyu/mlg/slovenian/am

# mkdir -p 1_mle_mfc
# cp -p $dir_src/1_mle_mfc/{*.pl,*.pm,*.sh} ./1_mle_mfc/

# mkdir -p 2_dnn_fb72/{1_down_pfile,2_pretrain,3_train}
# cp -p $dir_src/2_dnn_fb72/1_down_pfile/{*.pl,*.sh} ./2_dnn_fb72/1_down_pfile/
# cp -p $dir_src/2_dnn_fb72/2_pretrain/{*.pl,*.sh} ./2_dnn_fb72/2_pretrain/
# cp -p $dir_src/2_dnn_fb72/3_train/{*.pl,*.sh} ./2_dnn_fb72/3_train/

# mkdir -p 3_ub_fb40/{0_data_process,1_down_pfile,2_train}
# cp -p $dir_src/3_ub_fb40/0_data_process/{*.pl,*.sh} ./3_ub_fb40/0_data_process/
# cp -p $dir_src/3_ub_fb40/1_down_pfile/{*.pl,*.sh} ./3_ub_fb40/1_down_pfile/
# cp -p $dir_src/3_ub_fb40/2_train/{*.py,*.sh,*.ini} ./3_ub_fb40/2_train/
# cp -p -r $dir_src/3_ub_fb40/2_train/asr ./3_ub_fb40/2_train/

mkdir -p 4_cnn_fb40/{1_down_pfile,2_mxnet_ce}
cp -p $dir_src/4_cnn_fb40/1_down_pfile/{*.pl,*.sh} ./4_cnn_fb40/1_down_pfile/
cp -p $dir_src/4_cnn_fb40/2_mxnet_ce/{*.pl,*.py,*.sh,*.json,*.md,dmlc-submit,bashrc*} ./4_cnn_fb40/2_mxnet_ce/
cp -p -r $dir_src/4_cnn_fb40/2_mxnet_ce/{dmlc_tracker,python} ./4_cnn_fb40/2_mxnet_ce/
