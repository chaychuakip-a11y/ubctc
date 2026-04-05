dir_bin_mat=/work1/asrdictt/taoyu/bin/mat

model_input=$(pwd)/train_ctc_init_ce/model.iter13.part4
output_name=ko_2l1u2d_17kh_aug_h9i13p4
dir_out=$(dirname $model_input)/weights

if [ ! -e $dir_out ]; then mkdir $dir_out; fi

module unload gcc cuda
/work1/asrdictt/taoyu/python/caffe/pytorch2mat_ub.py $model_input $dir_out/${output_name}.mat

matlab -nosplash -nodisplay -r "path(path, '$dir_bin_mat'); convert_float_to_8bit_fixpoint('$dir_out/${output_name}.mat', '$dir_out/${output_name}_fix.mat', '$dir_out/param.mat', '$dir_out/mat_names.txt'); exit()"
