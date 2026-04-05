#!/usr/bin/perl -w
use strict;
use warnings;

my $lrate         = 0.01;
my $wd            = 0.00001;
my @scale         = (1);                  # lrate decay
my $out_dir       = "./mlp_cpu";
my $model_prefix  = "$out_dir/dcnn";
my $num_servers   = 1;
my $num_workers   = 1;
my $num_epochs    = 1;
my $epoch_size    = 1;
my $frame_num     = 2048;
my $pad_num       = 16;
my $window_width  = 31;
my $pad_between   = 48;
my $frame_predict = 8;
my $num_classes   = 9004;                                                                       #### size of classification layer
my $bashrc        = "/home3/asrdictt/taoyu/bashrc_mxnet";
my $datadir       = "/yrfs4/asrdictt/taoyu/mlg/turkish/am/4_cnn_fb40/1_down_pfile/lmdb_fb40";
my $seed          = 27863875;
my $sync_freq     = 10;
my $alpha         = 1.0;
my $blr           = 1.0;
my $bm            = 0.94;
my $display_freq  = 10;
my $gpus          = "0";
my $optimizer     = "NAG";
my $kv_store      = "dist_device_sync";
my $network       = "cpu";

my $cmd .= "python train_dcnn.py ";
$cmd    .= "--model-prefix $model_prefix ";
$cmd    .= "--epoch-size $epoch_size ";
$cmd    .= "--optimizer $optimizer ";
$cmd    .= "--sync-freq $sync_freq ";
$cmd    .= "--wd $wd ";
$cmd    .= "--gpus $gpus ";
$cmd    .= "--display-freq $display_freq ";
$cmd    .= "--frame-num $frame_num ";
$cmd    .= "--pad-num $pad_num ";
$cmd    .= "--network $network ";
$cmd    .= "--frame-predict $frame_predict ";
$cmd    .= "--window-width $window_width ";
$cmd    .= "--pad-between $pad_between ";
$cmd    .= "--num-classes $num_classes ";
$cmd    .= "--lmdbdir $datadir ";

my $finalmodel = "";
my $initmodel  = sprintf("%s-0-%04d.params", $model_prefix, 0);
print "$initmodel\n";
exit if (-e $initmodel);

mkdir $out_dir;
for (my $iter = 0; $iter < $num_epochs; $iter++)
{
    my $part_id = 0;
    my $lr_id   = $iter;

    my $cur_lrate = $lrate / $scale[$lr_id];
    my $cur_cmd   = $cmd;
    print "epoch $iter, learning rate is $cur_lrate\n";

    my $epochs     = $iter + 1;
    my $load_epoch = $iter + 0;
    my $outlog     = "$out_dir/init_$iter.log";
    my $model      = sprintf("%s-0-%04d.params", $model_prefix, $epochs);
    print "$model\n";
    my $cnt = 0;
    while (!-e $model)
    {
        $cnt++;
        if ($cnt >= 3)
        {
            print "Failed to train\n";
            exit(1);
        }
        print $model. " not exists, start epoch " . $load_epoch . "\n";
        if ($iter > 0) { $cur_cmd .= "--load-epoch $load_epoch "; }
        $cur_cmd .= "--lr $cur_lrate ";
        $cur_cmd .= "--seed $seed ";
        $cur_cmd .= "--num-epochs $epochs ";
        $cur_cmd .= "--part-id $part_id ";
        system("echo source $bashrc >$out_dir/run.sh");
        system("echo $cur_cmd >>$out_dir/run.sh");
        system("chmod 777 $out_dir/run.sh");
        system("sh $out_dir/run.sh 1>$outlog 2>&1");
    }
    $finalmodel = $model;
    $seed += 375;
    print "Finished!\n";
}

system("mv $out_dir/dcnn-0-symbol.json $out_dir/dec_cpu.json");
