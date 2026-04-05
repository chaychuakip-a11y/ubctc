#!/usr/bin/perl -w
use strict;
use warnings;

@ARGV == 1 || die "Usage: train.pl gpus[-1:means run in dlp, >=0: means run in local and the gpu id to use is \$ngpu]\n";

my ($gpus) = @ARGV;    #### gpu id to use

my $lrate         = 0.002;
my $wd            = 2.5e-5;
my @scale         = (1, 1, 1);                                                                        # lrate decay
my $outdir        = "./mlp";
my $model_prefix  = "$outdir/dcnn_init";
my $num_servers   = 1;
my $num_workers   = 1;
my $ngpu_per_worker = 1;
my $num_epochs    = scalar(@scale);
my $epoch_size    = 20000;
my $bunchsize     = 2048;
my $pad_num       = 16;
my $pad_between   = 8;
my $window_width  = 31;
my $frame_predict = 8;
my $num_classes   = 9004;                                                                       #### size of classification layer
my $bashrc        = "/home3/asrdictt/taoyu/bashrc_mxnet";
my $datadir       = "/yrfs4/asrdictt/taoyu/mlg/turkish/am/4_cnn_fb40/1_down_pfile/lmdb_fb40";
my $npart         = 4;
my $seed          = 27863875;
my $sync_freq     = 10;
my $alpha         = 1.0;
my $blr           = 1.0;
my $bm            = 0.94;
my $display_freq  = 10;
my $optimizer     = "NAG";
my $kv_store      = "dist_device_sync";
my $network       = "att";
my $wait_time     = 10 * 60;

mkdir $outdir;

open(OUT, ">$outdir/run_kill_cmd.sh") || die;
print OUT "ps ux | grep -v grep | grep 'train_dcnn'\n";
print OUT "ps ux | grep -v grep | grep 'train_dcnn' | awk '{print \$2}' | xargs kill -9\n";
close OUT;
open(OUT, ">$outdir/run_kill.sh") || die;
print OUT "source $bashrc\n";
print OUT "#### Kill program on remote machines\n";
print OUT "python dmlc-submit --cluster mpi --num-servers $num_servers --num-workers $num_workers sh $outdir/run_kill_cmd.sh\n";
print OUT "#### Kill program on local machine\n";
print OUT "sh $outdir/run_kill_cmd.sh\n";
close OUT;

my $cmd = "";
if ($gpus == -1)
{
    $cmd .= "python dmlc-submit --cluster mpi ";
    $cmd .= "--num-servers $num_servers ";
    $cmd .= "--num-workers $num_workers ";
}
$cmd .= "python train_dcnn.py ";
$cmd .= "--model-prefix $model_prefix ";
$cmd .= "--epoch-size $epoch_size ";
$cmd .= "--optimizer $optimizer ";
$cmd .= "--sync-freq $sync_freq ";
$cmd .= "--wd $wd ";
if ($gpus == -1)
{
    $cmd .= "--multi-node True ";
    $cmd .= "--ngpu-per-worker $ngpu_per_worker ";
}
else
{
    $cmd .= "--gpus $gpus ";
}
$cmd .= "--display-freq $display_freq ";
$cmd .= "--frame-num $bunchsize ";
$cmd .= "--pad-num $pad_num ";
$cmd .= "--network $network ";
$cmd .= "--frame-predict $frame_predict ";
$cmd .= "--window-width $window_width ";
$cmd .= "--pad-between $pad_between ";
$cmd .= "--num-classes $num_classes ";
$cmd .= "--lmdbdir $datadir ";

my $finalmodel = "";
my $initmodel  = sprintf("%s-0-%04d.params", "$outdir/dcnn", 0);
exit if (-e $initmodel);

for (my $iter = 0; $iter < $num_epochs; $iter++)
{
    my $part_id = $iter % $npart;
    my $lr_id   = $iter;

    my $cur_lrate = $lrate / $scale[$lr_id];
    my $cur_cmd   = $cmd;
    print "epoch $iter, learning rate is $cur_lrate\n";

    my $epochs     = $iter + 1;
    my $load_epoch = $iter + 0;
    my $outlog     = "$outdir/init_$iter.log";
    my $model      = sprintf("%s-0-%04d.params", $model_prefix, $epochs);

    if ($iter > 0) { $cur_cmd .= "--load-epoch $load_epoch "; }
    $cur_cmd .= "--lr $cur_lrate ";
    $cur_cmd .= "--seed $seed ";
    $cur_cmd .= "--num-epochs $epochs ";
    $cur_cmd .= "--part-id $part_id ";

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

        open(OUT, ">$outdir/run_init.sh") || die;
        print OUT "source $bashrc\n";
        print OUT "nohup perl checkstuck.pl $outlog $model $wait_time $outdir & \n" if ($gpus == -1);
        print OUT "$cur_cmd\n";
        close OUT;
        system("sh $outdir/run_init.sh 1>$outlog 2>&1");
    }
    $finalmodel = $model;
    $seed += 375;
    print "Finished!\n";
}

system("mv $finalmodel $initmodel") if (-e $finalmodel);
