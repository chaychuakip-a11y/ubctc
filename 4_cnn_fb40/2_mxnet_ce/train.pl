#!/usr/bin/perl -w
use strict;
use warnings;
use List::Util 'shuffle';

@ARGV == 1 || die "Usage: train.pl ngpu[-1:means run in dlp, >=0: means run in local and the number of gpu to use is \$ngpu]\n";

my ($ngpu)          = @ARGV;
my $num_servers     = $ngpu;
my $num_workers     = $ngpu;
my $ngpu_per_worker = $ngpu;

if ($ngpu == -1) #### when running in DLP, get number of gpu from system environment;
{
    if (!exists($ENV{'NWORKER'}))
    {
        print("Error: impossible, please check you running task by submit dlp and job type is MpiJob.\n");
        exit(1);
    }
    $num_servers = $ENV{'NWORKER'}*$ENV{'NGPU_PER_WORKER'};
    $num_workers = $ENV{'NWORKER'}*$ENV{'NGPU_PER_WORKER'};
    $ngpu_per_worker = $ENV{'NGPU_PER_WORKER'};
}
else
{
    print("Error: running in local machine is not supported yet.\n");
    exit(1);
}
print("ngpu_per_worker: $ngpu_per_worker, num_workers: $num_workers\n");

my $lrate         = 0.02;
my $hotlrate      = 0.005;
my $hotpartnum    = 1;
my $wd            = 0.00001;
my $half          = 4;
my $totaliter     = 14;
my $outdir        = "./mlp";
my $model_prefix  = "$outdir/dcnn";
my $bunchsize     = 4096;
my $padnum        = 16;
my $pad_between   = 8;
my $window_width  = 31;
my $frame_predict = 8;
my $num_classes   = 9004;                                                                                                                  #### size of classification layer
my $bashrc        = "/home3/asrdictt/taoyu/bashrc_mxnet";
my $datadir       = "/train34/sli/permanent/taoyu/mlg/korean/am/4_cnn_fb40/1_down_pfile/lmdb_fb40";
my $npart         = 10;
my $labpfile      = "/yrfs4/asrdictt/taoyu/mlg/korean/am/4_cnn_fb40/1_down_pfile/lib_fb40/lab.pfile0";                                    #IN
my $pfilelen      = "/yrfs4/asrdictt/taoyu/mlg/korean/am/4_cnn_fb40/1_down_pfile/lib_fb40/pfile.len.txt";                                 #OUT
my $pfile_LF_inc  = 1;
my $pre_model     = "/work/asrdictt/hjwang11/english/gongban/en_cloud/20230223_fintune/2.train_hybirdcnn_mxnet/mlp/dcnn-0-0060.params";    #### init model
my $pre_symbol    = "/work/asrdictt/hjwang11/english/gongban/en_cloud/20230223_fintune/2.train_hybirdcnn_mxnet/mlp/dcnn-0-symbol.json";
my $pre_dec       = "./dec_cpu.json";
my $cmd_line      = "perl /work1/asrdictt/taoyu/sbin/GetAvgSent.pl $bunchsize $padnum $labpfile $pfilelen $pad_between $pfile_LF_inc";
my $avgsentnum    = `$cmd_line`;
chomp($avgsentnum);
print($cmd_line. "\n");
print "avgsentnum:$avgsentnum\n";
my @iter_part     = (                                                                                                                     #### use trainset part numbers in each iter
    "0,1,2,3,4,5,6,7,8,9",
    "0,1,2,3,4,5,6,7,8,9",
    "0,1,2,3,4,5,6,7,8,9",
    "0,1,2,3,4,5,6,7,8,9",
    "0,1,2,3,4,5,6,7,8,9",
    "0,1,2,3,4,5,6,7,8,9",
    "0,1,2,3,4,5,6,7,8,9",
    "0,1,2,3",
    "4,5,6,7",
    "8,9,0,1",
    "2,3,4,5",
    "6,7,8,9",
    "0,1,2,3",
    "4,5,6,7",
);
if ($totaliter != scalar(@iter_part))
{
    print("Error: mismatch \$totaliter with \@iter_part, change \$totaliter to \@iter_part\n");
    exit(1);
    $totaliter = scalar(@iter_part);
}
foreach my $i (0..$#iter_part)
{
    my @index   = split/\,/,$iter_part[$i];
    foreach my $j(0..$#index)
    {
        if ($index[$j] >= $npart || $index[$j] < 0)
        {
            print("Error: \@iter_part set error, part number exceed \$npart\n");
            exit(1);
        }
    }
}

foreach my $i(0..$npart-1)
{
    my $curdatadir = "$datadir/train_part$i";
    while (!-e "$curdatadir/done")
    {
        print("waiting 30 seconds for $curdatadir/done\n");
        sleep(30);
    }
}

system("mkdir -p $outdir");
if ($pre_model ne "")
{
    system("cp $pre_symbol $outdir/dcnn-0-symbol.json");
    system("cp $pre_dec $outdir/dec_cpu.json");
    system("cp $pre_model $outdir/dcnn-0-0000.params");
}

my $sync_freq   = 50;
my $alpha       = 1.0;
my $blr         = 1.0;
my $bm          = 0.94;
if ($num_servers == 4)
{
    $alpha = 0.75;
    $bm    = 0.75;
}
elsif ($num_servers == 8)
{
    $bm = 0.9;
}
elsif ($num_servers == 12)
{
    $bm = 0.92;
}
elsif ($num_servers == 16)
{
    $bm = 0.94;
}
elsif ($num_servers == 20)
{
    $bm = 0.954;
}
elsif ($num_servers == 24)
{
    $bm = 0.962;
}
elsif ($num_servers == 28)
{
    $bm = 0.967;
}
elsif ($num_servers == 32)
{
    $bm = 0.972;
}
elsif ($num_servers != 1)
{
    die "Error: gpunum $num_servers not support.";
}

my $optimizer = "NAG";
my $kv_store  = "dist_device_sync";
my $wait_time = 1.5 * 60;
my $network   = "att";
my $seed      = 27863875;

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

#my $cmd = "python dmlc-submit --cluster ssh ";#ssh local:mpi
my $cmd = "python dmlc-submit --cluster mpi ";
$cmd .= "--num-servers $num_servers ";
$cmd .= "--num-workers $num_workers ";
#$cmd .= "--host-file hosts ";
$cmd .= "python train_dcnn.py ";
$cmd .= "--kv-store $kv_store ";
$cmd .= "--model-prefix $model_prefix ";
$cmd .= "--optimizer $optimizer ";
$cmd .= "--sync-freq $sync_freq ";
$cmd .= "--wd $wd ";
$cmd .= "--alpha $alpha ";
$cmd .= "--blr $blr ";
$cmd .= "--bm $bm ";
$cmd .= "--frame-num $bunchsize ";
$cmd .= "--pad-num $padnum ";
$cmd .= "--network $network ";
$cmd .= "--frame-predict $frame_predict ";
$cmd .= "--use-bmuf True ";
$cmd .= "--multi-node True ";
$cmd .= "--ngpu-per-worker $ngpu_per_worker ";
$cmd .= "--window-width $window_width ";
$cmd .= "--pad-between $pad_between ";
$cmd .= "--num-classes $num_classes ";
$cmd .= "--lmdbdir $datadir ";

my $preCV     = 0;
my $curCV     = 0;
my $pre_epoch = 0;
my $begin_i   = 0;
my $begin_j   = 0;
if (-e "$outdir/Allcv.log")
{
    open(IN, "$outdir/Allcv.log") || die;
    while (<IN>)
    {
        if (/^iter(\d+)_part(\d+) epoch:(\d+) lrate:(\S+) cv:(\S+)/)
        {
            $begin_i   = $1;
            $begin_j   = $2;
            $pre_epoch = $3;
            $lrate     = $4 if ($begin_i > 0 || ($begin_i == 0 && $begin_j >= $hotpartnum));
            $curCV     = $5;
            $preCV     = $curCV if ($curCV > $preCV);
        }
        $lrate = $1 / 2 if (/badrun.*lrate:(\S+)/);
    }
    close IN;

    if ($preCV > 0)
    {
        $begin_j++;
        my @index   = split/\,/,$iter_part[$begin_i];
        if ($begin_j > $#index)
        {
            $begin_j = 0;
            $begin_i++;
        }
    }

}
print("begin training at iter $begin_i part $begin_j\n");
open(CV, ">>$outdir/Allcv.log") || die;

for (my $i = $begin_i; $i < $totaliter; $i++)
{
    my @index   = split/\,/,$iter_part[$i];
    if ($i >= $half && $begin_j == 0)
    {
        $lrate = $lrate / 2.0;
    }

    for (my $j = $begin_j; $j <= $#index; $j++)
    {
        my $part_id    = $index[$j];
        my $curdatadir = "$datadir/train_part$index[$j]";
        while (!-e "$curdatadir/done") { sleep(30); }
        my $totalsentnum_part = getSentNum($curdatadir);
        my $partnum           = int($totalsentnum_part / $avgsentnum);

        my $cur_epoch = $pre_epoch + 1;
        my $outlog    = "$outdir/log$cur_epoch.log";
        my $model     = sprintf("%s-0-%04d.params", $model_prefix, $cur_epoch);

        my $curlrate = $lrate;
        $curlrate = $hotlrate if ($i == 0 && $j < $hotpartnum);

        if (!-e $model)
        {
            print "start iter$i part$j epoch:$cur_epoch $curdatadir\n";

            my $cur_cmd = $cmd;
            $cur_cmd .= "--epoch-size $partnum ";
            $cur_cmd .= "--load-epoch $pre_epoch ";
            $cur_cmd .= "--lr $curlrate ";
            $cur_cmd .= "--seed $seed ";
            $cur_cmd .= "--num-epochs $cur_epoch ";
            $cur_cmd .= "--part-id $part_id ";

            open(OUT, ">$outdir/run.sh") || die;
            print OUT "source $bashrc\n";
            print OUT "nohup perl checkstuck.pl $outlog $model $wait_time $outdir & \n";
            print OUT "$cur_cmd\n";
            close OUT;
            system("sh $outdir/run.sh 1>$outlog 2>&1");

            sleep 3;
        }
        $seed += 375;

        #print("$model\n");
        die "Stop at iter$i part$j epoch:$cur_epoch" if (!-e $model);

        $curCV = GetCV($outlog);

        if ((($curCV + 0.02 < $preCV && $i > 0) || $curCV < 0.2))
        {
            print CV " badrun iter$i\_part$j epoch:$cur_epoch lrate:$curlrate cv:$curCV\n";
            
            unlink $model;
            system("mv $outlog $outlog.bad");
            if ($i == 0 && $j < $hotpartnum)
            {
                $hotlrate = $hotlrate / 2;
            }
            else
            {
                $lrate = $lrate / 2.0;
            }
            $j--;
        }
        else
        {
            print CV "iter$i\_part$j epoch:$cur_epoch lrate:$curlrate cv:$curCV\n";
            $preCV     = $curCV if ($curCV > $preCV);
            $pre_epoch = $cur_epoch;
        }
        die "stop"           if (-e "$outdir/stop");
        system("touch stop") if ($i == $totaliter - 1 && $j == $#index);
    }
    $begin_j = 0;
}
system("sh $outdir/run_kill.sh");

sub getRandindexPart
{
    my ($randfile, $start, $end) = @_;
    if (!-e "$randfile")
    {
        my $randdir = $randfile;
        $randdir =~ s/\/[^\/]+$//;
        system("mkdir -p $randdir");
        open(OUT, ">$randfile") || die "$randfile";
        my @ind   = ($start .. $end);
        my @index = shuffle(@ind);
        foreach (@index)
        {
            print OUT "$_\n";
            print "$_\n";
        }
        close OUT;
        return \@index;
    }
    else
    {
        open(IN, "$randfile") || die "$randfile";
        my @index = <IN>;
        close IN;
        chomp(@index);
        return \@index;
    }

}

sub getSentNum
{
    my ($curdatadir) = @_;
    my $totalsentnum_part = 0;
    open(IN, "$curdatadir/info.txt") || die "can't open $curdatadir/info.txt";
    while (<IN>) { $totalsentnum_part = $1 if (/totalsents:(\d+)/); }
    close IN;
    return $totalsentnum_part;

}

sub GetCV
{
    my ($log) = @_;
    my $curCV = 0;
    my $count = 0;
    open(IN, $log) || die;
    while (<IN>)
    {
        if (/Validation-accuracy=(\S+)/)
        {
            $count++;
            $curCV += $1;
        }

    }
    close IN;
    if ($count > 0)
    {
        $curCV /= $count;
        $curCV = int($curCV * 10000 + 0.5) / 10000;
    }
    return $curCV;
}
