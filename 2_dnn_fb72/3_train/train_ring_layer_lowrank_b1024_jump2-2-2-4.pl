use strict;

#### tools
my $bin_asgd       = "source /home3/asrdictt/taoyu/cuda11.1.bashrc;/work1/asrdictt/taoyu/tools/dnn_ring_layer/ring_layer_lowrank/ring_layer-V";
my $bin_crossValid = "source /home3/asrdictt/taoyu/cuda11.1.bashrc;/work1/asrdictt/taoyu/tools/dnn_ring_layer/crossValid_lowrank/crossValid-V";
my $bin_qnnorm     = "/work1/asrdictt/taoyu/tools/QN/bin/qnnorm";
my $bin_pfile_info = "/work1/asrdictt/taoyu/tools/QN/bin/pfile_info";
my $bin_matlab2txt = "/work1/asrdictt/taoyu/tools/bin/wts_matlab2txt";
my $bin_MultiLast  = "/work1/asrdictt/taoyu/tools/bin/MultiLastWeights";

### fea and network
my $fea_dim     = 72;
my $fea_context = 11;
my $outsize     = 9004;
my @hid_layers  = (2048, 2048, 2048, 2048, 2048, 2048, 512);
my $numHid      = join("_", @hid_layers);
my $hidname     = join(",", @hid_layers);

my $insize      = $fea_dim * $fea_context;
my $numlayers   = @hid_layers+2;

### set ring layer device ID
my $device_id   = 0; ### set GPUID
my $gpu_count   = 8; ### 4 or 8
my $gpu_arrays  = join(",", (0..$gpu_count-1));

my $step        = 1;
my $recvThres   = 1;

my ($sendThres, $sendBase, $divide_arg);

if($gpu_count == 4)
{
	$sendThres   = 20;
	$sendBase    = 20;
	$divide_arg  = "1-".scalar(@hid_layers+1);
}
elsif($gpu_count == 8)
{
	$sendThres   = 100;
	$sendBase    = 100;
	$divide_arg  = join(",", map($_ .= "-$_", (1..@hid_layers+1)));
}
else
{
	die "Error: gpu_count may be wrong, please check\n";
}

#### set lrate
my $momentum    = 0.0;
my $learn_rate  = 1.024;#1.024; 4.096
my $weight_cost = 0.0;

my $init_random_seed = 27863875;

#resource dir
my $init_wts    = "/work1/asrdictt/taoyu/mlg/korean/am/2_dnn_fb72/2_pretrain/cw11_h2048_2048_2048_2048_2048_2048_lrate0.0025_M0.9_decay0.0008_bunch1024_jumpframe8/mlp.0.rank512.wts";
system("/work1/asrdictt/taoyu/sbin/wait_file.pl $init_wts; /work1/asrdictt/taoyu/sbin/wait_time.pl 1");
my $dir_lib0    = "/work1/asrdictt/taoyu/mlg/korean/am/2_dnn_fb72/1_down_pfile/lib_fb72";
my $dir_lib     = $dir_lib0; $dir_lib =~ s#/work1/#/yrfs4/#;
my $file_norm   = "$dir_lib0/fea.norm";
my $nSplit      = 10;
my @nCV         = map {$_ = 1000} (0..$nSplit-1);
my @pfile_fea   = $nSplit == 1 ? ("$dir_lib/fea.pfile0") : map {$_ = "$dir_lib/fea.pfile$_"} (0..$nSplit-1);
my @pfile_lab   = $nSplit == 1 ? ("$dir_lib/lab.pfile0") : map {$_ = "$dir_lib/lab.pfile$_"} (0..$nSplit-1);
my @file_norm   = $nSplit == 1 ? ("$dir_lib/fea.norm0" ) : map {$_ = "$dir_lib/fea.norm$_" } (0..$nSplit-1);
@nCV == $nSplit || die "Error: mismatch";
@pfile_fea == $nSplit && @pfile_lab == $nSplit && @file_norm == $nSplit || die "Error: mismatch";

my $nEpoch = 10;   ## how many epochs

###set bunchsize and learning rate
my $bunchsize       = 4096;
my $nBunchPerChunk  = 100;  ## how many bunchs per chunk
my $chuncksize      = $nBunchPerChunk * $bunchsize;  ## how many samples per chunk

my @bs_scale    = (2,1,1,1,1,1,1,1,1,1);        ## note: bunchsize scale, 2 means half the bunchsize
#my @lr_scale    = (5.3333,2,1,1,1,2,4,8,16,32); ## note: 2 means half the learning rate
my @lr_scale  = (2.6667,1,1,1,1,2,4,8,16,32); ## note: 2 means half the learning rate

###set jumpframe and discard
my $jumpframe = "3-2-5";
my @nmod;
my @remainder;
if ($jumpframe eq "3-2-5")
{
	@nmod      = (3,3,3,2,2,5,5,5,5,5);
	@remainder = (0,1,2,0,1,0,1,2,3,4);
}
elsif ($jumpframe eq "2-2-2-4")
{
	@nmod      = (2,2,2,2,2,2,4,4,4,4);
	@remainder = (0,1,0,1,0,1,0,1,2,3);
}
elsif($jumpframe eq "")
{
	@nmod      = (1,1,1,1,1,1,1,1,1,1);
	@remainder = (0,0,0,0,0,0,0,0,0,0);
}
else
{
	die "Error: not supported jumpframe config: $jumpframe\n";
}

my $discard_prob = 0;
my @discardLabs  = ($outsize-4,$outsize-3,$outsize-2,$outsize-1);
my $discardLabs  = join(",", @discardLabs);
my $train_ratio  = 1; ##note: only select part of training data

#set output dir
my $dir_mlp      = "./mlp-ring-h${numHid}-cw${fea_context}-targ${outsize}-step${step}_b${bunchsize}_jumpframe${jumpframe}_discard${discard_prob}";
my $dir_cfg      = "$dir_mlp/config";
my $output_wts   = "$dir_mlp/mlp";
my $output_log   = "$dir_mlp/mlp";
print $dir_mlp."\n";
mkdir $dir_mlp if !-e $dir_mlp;
mkdir $dir_cfg if !-e $dir_cfg;

my @nSent;
my @nFrame;
my @train_bp_range;
my @train_cv_range;
my $pfile_fea_cur;
my $pfile_lab_cur;
my $file_norm_cur;
my $pfile_fea_cv;
my $pfile_lab_cv;
my $epoch_cur;
my $i;
my $j;
my $cmd;

# judge init_wts or out_wts to decide do train or not
my $init_wts_bytes = -s $init_wts;
if(!defined($init_wts_bytes) || $init_wts_bytes == 0)
{
	die "ERROR: init_wts $init_wts does not exist or is empty\n";
}

#### set train_range and cv_range
foreach $j(0..$nSplit-1)
{
#	$pfile_fea_cur = $nSplit > 1 ? $pfile_fea.$j : $pfile_fea;
	$pfile_fea_cur = $pfile_fea[$j];
	my $tmp = `$bin_pfile_info -i $pfile_fea_cur`;
	if($tmp && $tmp =~ /(\d+)\s*sentences,\s*(\d+)\s*frames/)
	{
		$nSent[$j] = $1;
		$nFrame[$j] = $2;
		$nSent[$j] > $nCV[$j] || die "Error: It is impossible\n";
	}
	else
	{
		die "Error: fail to get sentence count, from pfile: $pfile_fea_cur\n";
	}
	my $tur = join("-", 0, int($nSent[$j]*$train_ratio)-$nCV[$j]-1);
	my $cur = join("-", $nSent[$j]-$nCV[$j], $nSent[$j]-1);

	print "$j: train: $tur\n";
	print "$j: cv: $cur\n";
}

if(!-e $file_norm)
{
	die "Error: no exist norm file: $file_norm\n";
}

my $pre_wts = $init_wts;

foreach my $epoch (0..$nEpoch-1)
{
	my $nmod           = $nmod[$epoch];
	my $remainder      = $remainder[$epoch];
	my $cur_bunchsize  = int($bunchsize/$bs_scale[$epoch]/$nmod);
	my $cur_learn_rate = $learn_rate/$lr_scale[$epoch];

	foreach my $j(0..$nSplit-1)
	{
		$epoch_cur     = $epoch*$nSplit+$j;
		$pfile_fea_cur = $pfile_fea[$j];
		$pfile_lab_cur = $pfile_lab[$j];

		my $train_bp_range = join("-", 0, int($nSent[$j]*$train_ratio)-$nCV[$j]-1);
		my $train_cv_range = join("-", $nSent[$j]-$nCV[$j], $nSent[$j]-1);
		
		$pfile_fea_cv  = $pfile_fea[0];
		$pfile_lab_cv  = $pfile_lab[0];
		my $train_cv_range_cv = join("-", $nSent[0]-$nCV[0], $nSent[0]-1);

		my $in_wts      = $pre_wts;
		my $out_wts     = "$output_wts.$epoch_cur.wts";
		$pre_wts        = $out_wts;

		my $file_config_tr = "$dir_cfg/dnn_config_tr.$epoch_cur";
		my $file_config_cv = "$dir_cfg/dnn_config_cv.$epoch_cur";
		my @config_tr;
		my @config_cv;

		### create training config file
		push @config_tr, "init_random_seed = $init_random_seed";
		push @config_tr, "fea_dim          = $fea_dim";
		push @config_tr, "fea_context      = $fea_context";
		push @config_tr, "dnn_layers       = {$insize,$hidname,$outsize}";
		push @config_tr, "bunchsize        = $cur_bunchsize";
		push @config_tr, "chuncksize       = $chuncksize";
		push @config_tr, "nmod             = $nmod";
		push @config_tr, "remainder        = $remainder";
		push @config_tr, "discard_prob     = $discard_prob";
		push @config_tr, "discardLabs      = {$discardLabs}";
		push @config_tr, "momentum         = $momentum";
		push @config_tr, "learn_rate       = $cur_learn_rate";
		push @config_tr, "weight_cost      = $weight_cost";
		push @config_tr, "file_norm        = $file_norm";
		push @config_tr, "pfile_fea        = $pfile_fea_cur";
		push @config_tr, "pfile_lab        = $pfile_lab_cur";
		push @config_tr, "init_wts         = $in_wts";
		push @config_tr, "train_bp_range   = $train_bp_range";
		push @config_tr, "train_cv_range   = $train_cv_range";
		push @config_tr, "output_wts       = $output_wts";
		push @config_tr, "output_log       = $output_log";
		push @config_tr, "gpu_arrays       = {$gpu_arrays}";
		push @config_tr, "step             = $step";
		push @config_tr, "sendThres        = $sendThres";
		push @config_tr, "recvThres        = $recvThres";
		push @config_tr, "sendBase         = $sendBase";
		push @config_tr, "divide_arg       = $divide_arg";

		open OUT, ">", $file_config_tr || die "Error: can not write file: $file_config_tr";
		print OUT join("\n",@config_tr);
		close OUT;

		### create crossValid config file
		push @config_cv, "init_random_seed = 27863875";
		push @config_cv, "device_id        = $device_id";
		push @config_cv, "fea_dim          = $fea_dim";
		push @config_cv, "fea_context      = $fea_context";
		push @config_cv, "dnn_layers       = {$insize,$hidname,$outsize}";
		push @config_cv, "bunchsize        = 1024";
		push @config_cv, "chuncksize       = $chuncksize";
		push @config_cv, "momentum         = $momentum";
		push @config_cv, "learn_rate       = $cur_learn_rate";
		push @config_cv, "weight_cost      = $weight_cost";
		push @config_cv, "file_norm        = $file_norm";
		push @config_cv, "pfile_fea        = $pfile_fea_cv";
		push @config_cv, "pfile_lab        = $pfile_lab_cv";
		push @config_cv, "init_wts         = $out_wts";
		push @config_cv, "train_bp_range   = $train_cv_range_cv";
		push @config_cv, "train_cv_range   = $train_cv_range_cv";
		push @config_cv, "output_wts       = ${output_wts}_cv";
		push @config_cv, "output_log       = ${output_log}_cv";

		open(OUT, ">", $file_config_cv) || die "Error: can not write file: $file_config_cv";
		print OUT join "\n",@config_cv;
		close OUT;

		#### judge init_wts or out_wts to decide do train or not
		if (!-e $in_wts || ((-s $in_wts) != $init_wts_bytes + 5 && (-s $in_wts) != $init_wts_bytes))
		{
			die "Error: size of input weights file $in_wts, not equal to init_wts_bytes $init_wts_bytes\n";
		}
		if (-e $out_wts && ((-s $out_wts) == $init_wts_bytes + 5 || (-s $out_wts) == $init_wts_bytes))
		{
			print "output weights file $out_wts exists, epoch $epoch_cur done\n";
			next;
		}

		#### training
		print "start  training epoch $epoch_cur\n";
		$cmd = "$bin_asgd $file_config_tr $epoch_cur";
		print $cmd."\n";
		system($cmd);
		print "finish training epoch $epoch_cur\n";

		#### crossvaild
		print "start  crossValid epoch $epoch_cur\n";
		$cmd = "$bin_crossValid $file_config_cv 0";
		print $cmd."\n";
		system($cmd);
		system("mv ${output_log}_cv.0.log ${output_log}.cv.$epoch_cur.log");
		system("rm ${output_wts}_cv.0.wts");
		print "finish crossValid epoch $epoch_cur\n";

		if($epoch == $nEpoch-1 && $j == $nSplit-1)
		{
			my $layer_name = join(",", ($insize, @hid_layers, $outsize));
			system("$bin_matlab2txt $out_wts $out_wts.txt $numlayers");
			my $numlayers_merged = $numlayers-1;
			system("$bin_MultiLast $out_wts.merge $out_wts $layer_name");
			system("$bin_matlab2txt $out_wts.merge $out_wts.merge.txt $numlayers_merged");
		}
	}
	$init_random_seed += 345;
}
