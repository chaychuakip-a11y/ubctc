use strict;

my $config_data;
BEGIN {
    require "./utils.pl";
    $config_data = load_config();
    if (defined $config_data->{dir_sbin} && $config_data->{dir_sbin} ne '') {
        unshift @INC, $config_data->{dir_sbin};
    } else {
        die "Error: dir_sbin not set in config.json (needed to locate share_hadoop.pm)\n";
    }
    eval { require share_hadoop; share_hadoop->import(); };
    if ($@) {
        die "Error: cannot load share_hadoop from dir_sbin=$config_data->{dir_sbin}\n"
          . "  Check that share_hadoop.pm exists at that path.\n"
          . "  Current \@INC: " . join(", ", @INC) . "\n";
    }
}

my $jobname           = "dnnfa_maeopen";#SET
my $jobqueue          = $config_data->{jobqueue} || "nlp";#SET
my $num_reduce        = 10;
my $in_blocksize      = 512*1024*1024;
my $block_size        = 64*1024*1024;
my $replication       = 2;


my @hdir_src; if (@ARGV > 0) {
    @hdir_src = ($ARGV[0]);
} else {
    @hdir_src = (
                         "$config_data->{hdfs_out_root}/wav_noisy_maeopen_0.1.wav",
                        );
}
my $hdir_out          = ("$config_data->{hdfs_out_root}/wav_noisy_maeopen_0.1.wav_dnnfa");
my $hdir_src          = join(" -input ", @hdir_src);
my $dir_tmp           = "tmp"; mkdir $dir_tmp if !-e $dir_tmp;
foreach my $hdir_cur (@hdir_src)
{
	if($hdir_cur =~ /\*/ && $hdir_cur =~ /part/)
	{
		$hdir_cur =~ s#/[^/]+$##;
	}
	$hdir_cur .= '/_SUCCESS';
	system("$config_data->{dir_sbin}/wait_dir_hdfs.pl $hdir_cur");
}

##input resource  fa
my $config            = $config_data->{atom_config};
my $dir_ac            = $config_data->{dir_ac};
my $dir_lm            = $config_data->{dir_lm};
my $hmmlist           = "$dir_ac/hmmlist.final";
my $acmod_bin         = "$dir_ac/atom_acmod.dnn.bin";
my $fst_bin           = "$dir_lm/atom_fst.bin";
# my $wfst_bin          = "$dir_lm/atom_wfst.bin";
my $G_fst             = "$dir_lm/G.fst";
my $wts_file          = $config_data->{wts_file};
my $fea_norm          = $config_data->{fea_norm};
my $state_count       = $config_data->{state_count};

system("cp $config $dir_tmp/atom_fa.$jobname.cfg");
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","acmod_res",$acmod_bin);
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","hmm_res",$hmmlist);
# ChangePara("$dir_tmp/atom_fa.$jobname.cfg","wfst_res",$wfst_bin);
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","fst_res",$fst_bin);
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","weight_file",$wts_file);
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","norm_file",$fea_norm);
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","prior_file",$state_count);
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","layer_size","792,2048,2048,2048,2048,2048,2048,9004");
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","feat_dim",72);
ChangePara("$dir_tmp/atom_fa.$jobname.cfg","feat_descript","fb72");
$config = "$dir_tmp/atom_fa.$jobname.cfg";

##decode parameter
my $mlp_thread_nums   = 2;  #advise: set 1 if DNN MLP, 1-8 if RNN MLP
my $dec_thread_nums   = 1;  #advise: set 1-4 if running on hadoop, 1-12 if running at local
my $thread_nums       = $mlp_thread_nums + $dec_thread_nums;

##tools
my $dir_bin            = $config_data->{dir_bin};
my $bin_stream         = "$dir_bin/streamingAC-2.5.0.jar";
my $bin_selecttail     = "$dir_bin/selecttail";
my $bin_renametail     = "$dir_bin/renametail";
my $bin_randname       = "$dir_bin/randname";
my $bin_randnamered    = "$dir_bin/randnamered";
my $checkMLF           = "$dir_bin/checkMlfWithDict";
my $bin_atom           = "$dir_bin/atom-v20151016b/atom";
my $bin_libboost_so    = "$dir_bin/atom-v20151016b/libboost_thread.so.1.46.1";
my $bin_libboost_thread_so = "$dir_bin/atom-v20151016b/libboost_thread.so.1.58.0";
my $bin_libboost_system_so = "$dir_bin/atom-v20151016b/libboost_system.so.1.58.0";
my $bin_libmkl_lp64_so     = "$dir_bin/atom-v20151016b/libmkl_intel_lp64.so";
my $bin_libmkl_thread_so   = "$dir_bin/atom-v20151016b/libmkl_intel_thread.so";
my $bin_libmkl_core_so     = "$dir_bin/atom-v20151016b/libmkl_core.so";
my $bin_libiomp5_so        = "$dir_bin/atom-v20151016b/libiomp5.so";
my $bin_libimf_so          = "$dir_bin/atom-v20151016b/libimf.so";
my $bin_libsvml_so         = "$dir_bin/atom-v20151016b/libsvml.so";
my $bin_libirng_so         = "$dir_bin/atom-v20151016b/libirng.so";
my $bin_libintlc_so        = "$dir_bin/atom-v20151016b/libintlc.so.5";
my $bin_lattice_so     = "$dir_bin/atom-v20151016b/lattice.so";
my $bin_decode_so      = "$dir_bin/atom-v20151016b/decoder.so";
my $bin_ulm_so         = "$dir_bin/atom-v20151016b/rescore/ulmc.so";
my $bin_lm_so          = "$dir_bin/atom-v20151016b/rescore/LMTrie_F.so";
my $bin_res_so         = "$dir_bin/atom-v20151016b/rescore/ResMgr.so";
my $bin_cmvn           = "$dir_bin/htk-0.1.4/bin/cmvn_simple";
my $bin_raw_fea        = "$dir_bin/htk-0.1.4/bin/raw_fea";
my $config1            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerMFCC_0_D_A";
my $config2            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB24_0_D_A";
my $config3            = "$dir_bin/htk-0.1.4/cfg/config.fea.16K_offCMN_PowerFB40";

my @cmd_map;
my @cmd_red;
my @files;
my $cmd_map;
my $cmd_red;
my $files;
my $cmd;

@cmd_map = (
#::CmdToLocal("$checkMLF mlf_sy $dict"),
#::CmdToLocal("$bin_selecttail wav mlf_sy"),
::CmdToLocal("$bin_raw_fea $config2 fb72"),
::CmdToLocal("$bin_cmvn 2 24 1 fb72"),
# ::CmdToLocal("$bin_raw_fea $config3 fb40"),
::CmdToLocal("$bin_atom -c $config -mtn $mlp_thread_nums -dtn $dec_thread_nums"),
::CmdToLocal("$bin_selecttail wav mlf_sy mlf_fa_ph"),
::CmdToLocal("$bin_randname"),
);

@cmd_red = (
::CmdToLocal("$bin_randnamered"),
);

@files   = (
$config,
$acmod_bin,
$wts_file,
$fea_norm,
$state_count,
$bin_atom,
$bin_libboost_so,
$bin_libboost_thread_so,
$bin_libboost_system_so,
$bin_libmkl_lp64_so,
$bin_libmkl_thread_so,
$bin_libmkl_core_so,
$bin_libiomp5_so,
$bin_libimf_so,
$bin_libsvml_so,
$bin_libirng_so,
$bin_libintlc_so,
$bin_lattice_so,
$bin_decode_so,
$bin_ulm_so,
$bin_lm_so,
$bin_res_so,
$fst_bin,
# $wfst_bin,
$G_fst,
$hmmlist,
$bin_cmvn,
$bin_raw_fea,
$config1,
$config2,
$config3,
$bin_renametail,
$bin_selecttail,
$bin_randname,
$bin_randnamered,
);

$cmd_map = join(" | ", @cmd_map);
$cmd_red = join(" | ", @cmd_red);

if(@cmd_map > 1)
{
	open(OUT, ">", "$dir_tmp/mapper.$jobname.sh") || die $!;
	print OUT "#!/bin/bash\n";
	print OUT "$cmd_map\n";
	close OUT;
	push(@files, "$dir_tmp/mapper.$jobname.sh");
	$cmd_map = "bash ./mapper.$jobname.sh";
}

if(@cmd_red > 1)
{
	open(OUT, ">", "$dir_tmp/reducer.$jobname.sh") || die $!;
	print OUT "#!/bin/bash\n";
	print OUT "$cmd_red\n";
	close OUT;
	push(@files, "$dir_tmp/reducer.$jobname.sh");
	$cmd_red = "bash ./reducer.$jobname.sh";
}

$files = join(",", @files);

$cmd =  "hadoop jar $bin_stream "
."-Dmapreduce.map.java.opts=\"-Xmx96000m\" "
."-Dmapreduce.reduce.java.opts=\"-Xmx4096m\" "
."-Dmapreduce.map.cpu.vcores=$thread_nums "
."-Dmapreduce.map.failures.maxpercent=20 "
."-Dmapreduce.job.queuename=$jobqueue "
."-Dmapreduce.job.name=$jobname "
."-Dmapreduce.job.reduces=$num_reduce "
."-Dmapreduce.map.memory.mb=24000 "
."-Dmapreduce.reduce.memory.mb=3000 "
."-Ddc.input.block.size=$in_blocksize "
."-Ddfs.blocksize=$block_size "
."-Ddfs.replication=$replication "
."-files \"$files\" "
."-input $hdir_src "
."-output $hdir_out "
;

$cmd .= "-mapper \"$cmd_map\" " if(@cmd_map > 0);
$cmd .= "-reducer \"$cmd_red\" " if(@cmd_red > 0);

::RemoveHadoopDirIfExist($hdir_out);
::PR($cmd);
::SuccessOrDie("$hdir_out");

sub ChangePara
{
    my ($cfg,$paraname,$para)=@_;
    if ($para=~/\//)
    {
        if ($para!~/workdir/ && !-e $para)
        {
            die "Error: can't find $para\n";
        }
        $para=~s/.*\//\.\//;
    }
    open(IN,$cfg)||die"can't open $cfg";
    my @cfgcon=();
    while(<IN>)
    {
        push @cfgcon,"$_";
    }
    close IN;
    open(OUT,">","$cfg") || die"can't open $cfg\n";
    my $find=0;
    foreach (@cfgcon)
    {
        next if(/^\s*\#/);
        if(/^\s*$paraname\s*=/)
        {
            print OUT "$paraname = $para\n";
            $find=1;
        }
        else
        {
            print OUT "$_";
        }
    }
    die "can't find $paraname\n" if($find==0);
    close OUT;
}
