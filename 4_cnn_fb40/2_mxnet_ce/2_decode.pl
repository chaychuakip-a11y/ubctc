#!/usr/bin/perl
#use strict;
use warnings;
use File::Path;

use lib "/ps5/mkws/yfwu5/ps2/ASR_tool/bin";
use share_hadoop;

my $LM_Scale        = "10";
my $Word_Penalty    = "5";
my $mlp_prior_scale = "1";
my $Gen_Beam        = "200";
my $Hist_Threshold  = "7500";
my $Rescore_Scale   = $LM_Scale;
my $fbknum          = 40;
my $featype         = "fb$fbknum";    #nocmnfb cmnfb fea fb40
my $result_dir;
my $dir_atom        = "/ps5/mkws/yfwu5/ps2/ASR_tool/bin/atom_cnn/bin/atom_cnn_release/caffe-bmuf-rand-convpad-dilation-left";
my $mlpdir          = "/yrfs3/mkws/leiwang32/poland/5_train_hybridCNN/2_train_CE_tfmask/mlp/poland_2000h_add_2000hspeed_add_4000hnoise_dnn_fa_fb40_nocmn_rand_split_5_lr0.01";    # SET
my $epochs          = 20;                                                                                                                                                         # SET
my $fea_norm        = "/ps5/mkws/leiwang32/poland/6_train_VGG/1_lmdb/pfile_split/fb40.concat_mix.norm";                                                                           #SET
my $Wfst_Res        = "/ps5/mkws/leiwang32/poland/dict/LtsExample-20200803/应用实体发音预测/1027_新增/lm/lm_poland/lm.poland_app.corpus.3gram.pak/wfst.bin";                                ###20201027
my $main_result_dir = "$mlpdir/result_e$epochs\_lm$LM_Scale\_pri$mlp_prior_scale/hw_chuilei_app_city_20201019";                                                                   #

my @hdir_srcs = (

    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Voice_assistant_contro.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/app.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/page_contro.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Setting.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Telephone.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Translation.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Video.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Music.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Weather.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Navigation.fea_fb_40d_nocmn",
    # "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/Navigation_test.fea_fb_40d_nocmn",
    "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/app_20101017_test.fea_fb_40d_nocmn",
    "/workdir/mkws/gmkws/leiwang32/poland/fb40/fb40_test/weather_city_20101017_test.fea_fb_40d_nocmn",

);    # SET
my @testset = (

    # "Voice_assistant_contro",
    # "app",
    # "page_contro",
    # "Setting",
    # "Telephone",
    # "Translation",
    # "Video",
    # "Music",
    # "Weather",
    # "Navigation",
    # "Navigation_test",
    "app_20101017_test",
    "weather_city_20101017_test",

);    # SET
foreach my $testid (0 .. @hdir_srcs - 1)
{
    # foreach my $testid (0 .. 0) {

    my $job             = "dec_hybridcnn";                       #SET
    my $jobqueue        = "mkws";                                #SET
    my $num_reduce      = 1;
    my $in_blocksize    = 256 * 1024 * 1024;
    my $block_size      = 256 * 1024 * 1024;
    my $MLP_thread_nums = 2;
    my $DEC_thread_nums = 6;
    my $thread_nums     = $MLP_thread_nums + $DEC_thread_nums;

    my $hdir_src = $hdir_srcs[$testid];

    # my $hdir_src     = join(" -input ", @hdir_srcs);
    my $test_name = $testset[$testid];

    # my $config		  = "/lustre2/asr/dyliu2/chnDict/Test/res/atom_decode_pos.cfg";
    my $config = "./atom_decode_pos.cfg";
    $result_dir = "$main_result_dir/$test_name";    # SET

    mkpath($result_dir);

    ###output
    my $hdir_out = "/workdir/mkws/gmkws/leiwang32/poland/decode/decode_$job/poland190814_zx240h_train\_tfmask_simple_navigation_binglian/$test_name";    #SET
    ::MakeHadoopDirIfNotExist($hdir_out);
    my $acmod_res   = "/ps5/mkws/leiwang32/poland/2_mle_train/pack_atom_state/atom_acmod_poland_2000h_train_dnn.bin";                                    #SET
    my $mlp_res     = sprintf("%s-0-%04d.params", "$mlpdir/dcnn", $epochs);
    my $State_Prior = "/ps5/mkws/leiwang32/poland/2_mle_train/pack_atom_state/state.input.poland_2000h_train_16k_hmm_fa.txt";                            #SET

    # my $Wfst_Res    = "/bfs1/cv5/yingchen15/asr/poland/lm/output/lm_poland190814_zx240h_train/lm.poland190814_zx240h_train.3gram.wfst.bin";     #SET
    # my $Wfst_Res    = "/bfs1/cv5/yingchen15/asr/poland/lm/output/lm_poland190823_tx500h_zx240h_train/lm.poland190823_tx500h_zx240h_train.wordLE28.3gram.wfst.bin";     #SET
    # my $Wfst_Res    = "/bfs1/cv5/yingchen15/asr/poland/lm/output/lm_poland190823_tx500h_zx240h_train_splitNUM/lm.poland190823_tx500h_zx240h_train.3gram.wfst.bin";     #SET
    my $ULM_Name = "/ps2/asr/zpxie2/jobs/lm/LM/8_package_newData_LM/Try2_package/nextg_try2_package/nextg_en20170929_try2_NewLM/Friday_4gram_try2_newlm.bin";

    print "wait $mlp_res" if (!-e $mlp_res);
    while (!-e $mlp_res || -s $mlp_res < 100) { sleep(30); }

    #my ($feat_inc,$mlpoutdim,$padnum)=@{ChangeProto($mlpdir,$result_dir)};
    system("cp $config $result_dir/atom.cfg");
    ChangePara("$result_dir/atom.cfg", "acmod_res", $acmod_res);

    ChangePara("$result_dir/atom.cfg", "wfst_res", $Wfst_Res);

    # ChangePara("$result_dir/atom.cfg","ulm_name",$ULM_Name);
    ChangePara("$result_dir/atom.cfg", "prior_file",           $State_Prior);
    ChangePara("$result_dir/atom.cfg", "LM_scale",             $LM_Scale);
    ChangePara("$result_dir/atom.cfg", "word_penalty",         $Word_Penalty);
    ChangePara("$result_dir/atom.cfg", "rescore_word_penalty", $Word_Penalty);
    ChangePara("$result_dir/atom.cfg", "prior_scale",          $mlp_prior_scale);
    ChangePara("$result_dir/atom.cfg", "gen_beam",             $Gen_Beam);
    ChangePara("$result_dir/atom.cfg", "hist_threshold",       $Hist_Threshold);
    ChangePara("$result_dir/atom.cfg", "rescoring",            0);                  ## rescore 1
    ChangePara("$result_dir/atom.cfg", "rescore_scale",        $Rescore_Scale);
    ChangePara("$result_dir/atom.cfg", "feat_descript",        "pos");

    ChangePara("$result_dir/atom.cfg", "feat_dim", 9004);

    $config = "$result_dir/atom.cfg";

    ##tools

    my $dir_bin                = "/home/mkws/huangchen3/bin/hadoop/bin";
    my $bin_atom               = "$dir_atom/atom";
    my $bin_lattice_so         = "$dir_atom/lattice.so";
    my $bin_decode_so          = "$dir_atom/decoder.so";
    my $bin_libboost_so        = "$dir_atom/libboost_thread.so.1.46.1";
    my $bin_caffe1             = "$dir_atom/libcaffe-nv.so";
    my $bin_caffe2             = "$dir_atom/libcaffe-nv.so.0";
    my $bin_caffe3             = "$dir_atom/libcaffe-nv.so.0.13.0";
    my $bin_libgflags          = "$dir_atom/libgflags.so.2";
    my $bin_libgflags0         = "$dir_atom/libgflags.so.0";
    my $bin_libglog            = "$dir_atom/libglog.so.0";
    my $bin_libopencv          = "$dir_atom/libopencv_core.so.2.4";
    my $bin_libprotobuf        = "$dir_atom/libprotobuf.so.8";
    my $bin_libleveldb         = "$dir_atom/libleveldb.so.1";
    my $bin_liblmdb            = "$dir_atom/liblmdb.so";
    my $libhdf5_hl             = "$dir_atom/libhdf5_hl.so.6";
    my $libhdf5                = "$dir_atom/libhdf5.so.6";
    my $opencv_highgui         = "$dir_atom/libopencv_highgui.so.2.4";
    my $opencv_imgproc         = "$dir_atom/libopencv_imgproc.so.2.4";
    my $bin_LMTrie_so          = "/ps5/mkws/yfwu5/ps2/ASR_tool/bin/atom_20160513/atom-1.1.2/rescore/LMTrie_F.so";
    my $bin_ResMgr_so          = "/ps5/mkws/yfwu5/ps2/ASR_tool/bin/atom_20160513/atom-1.1.2/rescore/ResMgr.so";
    my $bin_ulmc_so            = "/ps5/mkws/yfwu5/ps2/ASR_tool/bin/atom_20160513/atom-1.1.2/rescore/ulmc.so";
    my $bin_randname           = "$dir_bin/randname";
    my $bin_randnamered        = "$dir_bin/randnamered";
    my $bin_stream             = "$dir_bin/streamingAC-2.5.0.jar";
    my $bin_selecttail         = "$dir_bin/selecttail";
    my $bin_libboost_thread_so = "/ps5/mkws/yfwu5/ps2/ASR_tool/bin/resample/bin/atom/libboost_thread.so.1.58.0";
    my $bin_libboost_system_so = "/ps5/mkws/yfwu5/ps2/ASR_tool/bin/resample/bin/atom/libboost_system.so.1.58.0";

    my $cmd_map;
    my $cmd_red;
    my $cmd;

    ::RemoveHadoopDirIfExist($hdir_out);

    my $mxnetbin = "/home/mkws/huangchen3/tools/dyliu2/code/mxengine_decode_cpu/3-asr-predict/asr-dnn-predict";
    system("perl /home/mkws/huangchen3/tools/dyliu2/code/mxengine_decode_cpu/3-asr-predict/convert_json.pl $mlpdir/dec_cpu.json $result_dir/dec_cpu.json");
    my $mxnetcfg = "$result_dir/dec_cpu.json";    #out

    my $forwardcmd = "$mxnetbin feanorm=$fea_norm model=$mlp_res config=$mxnetcfg predict=8 predict_mask=4 window_width=31 padnum=16 maxfealen=4096 feadim=$fbknum usegpu=0 feat_descript=$featype pos_descript=pos outkind=logsoftmax display=1";

    $cmd_map = join(" | ", (::CmdToLocal("$forwardcmd"), ::CmdToLocal("$bin_atom -c $config -mtn $MLP_thread_nums -dtn $DEC_thread_nums"), ::CmdToLocal("$bin_selecttail result"),));

    $cmd_red = ::CmdToLocal("$bin_randnamered");

    open(OUT, ">", "./mapper.$job.sh") || die $!;
    print OUT "$cmd_map\n";
    close OUT;

    my $files_nnso = GetDependSo($mxnetbin);

    $cmd_map = "bash ./mapper.$job.sh";
    my $files = join ",", (
        $bin_randname,
        $bin_randnamered,
        $bin_selecttail,
        $bin_atom,
        $bin_lattice_so,
        $bin_decode_so,
        $bin_libboost_so,
        $bin_LMTrie_so,
        $bin_ResMgr_so,
        $bin_ulmc_so,
        $bin_caffe1,
        $bin_caffe2,
        $bin_caffe3,
        $bin_libgflags,
        $bin_libgflags0,
        $bin_libglog,
        $bin_libopencv,
        $bin_libprotobuf,
        $bin_libleveldb,
        $bin_liblmdb,
        $libhdf5_hl,
        $libhdf5,
        $opencv_highgui,
        $opencv_imgproc,

        $bin_libboost_thread_so,
        $bin_libboost_system_so,

        $config,
        $acmod_res,
        $mlp_res,
        $fea_norm,
        $State_Prior,
        $Wfst_Res,

        # $ULM_Name,

        $mxnetbin,
        $mxnetcfg,
        $files_nnso,
        "./mapper.$job.sh"
    );

    $cmd = "hadoop jar $bin_stream "
        . "-Dmapreduce.map.java.opts=\"-Xmx36000m\" "
        . "-Dmapreduce.map.failures.maxpercent=20 "
        . "-Dmapreduce.job.queuename=$jobqueue "
        . "-Dmapreduce.job.name=$job "
        . "-Dmapreduce.job.reduces=$num_reduce "
        . "-Dmapreduce.map.memory.mb=40000 "
        . "-Dmapreduce.reduce.memory.mb=3000 "
        . "-Dmapreduce.map.cpu.vcores=$thread_nums "
        . "-Ddc.input.block.size=$in_blocksize "
        . "-Ddfs.block.size=$block_size "
        . "-Dmapreduce.task.timeout=2000000 "
        . "-Ddfs.replication=1 "
        . "-files $files "
        . "-input $hdir_src "
        . "-output $hdir_out "
        . "-mapper \"$cmd_map\" "
        . "-reducer \"$cmd_red\" ";

    ::PR($cmd);
    ::SuccessOrDie("$hdir_out");

    print "get result from hadoop\n";
    system("hdfs dfs -cat $hdir_out/*part* | $dir_bin/selecttail result | /ps5/mkws/yfwu5/ps2/ASR_tool/bin/unpacker 6 - $result_dir result");

    # system("hdfs dfs -cat $hdir_out/*part* | $dir_bin/selecttail result | /lustre2/asr/dyliu2/chnDict/Test/bin/unpacker 6 - $result_dir result");
    # system("perl ./decode_bin/stat_acc_Robust_Transe.pl $result_dir/result/result.txt $test_name");

    #	if($testid==0){
    # system("hdfs dfs -cat $hdir_out/*part* | $dir_bin/selecttail result | /lustre2/asr/dyliu2/chnDict/Test/bin/unpacker 6 - $result_dir result");
    # system("perl /lustre1/mkws/yfwu5/WZ_Tibetan/from_yzhuang4/18_hybridCNN_2/decode_bin/stat_acc_Robust_Transe.pl $result_dir/result/result.txt");

    #	}elsif($testid==1){
    # system("hdfs dfs -cat $hdir_out/*part* | $dir_bin/rec2mlf result mlf_rec | $dir_bin/selecttail mlf_sy mlf_rec | $dir_bin/fea_lab_lat_unpack_1 - $result_dir mlf_rec mlf_sy");
    # # system("perl /lustre2/asr/dyliu2/chnDict/Test/bin/Eng/stat_acc_eng.pl $result_dir/out.mlf_rec");
    # system("perl /lustre1/mkws/huangchen3/asr/japanese/hybridCNN/decode/stat_acc_eng.pl $result_dir/out.mlf_rec");
    #	}
}

system("rm -rf sbatch*");
system("rm -rf *~");

sub ChangePara
{
    my ($cfg, $paraname, $para) = @_;
    open(IN, $cfg) || die "can't open $cfg";
    if ($para =~ /\//)
    {
        if ($para !~ /workdir/ && !-e $para)
        {
            die "Error: can't find $para\n";
        }
        $para =~ s/.*\//\.\//;
    }

    my @cfgcon = ();
    while (<IN>) { push @cfgcon, "$_"; }
    close IN;
    open(OUT, ">:unix", "$cfg") || die "can't open $cfg";
    my $find = 0;
    foreach (@cfgcon)
    {
        next if (/^\s*\#/);
        if (/^\s*$paraname\s*=/)
        {
            print OUT "$paraname = $para\n";
            $find = 1;
        }
        else { print OUT "$_"; }
    }

    #print OUT "$paraname = $para\n" if($find==0);
    die "can't find $paraname\n" if ($find == 0);
    close OUT;

}

sub ChangeProto
{
    my ($mlpdir, $result_dir) = @_;
    my @paras = ();
    system("cp /lustre2/asr/dyliu2/DCNN/CE/fully/mlp/test_dim$fbknum.head $result_dir/test.prototxt");
    open(OUT, ">>$result_dir/test.prototxt") || die;
    open(IN,  "$mlpdir/caffe.prototxt")      || die "Error: can't open $mlpdir/caffe.prototxt";
    my $layercount   = 0;
    my $printstart   = 0;
    my $feat_inc     = 1;
    my $feat_inc_out = 1;
    my $mlpoutdim;
    my $padnum     = 0;
    my $printlabel = 0;
    my $convout    = 0;

    while (<IN>)
    {
        if (/^\s*layer\s*\{\s*$/)
        {
            $layercount++;
            $printstart = 1 if ($layercount > 2);
        }
        if ($printstart)
        {
            s/CUDNN/CAFFE/;
            print OUT "$_";
        }
        if (/convout/)
        {
            $convout = 1;
        }
        $mlpoutdim = $1 if (/num_output\s*:\s*(\d+)/);
        if (/numframepredict\s*:\s*(\d+)/)
        {
            if ($convout)
            {
                $feat_inc_out = $1;
            }
            else
            {
                $feat_inc = $1;
            }
        }

        $padnum = $1 if (/padnum\s*:\s*(\d+)/);
    }
    close IN;
    close OUT;
    $mlpoutdim *= $feat_inc if ($feat_inc_out == 1);    #deconv
    $paras[0] = $feat_inc;
    $paras[1] = $mlpoutdim;
    $paras[2] = $padnum;
    return \@paras;
}

sub GetDependSo
{
    my ($tool) = @_;

    #my $sos= `ldd $tool `;
    # my $sos= `cat /home/mkws/huangchen3/tools/dyliu2/code/mxengine_decode_cpu/3-asr-predict/dependso.txt`;
    my $sos    = `cat /yrfs3/mkws/leiwang32/poland/5_train_hybridCNN/2_train_CE_tfmask/dependso.txt`;
    my @sos    = split /\n/, $sos;
    my $result = "";
    foreach (@sos)
    {
        my $so = "";
        if (/ (\S+) \(/)
        {
            $so = $1;
        }
        if (/(\S+) => not found/)
        {
            my $soname = $1;
            my $path   = $tool;
            $path =~ s/\/[^\/]+$//;
            $so = "$path/$soname";
        }
        if ($so ne "" && $so !~ /^\/lib64/)
        {
            die "get $so faild from ldd $tool\n" if (!-e $so);
            $result .= "$so,";
        }
    }
    $result =~ s/,$//;
    return $result;
}

