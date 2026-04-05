#!/usr/bin/perl -w

# --------------------------------------------------------- #
#                                                           #
#       The iFlytek EasyTraining Toolkit For Hadoop         #
#                                                           #
# --------------------------------------------------------- #
#          Copyright(c) iFlytek Corporation, 2012           #
#                     HeFei, Anhui, PRC                     #
#                 http://www.iflytek.com                    #
# --------------------------------------------------------- #
#   File: share_hadoop.pm: Global function defination       #
#                      Author: feihe                        #
#                      modify: taozhou2                     #
# --------------------------------------------------------- #
#   pay attention:                                          #
#      arg: $FUNCTION_ENABLE is the HPR control seitch      #
#           if you want to enable HPR, set it 1;            #
#           if you want to disable HPR, set it 0;           #
# --------------------------------------------------------- #

use strict;
use lib ".";
use lib "/work1/asrdictt/taoyu/sbin";
use share_hadoop;
use File::Copy;
use File::Basename;
use File::Path;

### Global variables
@ARGV == 1 || die "Usage: pl GV-MLE.pm\n";
-e $ARGV[0] || die "Error:  Missing or cannot find config file $ARGV[0]";
require("$ARGV[0]");

our $hdfs_bin_path;
our $g_hadoopdir_bin;

our $g_hdfs_port;

## Local
our $g_dir_done;
our $g_dir_final;
our $g_dir_lib;
our $l_dict_Mono;
our $l_dict_Align;
our $l_dict_Test;
our $l_cmd_EasyTraining_ModelGen;
our $g_auto_qst;
our $l_qst_QuestionSet;

## Hadoop
our $g_hadoopdir_Home;
our $g_hadoopdir_src_Mlf;

our $hadoop_work_dir;
our $g_hadoopdir_lib;
our $g_hadoopdir_acc;
our $g_hadoopdir_hmm;

## dict in hadoop
our $g_dict_Mono;
our $g_dict_Align;
our $g_cmd_EasyTraining_ModelGen;
our $g_qst_QuestionSet;
our $g_dict_Test;

our $g_hadoopdir_Mono;
our $g_hadoopdir_Context;
our $g_hadoopdir_Tcontext;

our $g_hadoopdir_phone_Mlf;
our $g_hadoopdir_fea_fixspsil;

our $g_hadoopdir_mono;
our $g_hadoopdir_context;
our $g_hadoopdir_tcontext;
our $g_hadoopdir_final;

our $g_logfile;
our $g_hmmlist_mono;
our $g_hmmlist_context;
our $g_hmmlist_tcontext;
our $g_hmmlist_au;
our $g_hmmlist_final;



## MapRed
our $g_feaVecSize;
our $g_nIter_mono;
our $g_bCheckScript;
our $g_strNonPhoneList;
our $g_nIter_afterFA;
our $g_nIter_xwrd;

our $g_nomerge_clustering;
our $g_nTieLevel;
our $g_strOutlier;
our $g_strInitTB;
our $g_strTiedNum;
our @g_strContextIndependentPhone;
our $g_bhmmlist_au_all;
our @g_strContextFreePhone;
our $g_strBinaryFlag;
our @g_strNonPhoneMU;
our $g_nIter_cxwrd;

our @g_pth_Phone;
our @g_pth_NonPhone;

our $g_strPruning;
our $g_strBeam;

our $g_mapred_task_name;
our $g_jobqueue;
our $g_reduces_num;
our $g_input_block_size;
our $g_hdfs_block_size;

## Bin
our $l_bin_hhed;
our $l_txt_null;
our $l_yarn_jar;
our $l_streaming;
our $g_streaming;
our $g_yarn_jar;
our $yarn_time_out;
our $yarn_memory;
our $mapred_memory;
our $g_bin_EasyTraining;
our $g_bin_pakeditmap;
our $g_bin_pakeditred;
our $g_bin_hcompvmap;
our $g_bin_hcompvred;
our $g_bin_genglobal;
our $g_bin_hermap;
our $g_bin_herred;
our $g_bin_HHEd_ifly;
our $g_bin_genhmmlistau;
our $g_bin_autoque;

our $FUNCTION_ENABLE;

my $HADOOP_JOB_ENABLE = 2;
my $LOCAL_JOB_ENABLE  = 1;
my $NULL_JOB          = 0;

### Fix Hadoop2
### Local Parameters
my $hadoop_cmd;
my $current_jobname;
my $result_tmp        = 0;
my $cmdline           = "";

our $g_bin_localrun;
our $g_bin_hadoopcat;
our $l_strNonPhoneList;
our $l_strContextIndependentPhone;
our $l_strContextFreePhone;
our $l_strNonPhoneMU;
our $l_strPruning;

-e $l_yarn_jar || die "missing yarn.jar\n";
-e $l_streaming || die "missing streaming.jar\n";

#################################################################################
#----------------------------- Mono-phone training -----------------------------#
#################################################################################

system("mkdir -p $g_dir_done");

if(!-e "$g_dir_done/DONE.0.CopyBasicRes")
{
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_lib");
    !system("hdfs dfs -put -f $g_dir_lib/$l_dict_Mono $g_dict_Mono") || die "fail copy to dfs: $g_dir_lib/$l_dict_Mono";
    !system("hdfs dfs -put -f $g_dir_lib/$l_dict_Align $g_dict_Align") || die "fail copy to dfs: $g_dir_lib/$l_dict_Align";
    !system("hdfs dfs -put -f $g_dir_lib/$l_cmd_EasyTraining_ModelGen $g_cmd_EasyTraining_ModelGen") || die "fail copy to dfs: $g_dir_lib/$l_cmd_EasyTraining_ModelGen";
    if(!$g_auto_qst)
    {
        !system("hdfs dfs -put -f $g_dir_lib/$l_qst_QuestionSet $g_qst_QuestionSet") || die "fail copy to dfs: $g_dir_lib/$l_qst_QuestionSet";
    }
    system("touch $g_dir_done/DONE.0.CopyBasicRes");
}

### 0 Make sp MLF ( word to phone )
if ( !-e "$g_dir_done/DONE.0.MakeSpMLF" )
{
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_acc");
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_hmm");
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
### edit mlf with dict from word to phone
    my $hs = new HS;
    $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_0_makesp_mono_mlf", $g_hadoopdir_phone_Mlf, $g_logfile);

    $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
        ."-files \"${g_hdfs_port}$g_bin_pakeditmap,${g_hdfs_port}$g_dict_Mono\" "
        ."-Ddc.input.block.size=$g_input_block_size "
        ."-Ddfs.blocksize=$g_hdfs_block_size "
        ."-numReduceTasks 0 "
        ."-input $g_hadoopdir_src_Mlf -output $g_hadoopdir_phone_Mlf "
        ."-mapper \"./".basename($g_bin_pakeditmap)." -GenMonoMlf ".basename($g_dict_Mono)."\" "
        ;

    $hs->run($hadoop_cmd,$FUNCTION_ENABLE);
    system("touch $g_dir_done/DONE.0.MakeSpMLF");
}

### 0 Make sp MLF and Gen hmmlist ( print monophone list )
if ( !-e "$g_dir_done/DONE.0.GenMonoList1" )
{
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_mono");
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_Mono");
### gen hmmlist.mono
    my $hs = new HS;
    $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_0_gen_mono_phone_list", "$g_hadoopdir_Mono/phone_list", $g_logfile);

    $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
        ."-files ${g_hdfs_port}$g_bin_pakeditmap "
        ."-Ddc.input.block.size=$g_input_block_size "
        ."-Ddfs.blocksize=$g_hdfs_block_size "
        ."-numReduceTasks $g_reduces_num "
        ."-input $g_hadoopdir_phone_Mlf -output $g_hadoopdir_Mono/phone_list "
        ."-mapper \"./".basename($g_bin_pakeditmap)." -MonoPhoneList\" ";

    $hs->run($hadoop_cmd,$FUNCTION_ENABLE);
    system("touch $g_dir_done/DONE.0.GenMonoList1");
}

### 0 Make sp MLF and Gen hmmlist ( make hmmlist.mono )
if( !-e "$g_dir_done/DONE.0.GenMonoList2" )
{
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
    $cmdline     = "perl,./hadoopcat.pl,"
        ."$g_hadoopdir_Mono/phone_list/*part*,"
        .basename($g_hmmlist_mono).","
        ."./".basename($g_bin_pakeditred).","
        ."argvpro_PrintPhoneList";

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_hadoopcat,$g_bin_pakeditred\" "
            ."-output \"".basename($g_hmmlist_mono)."\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"GenMonoListLocal\" "
            );

    if( $result_tmp != 0 ){
        die "ERROR : yarn application to make hmmlist.mono \n";
    }
    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/".basename($g_hmmlist_mono)." $g_hmmlist_mono");
    if( $result_tmp != 0 ){
        die "ERROR : make hmmlist.mono failed \n";
    }

    system("touch $g_dir_done/DONE.0.GenMonoList2");
}

### 1 Calculate global mean and variance by MR
if ( !-e "$g_dir_done/DONE.1.HCompV" )
{
    my $hs = new HS;
    $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_1_HCompV", "$g_hadoopdir_Mono/hv_result", $g_logfile);

    $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
        ."-files ${g_hdfs_port}$g_bin_hcompvmap,${g_hdfs_port}$g_bin_hcompvred "
        ."-Ddc.input.block.size=$g_input_block_size "
        ."-Ddfs.blocksize=$g_hdfs_block_size "
        ."-numReduceTasks 1 "
        ."-input $g_hadoopdir_src_Mlf -output $g_hadoopdir_Mono/hv_result "
        ."-mapper \"\./".basename($g_bin_hcompvmap)." 0.05 $g_feaVecSize\" -reducer \"./".basename($g_bin_hcompvred)."\" ";

### ! please pay your attention : this step must close the HPR function.
    $hs->run($hadoop_cmd,0);
    system("touch $g_dir_done/DONE.1.HCompV");
}

### 1 Calculate global mean and variance (get global and vFloors file)
if( !-e "$g_dir_done/DONE.1.HCompVLocal" )
{
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_Mono/hv_result/*part-* $g_hadoopdir_mono/hv_red.pak");
    if( $result_tmp != 0 ){
        die "ERROR : connot copy hv_result to HDFS , maybe $g_hadoopdir_mono/hv_red.pak already exist \n";
    }

    $cmdline     = "perl,./localrun.pl,"
        ."./".basename($g_bin_genglobal).","
        ."argvpro_I,hv_red.pak,"
        ."argvpro_O,./,"
        ."argvpro_n,5,"
        ."argvpro_f,0.01";

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"global,vFloors\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_genglobal,$g_hadoopdir_mono/hv_red.pak\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"HCompVLocal\" "
            );

    if( $result_tmp != 0 ){
        die "ERROR : yarn application to calculate global mean and variance \n";
    }
    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/global $g_hadoopdir_mono");
    if( $result_tmp != 0 ){
        die "ERROR : make global failed or global file already exist \n";
    }
    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/vFloors $g_hadoopdir_mono");
    if( $result_tmp != 0 ){
        die "ERROR : make vFloors failed or vFloors file already exist \n";
    }

    system("touch $g_dir_done/DONE.1.HCompVLocal");
}

### 2 Clone all mono-phones ( gen monophone HMM by global and vFloors )
if ( !-e "$g_dir_done/DONE.2.CloneAllMonoPhone" )
{
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
    $cmdline    = "perl,./localrun.pl,./".basename($g_bin_EasyTraining).","
        ."argvpro_ModelGen,"
        ."argvpro_H,./global,"
        ."argvpro_v,./vFloors,"
        ."argvpro_M,./MODELS.0,"
        .basename($g_cmd_EasyTraining_ModelGen);

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"MODELS.0\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_EasyTraining,$g_hadoopdir_mono/global,$g_hadoopdir_mono/vFloors,$g_cmd_EasyTraining_ModelGen\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"GenMonoListLocal\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS.0 $g_hadoopdir_mono/MODELS.0");
    if( $result_tmp != 0 ){
        die "ERROR : yarn application to gen monophone HMM \n";
    }
    system("touch $g_dir_done/DONE.2.CloneAllMonoPhone");
}

### Make FixSilSp directory and hmm0 directory for mono-phone model restimation
if ( !-e "$g_dir_done/DONE.3.mono1.FixSilSp" )
{
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_mono/FixSilSp");
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_mono/FixSilSp/hmm0");

    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_mono/MODELS.0 $g_hadoopdir_mono/FixSilSp/hmm0/MODELS");
    if( $result_tmp != 0 ){
        die "ERROR : MODELS already exist \n";
    }
}

### 3 Iteration 4 for mono-phone models
for (my $nIter = 0 ; $nIter < $g_nIter_mono ; $nIter++ )
{
    my $preiter = $nIter;
    my $outiter = $nIter + 1;
    next if ( -e "$g_dir_done/DONE.3.mono$outiter.FixSilSp" );

    my $outdir = "$g_hadoopdir_Mono/FixSilSp/$outiter";
    my $input_model = "$g_hadoopdir_mono/FixSilSp/hmm$preiter/MODELS";

###  Calculate occ
    my $hs = new HS;
    $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_MonoTrain$outiter", $outdir, $g_logfile);

    $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
        ."-files \"${g_hdfs_port}$g_bin_herred,${g_hdfs_port}$g_bin_hermap,${g_hdfs_port}$input_model,${g_hdfs_port}$g_hmmlist_mono\" "
        ."-Ddc.input.block.size=$g_input_block_size "
        ."-Ddfs.blocksize=$g_hdfs_block_size "
        ."-numReduceTasks $g_reduces_num "
        ."-input $g_hadoopdir_phone_Mlf -output $outdir "
        ."-mapper  \"./".basename($g_bin_hermap)." -A -V $g_strPruning -p 1 -H ".basename($input_model)." ".basename($g_hmmlist_mono)."\" "
        ."-reducer \"./".basename($g_bin_herred)." -A -V $g_strPruning -p 1 -H ".basename($input_model)." ".basename($g_hmmlist_mono)."\" ";

    $hs->run($hadoop_cmd,$FUNCTION_ENABLE);

###  Calculate new HMM
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_mono/FixSilSp/hmm$outiter");
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
    $cmdline    = "perl,./hadoopcat.pl,$outdir/*part*,output,"
        ."./".basename($g_bin_herred).","
        ."$l_strPruning,"
        #."argvpro_B,"
        ."argvpro_s,./out/occ.0,"
        ."argvpro_M,./out,"
        ."argvpro_H,./MODELS,./".basename($g_hmmlist_mono);

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"./out/MODELS,./out/occ.0\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_hadoopcat,$g_bin_herred,$input_model,$g_hmmlist_mono\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"FixSilSpHmm$outiter\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS $g_hadoopdir_mono/FixSilSp/hmm$outiter/MODELS");
    if( $result_tmp != 0 ){
        die "ERROR : yarn application to make MODELS before FA \n";
    }
    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/occ.0 $g_hadoopdir_mono/FixSilSp/hmm$outiter/occ.0");
    if( $result_tmp != 0 ){
        die "ERROR : yarn application to make occ.0 before FA \n";
    }

    system("touch $g_dir_done/DONE.3.mono$outiter.FixSilSp");
}

if ( $g_bCheckScript == 1 )
{
### 4 Make Alignment directory and hmm0 directory for mono-phone model reestimation
    if ( !-e "$g_dir_done/DONE.4.MakeMultiPronunciationDict" )
    {
        ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
        my $hs      = new HS;
        $cmdline    = "perl,./localrun.pl,./".basename($g_bin_EasyTraining).","
            ."argvpro_MakeMPDict,".basename($g_dict_Mono).",dict.align,$l_strNonPhoneList";
        $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"dict.align\" "
                ."-shell_command \"$cmdline\" "
                ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_EasyTraining,$g_dict_Mono\" "
                ."-work_dir \"$hadoop_work_dir\" "
                ."-queuename $g_jobqueue "
                ."-run_memory $yarn_memory "
                ."-local_run_time_out $yarn_time_out "
                ."-appname \"AlignDict\" "
                );

        ::MakeOrReFreshHadoopDir("$g_hadoopdir_mono/Align");
        ::MakeOrReFreshHadoopDir("$g_hadoopdir_mono/Align/hmm0");

        if(1)
        {
            $result_tmp = system("hdfs dfs -cp -f $g_dict_Align $g_hadoopdir_mono/Align/dict.align");
        }
        else
        {
            $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/dict.align $g_hadoopdir_mono/Align/dict.align");
        }
        if( $result_tmp != 0 ){
            die "ERROR : make dict.align failed \n";
        }

        $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_mono/FixSilSp/hmm$g_nIter_mono/MODELS $g_hadoopdir_mono/Align/hmm0/MODELS");
        if( $result_tmp != 0 ){
            die "ERROR : copy FixSilSp MODELS to Align MODELS.0 failed \n";
        }

        system("touch $g_dir_done/DONE.4.MakeMultiPronunciationDict");
    }

    my $input_dict  = "$g_hadoopdir_mono/Align/dict.align";
    my $input_model = "$g_hadoopdir_mono/Align/hmm0/MODELS";

### 5 FixSilSpMlf
    if ( !-e "$g_dir_done/DONE.5.FixSilSpMlf" )
    {
### edit mlf with dict from word to phone
        my $hs = new HS;
        $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_FixSilSpMlf", $g_hadoopdir_fea_fixspsil, $g_logfile);

        $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
            ."-files ${g_hdfs_port}$g_bin_pakeditmap,${g_hdfs_port}$g_bin_pakeditred,"
            ."${g_hdfs_port}$input_dict,${g_hdfs_port}$g_hmmlist_mono,${g_hdfs_port}$input_model "
            ."-Ddc.input.block.size=$g_input_block_size "
            ."-Ddfs.blocksize=$g_hdfs_block_size "
            ."-numReduceTasks 0 "
            ."-input $g_hadoopdir_src_Mlf -output $g_hadoopdir_fea_fixspsil "
            ."-mapper  \"./".basename($g_bin_pakeditmap)
            ." -FixPhoneMlf ".basename($input_model)." ".basename($g_hmmlist_mono)." ".basename($input_dict)." $g_strBeam\" "
            ."-reducer \"./".basename($g_bin_pakeditred)."\" ";

        $hs->run($hadoop_cmd,$FUNCTION_ENABLE);
        system("touch $g_dir_done/DONE.5.FixSilSpMlf");

    }

### 6 Iteration for mono-phone models
    for (my $nIter = 0 ; $nIter < $g_nIter_afterFA ; $nIter++ )
    {
        my $preiter = $nIter;
        my $outiter = $nIter + 1;
        next if ( -e "$g_dir_done/DONE.6.align$outiter.retrain" );

        my $outdir = "$g_hadoopdir_Mono/Align/$outiter";
        my $input_model = "$g_hadoopdir_mono/Align/hmm$preiter/MODELS";

        my $hs = new HS;
        $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_MonoAfterFA$outiter", $outdir, $g_logfile);

        $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
            ."-files ${g_hdfs_port}$g_bin_herred,${g_hdfs_port}$g_bin_hermap,${g_hdfs_port}$input_model,${g_hdfs_port}$g_hmmlist_mono "
            ."-Ddc.input.block.size=$g_input_block_size "
            ."-Ddfs.blocksize=$g_hdfs_block_size "
            ."-numReduceTasks $g_reduces_num "
            ."-input $g_hadoopdir_fea_fixspsil -output $outdir "
            ."-mapper  \"./".basename($g_bin_hermap)
            ." -A -V $g_strPruning -p 1 -H ".basename($input_model)." ".basename($g_hmmlist_mono)."\" "
            ."-reducer \"./".basename($g_bin_herred)
            ." -A -V $g_strPruning -p 1 -H ".basename($input_model)." ".basename($g_hmmlist_mono)."\" ";

        $hs->run($hadoop_cmd,$FUNCTION_ENABLE);

        ::MakeOrReFreshHadoopDir("$g_hadoopdir_mono/Align/hmm$outiter");

        ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

        $cmdline    = "perl,./hadoopcat.pl,$outdir/*part*,output,"
            ."./".basename($g_bin_herred).","
            ."$l_strPruning,"
            #."argvpro_B,"
            ."argvpro_s,./out/occ.0,"
            ."argvpro_M,./out,"
            ."argvpro_H,./MODELS,".basename($g_hmmlist_mono);

        $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"./out/MODELS,./out/occ.0\" "
                ."-shell_command \"$cmdline\" "
                ."-upload_files \"$g_yarn_jar,$g_bin_hadoopcat,$g_bin_herred,$input_model,$g_hmmlist_mono\" "
                ."-work_dir \"$hadoop_work_dir\" "
                ."-queuename $g_jobqueue "
                ."-run_memory $yarn_memory "
                ."-local_run_time_out $yarn_time_out "
                ."-appname \"AlignHmm$outiter\" "
                );

        $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS $g_hadoopdir_mono/Align/hmm$outiter/MODELS");
        if( $result_tmp != 0 ){
            die "ERROR : make MODELS in Align iterator $outiter \n";
        }
        $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/occ.0 $g_hadoopdir_mono/Align/hmm$outiter/occ.0");
        if( $result_tmp != 0 ){
            die "ERROR : make occ.0 in Align iterator $outiter \n";
        }
        system("touch $g_dir_done/DONE.6.align$outiter.retrain");
    }

    if(!-e "$g_dir_done/DONE.Finish.MonoTraining")
    {
        $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_mono/Align/hmm$g_nIter_afterFA/MODELS $g_hadoopdir_mono/MODELS.f");
        if( $result_tmp != 0 ){
            die "ERROR : copy final Align MODELS failed \n";
        }
        $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_mono/Align/hmm$g_nIter_afterFA/occ.0 $g_hadoopdir_mono/occ.f");
        if( $result_tmp != 0 ){
            die "ERROR : copy final Align occ failed \n";
        }

        system("touch $g_dir_done/DONE.Finish.MonoTraining");
    }
}
else
{
    $g_hadoopdir_fea_fixspsil = $g_hadoopdir_phone_Mlf;
    if (!-e "$g_dir_done/DONE.Finish.MonoTraining")
    {
        $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_mono/FixSilSp/hmm$g_nIter_mono/MODELS $g_hadoopdir_mono/MODELS.f");
        if( $result_tmp != 0 ){
            die "ERROR : copy final Align MODELS failed \n";
        }
        $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_mono/FixSilSp/hmm$g_nIter_mono/occ.0 $g_hadoopdir_mono/occ.f");
        if( $result_tmp != 0 ){
            die "ERROR : copy final Align occ failed \n";
        }

        system("touch $g_dir_done/DONE.Finish.MonoTraining");
    }
}

if($g_auto_qst && !-e "$g_dir_done/DONE.GetQuestionSet")
{
    $cmdline    = join(",", ("perl",basename($g_bin_localrun),"perl",basename($g_bin_autoque),basename("$g_hadoopdir_mono/MODELS.f"),basename("$g_hadoopdir_mono/occ.f"), "HHEd.QS"));

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,HHEd.QS,HHEd.QS.log\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_autoque,$g_hadoopdir_mono/MODELS.f,$g_hadoopdir_mono/occ.f\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"AutoQue\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/HHEd.QS $g_qst_QuestionSet");
    if( $result_tmp != 0){
        die "ERROR : copy question set failed \n";
    }
    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/HHEd.QS.log $g_qst_QuestionSet.log");
    if( $result_tmp != 0 ){
        die "ERROR : copy question set failed \n";
    }
    system("touch $g_dir_done/DONE.GetQuestionSet");
}

#################################################################################
#-------------------------- Context-dependent training -------------------------#
#################################################################################

### 7 Make context-dependent phone Mlf hmmlist.context
if ( !-e "$g_dir_done/DONE.7.Make_hmmlist.context" )
{
### Make context-dependent model directory directory and hmm0 directory
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_context");
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_Context");

    my $hs = new HS;
    $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_generatetriphonelist", "$g_hadoopdir_Context/tp_list", $g_logfile);

    $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
        ."-files ${g_hdfs_port}$g_bin_pakeditmap,${g_hdfs_port}$g_bin_pakeditred "
        ."-Ddc.input.block.size=$g_input_block_size "
        ."-Ddfs.blocksize=$g_hdfs_block_size "
        ."-numReduceTasks $g_reduces_num "
        ."-input $g_hadoopdir_fea_fixspsil -output $g_hadoopdir_Context/tp_list "
        ."-mapper \"./".basename($g_bin_pakeditmap)." -TriphoneList\" -reducer \"./".basename($g_bin_pakeditred)."\" ";

    $hs->run($hadoop_cmd,$FUNCTION_ENABLE);

    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

    $cmdline    = "perl,./hadoopcat.pl,$g_hadoopdir_Context/tp_list/*part*,".basename($g_hmmlist_context)
        .",./".basename($g_bin_pakeditred).","
        ."argvpro_PrintPhoneList";

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"".basename($g_hmmlist_context)."\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_hadoopcat,$g_bin_pakeditred\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"HmmlistContext\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/".basename($g_hmmlist_context)." $g_hmmlist_context");
    if( $result_tmp != 0 ){
        die "ERROR : make hmmlist.context failed \n";
    }

    system("touch $g_dir_done/DONE.7.Make_hmmlist.context");
}

### 8 Clone mono-phone model to context-dependent model
if ( !-e "$g_dir_done/DONE.8.MakeMono2ContextHHEdCommand" )
{
### clone mono-phones to context-dependent model
### make a command file "HHEd.Mono2Context"
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
    $cmdline = "perl,./localrun.pl,./".basename($g_bin_EasyTraining).","
        ."argvpro_MakeM2CHHEdCmd,".basename($g_hmmlist_mono).",".basename($g_hmmlist_context).",HHEd.Mono2Context";

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"HHEd.Mono2Context\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_EasyTraining,$g_hmmlist_context,$g_hmmlist_mono\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"HHEdMono2Context\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/HHEd.Mono2Context $g_hadoopdir_context/HHEd.Mono2Context");
    if( $result_tmp != 0 ){
        die "ERROR : Clone mono-phone MODELS to context MODELS command file failed \n";
    }
    system("touch $g_dir_done/DONE.8.MakeMono2ContextHHEdCommand");
}

### 9 make contextphone HMM
if ( !-e "$g_dir_done/DONE.9.HHEd" )
{
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_context/hmm0");
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

    $cmdline    = "perl,./localrun.pl,./".basename($g_bin_HHEd_ifly).","
        ."argvpro_B,"
        ."argvpro_H,./MODELS.f,"
        ."argvpro_M,./out,HHEd.Mono2Context,".basename($g_hmmlist_mono);

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"./out/MODELS.f\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_HHEd_ifly,$g_hmmlist_context,"
            ."$g_hmmlist_mono,$g_hadoopdir_mono/MODELS.f,$g_hadoopdir_context/HHEd.Mono2Context\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"MakeContextphoneHMM\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS.f $g_hadoopdir_context/hmm0/MODELS");
    if( $result_tmp != 0 ){
        die "ERROR : Clone mono MODELS to context MODELS failed \n";
    }

    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_context/hmm0/MODELS $g_hadoopdir_context/MODELS.0");
    if( $result_tmp != 0 ){
        die "ERROR : copy context MODELS failed , maybe $g_hadoopdir_context/MODELS.0 already exist \n";
    }
    system("touch $g_dir_done/DONE.9.HHEd");
}

### 10 Iteration for context-dependent models
for (my $nIter = 0 ; $nIter < $g_nIter_xwrd ; $nIter++ )
{
    my $preiter = $nIter;
    my $outiter = $nIter + 1;
    next if ( -e "$g_dir_done/DONE.10.xwrd$outiter" );

    my $input_model = "$g_hadoopdir_context/hmm$preiter/MODELS";
    my $outdir      = "$g_hadoopdir_Context/$outiter";

    my $hs = new HS;

    $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_xwrd_$outiter", $outdir, $g_logfile);
    $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
        ."-files ${g_hdfs_port}$g_bin_hermap,${g_hdfs_port}$g_bin_herred,${g_hdfs_port}$input_model,${g_hdfs_port}$g_hmmlist_context "
        ."-Ddc.input.block.size=$g_input_block_size "
        ."-Ddfs.blocksize=$g_hdfs_block_size "
        ."-numReduceTasks $g_reduces_num "
        ."-input $g_hadoopdir_fea_fixspsil -output $outdir "
        ."-mapper  \"./".basename($g_bin_hermap)." -A -V $g_strPruning -p 3 -H ".basename($input_model)." ".basename($g_hmmlist_context)."\" "
        ."-reducer \"./".basename($g_bin_herred)." -A -V $g_strPruning -p 1 -H ".basename($input_model)." ".basename($g_hmmlist_context)."\" ";
    $hs->run($hadoop_cmd,$FUNCTION_ENABLE);

    ::MakeOrReFreshHadoopDir("$g_hadoopdir_context/hmm$outiter");

    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

    $cmdline    = "perl,./hadoopcat.pl,$outdir/*part*,output,"
        ."./".basename($g_bin_herred).","
        ."$l_strPruning,"
        ."argvpro_B,"
        ."argvpro_s,./out/occ.0,"
        ."argvpro_M,./out,"
        ."argvpro_H,./MODELS,".basename($g_hmmlist_context);

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"./out/MODELS,./out/occ.0\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_hadoopcat,$g_bin_herred,$g_hadoopdir_context/hmm$preiter/MODELS,$g_hmmlist_context\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"ContextDependentHmm$outiter\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS $g_hadoopdir_context/hmm$outiter/MODELS");
    if( $result_tmp != 0 ){
        die "ERROR : make MODELS failed in Iteration for context-dependent models $outiter \n";
    }

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/occ.0 $g_hadoopdir_context/hmm$outiter/occ.0");
    if( $result_tmp != 0 ){
        die "ERROR : make occ.0 failed in Iteration for context-dependent models $outiter \n";
    }

    system("touch $g_dir_done/DONE.10.xwrd$outiter");

}

### copy file
if( !-e "$g_dir_done/DONE.Finish.ContextTraining" )
{
    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_context/hmm$g_nIter_xwrd/MODELS $g_hadoopdir_context/MODELS.f");
    if( $result_tmp != 0 )
    {
        die "ERROR : copy final MODELS to context MODELS.f \n";
    }
    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_context/hmm$g_nIter_xwrd/occ.0 $g_hadoopdir_context/occ.f");
    if( $result_tmp != 0 )
    {
        die "ERROR : copy final occ.0 to context occ.f \n";
    }

    system("touch $g_dir_done/DONE.Finish.ContextTraining");
}

#################################################################################
#----------------------- Tied-context-dependent training -----------------------#
#################################################################################

### 11 Decision tree based state tying
if ( !-e "$g_dir_done/DONE.11.DTTie" ) {

    ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext");
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_Tcontext");

    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_context/occ.f $g_hadoopdir_tcontext/occ.0");
    if( $result_tmp != 0 ){
        die "ERROR : copy occ.0 failed \n";
    }

    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_context/MODELS.f $g_hadoopdir_tcontext/MODELS.0");
    if( $result_tmp != 0 ){
        die "ERROR : copy MODELS.0 failed \n";
    }

    my $input_occ   = "$g_hadoopdir_tcontext/occ.0";
    my $input_model = "$g_hadoopdir_tcontext/MODELS.0";

    my $hs = new HS;
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/hmm0");
### make HHEd.conf
    $result_tmp = system("rm -rf $g_dir_done/hhed.conf");
    if ($g_nomerge_clustering) {
        $hs->sys("echo \"TREEMERGE=FALSE\" > $g_dir_done/hhed.conf");
    }
    else {
        $hs->sys("echo \"\" > $g_dir_done/hhed.conf");
    }

    ::DeleteHadoop2Dir("$g_hadoopdir_tcontext/hhed.conf");
    $result_tmp = system("hdfs dfs -put $g_dir_done/hhed.conf $g_hadoopdir_tcontext");
    if( $result_tmp != 0 ){
        die "ERROR : put hhed.conf failed \n";
    }
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

### make the command for HHEd_ifly to tie Decision tree
    $cmdline   = "perl,./localrun.pl,./".basename($g_bin_EasyTraining).","
        ."argvpro_MakeClstHHEdCmd_new,$g_nTieLevel,"
        ."./Clst.log,"
        ."./occ.0,./".basename($g_qst_QuestionSet).",./".basename($g_cmd_EasyTraining_ModelGen).","
        ."./HHEd.Clustering,"
        ."./TREE,./".basename($g_hmmlist_tcontext).","
        ."$g_strOutlier,$g_strInitTB,$g_strTiedNum,$l_strContextIndependentPhone";

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"HHEd.Clustering,Clst.log\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_EasyTraining,$input_occ,$g_qst_QuestionSet,$g_cmd_EasyTraining_ModelGen\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"HHEdClustering\" "
            );

    ::DeleteHadoop2Dir("$g_hadoopdir_tcontext/HHEd.Clustering");
    ::DeleteHadoop2Dir("$g_hadoopdir_tcontext/Clst.log");

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/HHEd.Clustering $g_hadoopdir_tcontext/HHEd.Clustering");
    if( $result_tmp != 0 ){
        die "ERROR : make HHEd.Clustering failed \n";
    }

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/Clst.log $g_hadoopdir_tcontext/Clst.log");
    if( $result_tmp != 0 ){
        die "ERROR : make Clst.log failed \n";
    }

    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
### tie the Decision tree and get MODELS
    $cmdline   = "perl,./localrun.pl,./".basename($g_bin_HHEd_ifly).","
        ."argvpro_B,"
        ."argvpro_H,".basename($input_model).","
        ."argvpro_C,hhed.conf,"
        ."argvpro_M,./out,HHEd.Clustering,".basename($g_hmmlist_context);

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,./out/".basename($input_model).",".basename($g_hmmlist_tcontext).",TREE\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_HHEd_ifly,$g_hadoopdir_tcontext/hhed.conf,"
            ."$g_hadoopdir_tcontext/HHEd.Clustering,$g_hmmlist_context,$input_model,$input_occ\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"InitModelsAfterTied\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/".basename($input_model)." $g_hadoopdir_tcontext/hmm0/".basename($input_model));
    if( $result_tmp != 0 ){
        die "ERROR : make MODELS failed after tie the Decision tree \n";
    }

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/TREE $g_hadoopdir_tcontext/");
    if( $result_tmp != 0 ){
        die "ERROR : make TREE failed after tie the Decision tree \n";
    }

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/".basename($g_hmmlist_tcontext)." $g_hadoopdir_tcontext/".basename($g_hmmlist_tcontext));
    if( $result_tmp != 0 ){
        die "ERROR : make hmmlist.tcontext failed after tie the Decision tree \n";
    }

    system("hdfs dfs -mv $g_hadoopdir_tcontext/hmm0/MODELS.0 $g_hadoopdir_tcontext/hmm0/MODELS");
    system("touch $g_dir_done/DONE.11.DTTie")
}

### 12 Add unseen tri-phone before Mixup

if ( !-e "$g_dir_done/DONE.12.SearchAll.MakeAllXwrdTriPhones" ) ### generate hmmlist.au
{
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
    if ( $g_bhmmlist_au_all == 0 )
    {
        $cmdline = "perl,./localrun.pl,./".basename($g_bin_EasyTraining).","
            ."argvpro_MakeAllXwrdTP".",./".basename($g_dict_Test).",./".basename($g_hmmlist_au)
            .join(",", ("", scalar(@g_strContextFreePhone),@g_strContextFreePhone,scalar(@g_strContextIndependentPhone),@g_strContextIndependentPhone));

        $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,".basename($g_hmmlist_au)."\" "
                ."-shell_command \"$cmdline\" "
                ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_EasyTraining,$g_dict_Test\" "
                ."-work_dir \"$hadoop_work_dir\" "
                ."-queuename $g_jobqueue "
                ."-run_memory $yarn_memory "
                ."-local_run_time_out $yarn_time_out "
                ."-appname \"MakeAllXwrdTP\" "
                );
    }
    else
    {
        $cmdline = "perl,./localrun.pl,perl,".basename($g_bin_genhmmlistau);

        $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,".basename($g_hmmlist_au)."\" "
                ."-shell_command \"$cmdline\" "
                ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_genhmmlistau,$g_hmmlist_mono\" "
                ."-work_dir \"$hadoop_work_dir\" "
                ."-queuename $g_jobqueue "
                ."-run_memory $yarn_memory "
                ."-local_run_time_out $yarn_time_out "
                ."-appname \"MakeHmmlistAU\" "
                );
    }

    system("hdfs dfs -cp -f $hadoop_work_dir/".basename($g_hmmlist_au)." $g_hmmlist_au");
    if( $result_tmp != 0 )
    {
        die "ERROR : make hmmlist.au failed \n";
    }

    system("touch $g_dir_done/DONE.12.SearchAll.MakeAllXwrdTriPhones");
}

### 13 MakeAddunseenHHEdCmd
if ( !-e "$g_dir_done/DONE.13.MakeAddunseenHHEdCmd" )
{
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

    $cmdline = "perl,./localrun.pl,./".basename($g_bin_EasyTraining).","
        ."argvpro_MakeAddunseenHHEdCmd,./HHEd.Addunseen,./TREE,"
        ."./".basename($g_hmmlist_au).",./".basename($g_hmmlist_final);

    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,HHEd.Addunseen\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_EasyTraining,$g_hadoopdir_tcontext/TREE,$g_hmmlist_au\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out "
            ."-appname \"HHEdAddunseen\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/HHEd.Addunseen $g_hadoopdir_tcontext/HHEd.Addunseen");

    if( $result_tmp != 0 )
    {
        die "ERROR : make AddunseenHHEdCmd failed \n";
    }
    $result_tmp = system("touch $g_dir_done/DONE.13.MakeAddunseenHHEdCmd");
}

### 14 Addunseen
if ( !-e "$g_dir_done/DONE.14.HHEd.Addunseen" )
{
    ::MakeOrReFreshHadoopDir("$hadoop_work_dir");
    ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/hmm1");

    $cmdline = "perl,./localrun.pl,./".basename($g_bin_HHEd_ifly).","
        ."argvpro_B,"
        ."argvpro_H,./MODELS,"
        ."argvpro_w,./out/MODELS,./HHEd.Addunseen,./".basename($g_hmmlist_tcontext);

    my $yarn_time_out_      = 100*3600*1000; ###ms
    $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,hmmlist.final,./out/MODELS\" "
            ."-shell_command \"$cmdline\" "
            ."-upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_HHEd_ifly,$g_hadoopdir_tcontext/hmm0/MODELS,"
            ."$g_hadoopdir_tcontext/HHEd.Addunseen,$g_hmmlist_tcontext,$g_hadoopdir_tcontext/TREE,$g_hmmlist_au\" "
            ."-work_dir \"$hadoop_work_dir\" "
            ."-queuename $g_jobqueue "
            ."-run_memory $yarn_memory "
            ."-local_run_time_out $yarn_time_out_ "
            ."-appname \"MakeHmmlistFinal\" "
            );

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/hmmlist.final $g_hmmlist_final");
    if( $result_tmp != 0 )
    {
        die "ERROR : make hmmlist.final failed \n";
    }

    $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS $g_hadoopdir_tcontext/hmm1/MODELS");
    if( $result_tmp != 0 )
    {
        die "ERROR : make MODELS failed after addunseen\n";
    }

    system("touch $g_dir_done/DONE.14.HHEd.Addunseen");
}

### Check mixture-up path
( @g_pth_Phone == @g_pth_NonPhone ) || die "ERROR: Phone and Non-Phone mixture-up path mismatch - (@g_pth_Phone) and (@g_pth_NonPhone)";
( ( $g_pth_Phone[0] == 1 ) && ( $g_pth_NonPhone[0] == 1 ) ) || die "ERROR: First element of the mixture-up path must be 1";

my $input_model = "$g_hadoopdir_tcontext/hmm1/MODELS";
my $input_occ;

### 15 Mixture-up
for ( my $l = 0 ; $l < @g_pth_Phone ; $l++ )
{
    my $m = $g_pth_Phone[$l]; # phone g
        my $n = $g_pth_NonPhone[$l]; # nonphone g

### Increase mixture or copy model
        if ( !-e "$g_dir_done/DONE.15.MixtureUp_$l" )
        {
            if ( $m == 1 && $n == 1 )
            {
                if( !-e "$g_dir_done/DONE.15.MixtureUp_0" )
                {
                    ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/mix$m");
                    ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/mix$m/hmm0");
                    $result_tmp = system("hdfs dfs -cp -f $input_model $g_hadoopdir_tcontext/mix$m/hmm0/MODELS");
                }
                system("touch $g_dir_done/DONE.15.MixtureUp_0");
            }
            else
            {
                if( $m != 1 )
                {
                    ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/mix$m");
                    ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/mix$m/hmm0");
                }
                else
                {
                    ::DeleteHadoop2Dir("$g_hadoopdir_tcontext/mix$m/HHEd.MixtureUp");
                    for (my $nIter = 1 ; $nIter < $g_nIter_cxwrd ; $nIter++ )
                    {
                        ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/mix$m/hmm$nIter");
                    }
                }
                ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

                $cmdline = "perl,./localrun.pl,./".basename($g_bin_EasyTraining).","
                    ."argvpro_MakeMUHHEdCmd,TRI,./HHEd.MixtureUp,$m,".basename($g_hmmlist_mono).",$n,$l_strNonPhoneMU";

                $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,HHEd.MixtureUp\" "
                        ."-shell_command \"$cmdline\" -upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_EasyTraining,$g_hmmlist_mono\" "
                        ."-work_dir \"$hadoop_work_dir\" "
                        ."-queuename $g_jobqueue "
                        ."-run_memory $yarn_memory "
                        ."-local_run_time_out $yarn_time_out "
                        ."-appname \"HHEdMixtureUp\" "
                        );
                $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/HHEd.MixtureUp $g_hadoopdir_tcontext/mix$m/HHEd.MixtureUp");

                if( $result_tmp != 0 )
                {
                    die "ERROR : make mixup command failed \n";
                }

                ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

                $cmdline = "perl,./localrun.pl,./".basename($g_bin_HHEd_ifly).","
                    ."argvpro_B,argvpro_H,".basename($input_model).",argvpro_M,./out,./HHEd.MixtureUp,".basename($g_hmmlist_final);

                $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"stdout,stderr,./out/MODELS\" "
                        ."-shell_command \"$cmdline\" -upload_files \"$g_yarn_jar,$g_bin_localrun,$g_bin_HHEd_ifly,$input_model,$g_hmmlist_final,$g_hadoopdir_tcontext/mix$m/HHEd.MixtureUp\" "
                        ."-work_dir \"$hadoop_work_dir\" "
                        ."-queuename $g_jobqueue "
                        ."-run_memory $yarn_memory "
                        ."-local_run_time_out $yarn_time_out "
                        ."-appname \"MakeMixupModels\" "
                        );
                ::DeleteHadoop2Dir("$g_hadoopdir_tcontext/mix$m/hmm0/MODELS");
                $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS $g_hadoopdir_tcontext/mix$m/hmm0");

                if( $result_tmp != 0 )
                {
                    die "ERROR : make mixup MODELS failed \n";
                }

                system("touch $g_dir_done/DONE.15.MixtureUp_$l");
            }
        }
    $input_model = "$g_hadoopdir_tcontext/mix$m/hmm0/MODELS";

### 16 Iteration for context-dependent models
    for (my $nIter = 0 ; $nIter < $g_nIter_cxwrd ; $nIter++ )
    {
        my $preiter = $nIter;
        my $outiter = $nIter + 1;
        next if ( -e "$g_dir_done/DONE.16.mix$l.hmm$outiter" );

        my $outdir   = "$g_hadoopdir_Tcontext/mix${l}_$outiter";
        $input_model = "$g_hadoopdir_tcontext/mix$m/hmm$preiter/MODELS";

        my $hs = new HS;
        $hs->init_hadoop($current_jobname = "${g_mapred_task_name}_mix${l}_$outiter", $outdir, $g_logfile);

        $hadoop_cmd = "hadoop jar $l_streaming -Dmapreduce.job.queuename=$g_jobqueue -Dmapreduce.job.name=$current_jobname "
            ."-files ${g_hdfs_port}$g_bin_hermap,${g_hdfs_port}$g_bin_herred,${g_hdfs_port}$input_model,${g_hdfs_port}$g_hmmlist_final "
            ."-Ddc.input.block.size=$g_input_block_size "
            ."-Dmapreduce.reduce.memory.mb=$mapred_memory "
            ."-Ddfs.blocksize=$g_hdfs_block_size "
            ."-numReduceTasks $g_reduces_num "
            ."-input $g_hadoopdir_fea_fixspsil -output $outdir "
            ."-mapper  \"./".basename($g_bin_hermap)." -A -V $g_strPruning -p 3 -H ".basename($input_model)." ".basename($g_hmmlist_final)."\" "
            ."-reducer \"./".basename($g_bin_herred)." -A -V $g_strPruning -p 1 -H ".basename($input_model)." ".basename($g_hmmlist_final)."\" ";

        $hs->run($hadoop_cmd,$FUNCTION_ENABLE);

        ::MakeOrReFreshHadoopDir("$g_hadoopdir_tcontext/mix$m/hmm$outiter");

        ::MakeOrReFreshHadoopDir("$hadoop_work_dir");

        $cmdline    = "perl,./hadoopcat.pl,$outdir/*part*,output,"
            ."./".basename($g_bin_herred).","
            ."$l_strPruning,"
            ."argvpro_B,argvpro_s,./out/occ.0,"
            ."argvpro_M,./out,"
            ."argvpro_H,./MODELS,".basename($g_hmmlist_final);

        $result_tmp = ::PR("yarn jar $l_yarn_jar MLETest -output \"./out/MODELS,./out/occ.0\" "
                ."-shell_command \"$cmdline\" "
                ."-upload_files \"$g_yarn_jar,$g_bin_hadoopcat,$g_bin_herred,$input_model,$g_hmmlist_final\" "
                ."-work_dir \"$hadoop_work_dir\" "
                ."-queuename $g_jobqueue "
                ."-run_memory $yarn_memory "
                ."-local_run_time_out $yarn_time_out "
                ."-appname \"Mixup$m\" "
                );

        $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/MODELS $g_hadoopdir_tcontext/mix$m/hmm$outiter");
        if( $result_tmp != 0 ){
            die "ERROR : failed make MODELS in Iteration for context-dependent models mix$m hmm$outiter \n";
        }

        $result_tmp = system("hdfs dfs -cp -f $hadoop_work_dir/occ.0 $g_hadoopdir_tcontext/mix$m/hmm$outiter/occ.0");
        if( $result_tmp != 0 ){
            die "ERROR : failed make occ in Iteration for context-dependent models mix$m hmm$outiter \n";
        }

        $result_tmp = system("touch $g_dir_done/DONE.16.mix$l.hmm$outiter");
    }
    $input_model = "$g_hadoopdir_tcontext/mix$m/hmm$g_nIter_cxwrd/MODELS";
    $input_occ   = "$g_hadoopdir_tcontext/mix$m/hmm$g_nIter_cxwrd/occ.0";
}

#################################################################################
#----------------------------- Finishing training ------------------------------#
#################################################################################

if(!-e "$g_dir_done/DONE.16.CopyFinalRes")
{
    my $tree    = "TREE";
    my $model   = "MODELS";
    my $hmmlist = "hmmlist.final";

    ::MakeOrReFreshHadoopDir("$g_hadoopdir_final");

    $result_tmp = system("hdfs dfs -cp -f $g_hadoopdir_tcontext/TREE $g_hadoopdir_final/$tree");
    $result_tmp = system("hdfs dfs -cp -f $input_model $g_hadoopdir_final/$model");
    $result_tmp = system("hdfs dfs -cp -f $g_hmmlist_final $g_hadoopdir_final/$hmmlist");

    ::MakeOrReFreshDir("$g_dir_final");
    $result_tmp = system("hdfs dfs -get $g_hadoopdir_mono/Align/dict.align $g_dir_final/");
    $result_tmp = system("hdfs dfs -get $g_hadoopdir_final/* $g_dir_final/");
    $result_tmp = system("hdfs dfs -get $g_hmmlist_mono $g_dir_final/");
    $result_tmp = system("hdfs dfs -get $g_hmmlist_au $g_dir_final/");
    ::PR("$l_bin_hhed -H $g_dir_final/$model -w $g_dir_final/$model.txt $l_txt_null $g_dir_final/$hmmlist");

    system("touch $g_dir_done/DONE.16.CopyFinalRes");
}
