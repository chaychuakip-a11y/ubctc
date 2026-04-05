#!/usr/bin/perl
use strict;
use warnings;

@ARGV >= 1 || die "usage: pl split_id\n";

my (@split_id) = @ARGV;

my $maxframe = 100000;
my $CVnum    = 1000;
my $npart2   = 1;
my $frameinc = 1;

my $dir_pwd  = `pwd`; chomp($dir_pwd); $dir_pwd =~ s#/work1/#/yrfs4/#;
my $dir_lib  = "$dir_pwd/lib_fb40";
# my $dir_lmdb = "$dir_pwd/lmdb_fb40";    # 运行出现 Bus error , 把输出目录删除，使用新的目录执行
my $dir_lmdb = "/train34/sli/permanent/taoyu/mlg/korean/am/4_cnn_fb40/1_down_pfile/lmdb_fb40";
my $nSplit   = 10;
my $feanorm  = "./lib_fb40/fea.norm";
while (!-e "$feanorm.done") { sleep(20); print "waitting file: $feanorm.done\n"; }

my $bin_pak2lmdb    = "/work1/asrdictt/taoyu/tools/bin/pak2lmdb";
my $bin_gen_keylist = "/work1/asrdictt/taoyu/tools/bin/NetTrain_generate_keylist.bin";
my $bin_pfile_info  = "/work1/asrdictt/taoyu/tools/QN/bin/pfile_info";

foreach my $split_id (@split_id)
{
    $split_id < $nSplit || die "Error: not support";
}

system("rm -rf stop");
foreach my $i (@split_id)
{
    my $ii = $i;

    $CVnum = 0 if ($i > 0);
    my $feapfile = "$dir_lib/fea.pfile$ii";
    my $labpfile = "$dir_lib/lab.pfile$ii";

    my $totalsent = `$bin_pfile_info $labpfile | tail -n 1 | sed "s: .*::"`;

    my $partnum = int(($totalsent - $CVnum) / $npart2);

    if ($i == 0 && (!-e "$dir_lmdb/test/data.mdb" || -s "$dir_lmdb/test/data.mdb" < 100))
    {
        my $begin         = $totalsent - $CVnum;
        my $end           = $totalsent - 1;
        my $dir_lmdb_part = "$dir_lmdb/test";
        system("rm -rf $dir_lmdb_part");
        system("mkdir -p $dir_lmdb_part");
        print "$begin $end\n";
        system("$bin_pak2lmdb $feapfile $labpfile $feanorm $dir_lmdb_part $begin $end $maxframe $frameinc >$dir_lmdb_part/info.txt");
        die "get lmdb $dir_lmdb_part faild." if (!-e "$dir_lmdb_part/data.mdb" || -s "$dir_lmdb_part/data.mdb" < 100);
        system("$bin_gen_keylist $dir_lmdb_part $dir_lmdb_part/keys.txt");
        system("touch $dir_lmdb_part/done");
    }

    foreach (my $j = 0; $j < $npart2; $j++)
    {
        my $begin         = $partnum * $j;
        my $end           = $partnum * ($j + 1) - 1;
        my $id            = $i * $npart2 + $j;
        my $dir_lmdb_part = "$dir_lmdb/train_part$id";

        while (!-e "$dir_lmdb_part/data.mdb" || -s "$dir_lmdb_part/data.mdb" < 100)
        {
            print "sub $dir_lmdb_part\n";
            print "$begin $end\n";
            system("rm -rf $dir_lmdb_part");
            system("mkdir -p $dir_lmdb_part");

            system("$bin_pak2lmdb $feapfile $labpfile $feanorm $dir_lmdb_part $begin $end $maxframe $frameinc >$dir_lmdb_part/info.txt");
        }
        die "get lmdb $dir_lmdb_part faild." if (!-e "$dir_lmdb_part/data.mdb" || -s "$dir_lmdb_part/data.mdb" < 100);
        print("$bin_gen_keylist $dir_lmdb_part $dir_lmdb_part/keys.txt\n");
        system("$bin_gen_keylist $dir_lmdb_part $dir_lmdb_part/keys.txt");
        system("touch $dir_lmdb_part/done");
        die "stop." if (-e "stop");

        print "finish $dir_lmdb_part\n";
    }
}
