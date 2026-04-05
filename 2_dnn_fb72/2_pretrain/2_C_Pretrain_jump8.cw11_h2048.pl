use strict;

my $dir_lib0       = "/work1/asrdictt/taoyu/mlg/korean/am/2_dnn_fb72/1_down_pfile/lib_fb72";
my $dir_lib        = $dir_lib0; $dir_lib =~ s#/work1/#/yrfs4/#;

my $bin_pretrain   = "source /home3/asrdictt/taoyu/cuda11.1.bashrc;/work1/asrdictt/taoyu/tools/jumpframe_Code_v2/Pretrain-A";
my $bin_get_rand   = "/work1/asrdictt/taoyu/tools/bin/get_randem";
my $bin_wts_concat = "/work1/asrdictt/taoyu/tools/bin/wts_concat";
my $bin_get_rand1  = "/work1/asrdictt/taoyu/tools/bin/get_normalized_rand_wts";
my $bin_wts_concat1= "/work1/asrdictt/taoyu/tools/bin/wts_concat_randlayers";
my $sbin_wait_file = "/work1/asrdictt/taoyu/sbin/wait_file.pl";

my $framefea       = 72;
my $context        = 11;
my $rank           = 512;
my $numPhones      = 9004;
my @hid_layers     = (2048, 2048, 2048, 2048, 2048, 2048);
my @epoches        = (4,3,3,3,3,3); ##(4,3,3,3,3,3);
@epoches == @hid_layers || die;

my $range          = 0.1;
my $lrate          = 0.0025;
my $bunchsize      = 1024;
my $traincache     = 256;
my $mometum        = 0.9;
my $weightcost     = 0.0008;

my $gpu_selected   = 0;
my $nkindmod       = 3;
my $nmod           = "8,8,8";

my $curlayer       = 1;
my $curepoch       = 1;

my $numlayer       = @hid_layers+1;
my $insize         = $framefea*$context;
my $numHid         = join("_", @hid_layers);
my $hidname        = join(",", @hid_layers);
my $layerepochs    = join(",", @epoches);

my $dir_wts        = "./cw${context}_h${numHid}_lrate${lrate}_M${mometum}_decay${weightcost}_bunch${bunchsize}_jumpframe8";
my $out_wts        = "$dir_wts/mlp.0.wts";
my $out_wts_lr     = "$dir_wts/mlp.0.rank$rank.wts";

my $numbunches;
my $cmd;

system("$sbin_wait_file $dir_lib/trainingbatchdata.done");

$numbunches = -s "$dir_lib/trainingbatchdata";
$numbunches = int($numbunches/4/$framefea/$bunchsize);
$numbunches > 0 || die "Error: wrong bunch number: $numbunches\n";

print "numbunches: $numbunches\n";

mkdir $dir_wts;

$cmd = "$bin_pretrain datadir=$dir_lib  wtsdir=$dir_wts  gpu_selected=$gpu_selected layersizes=$insize,$hidname  learnrate=$lrate curlayer=$curlayer curepoch=$curepoch numlayers=$numlayer bunchsize=$bunchsize context=$context feadim=$framefea traincache=$traincache numbunches=$numbunches  mometum=$mometum weightcost=$weightcost layerepochs=$layerepochs nkindmod=$nkindmod nmod=$nmod";
print $cmd."\n";
!system($cmd) or die;

$cmd = "$bin_get_rand $hid_layers[-1] $numPhones $range $dir_wts/$hid_layers[-1]\_$numPhones.random.wts";
print $cmd."\n";
system($cmd);

$numlayer = @hid_layers+2;
$cmd = "$bin_wts_concat $out_wts $numlayer $insize";

foreach my $i(0..$#hid_layers)
{
	my $j = $i+1;
	$cmd .= " $hid_layers[$i] $dir_wts/rbm$j.wts";
}
$cmd .= " $numPhones $dir_wts/$hid_layers[-1]\_$numPhones.random.wts";
print $cmd."\n";
!system($cmd) or die;
system("touch $out_wts.done");

if($rank > 0)
{
	$cmd = "$bin_get_rand1 3 $hid_layers[-1] $rank $numPhones $dir_wts $range";
	system($cmd);

	$numlayer = @hid_layers+1;
	$cmd = "$bin_wts_concat1 $out_wts_lr $out_wts $numlayer $hid_layers[-1] $rank $dir_wts/$hid_layers[-1]_$rank.rand_norm.wts $numPhones $dir_wts/${rank}_$numPhones.rand_norm.wts";
	system($cmd);
	system("touch $out_wts_lr.done");
}