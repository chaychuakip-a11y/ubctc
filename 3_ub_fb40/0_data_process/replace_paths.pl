use strict;
use warnings;
use File::Basename;

my @files = glob("*.pl");
foreach my $file (@files) {
    next if $file eq 'utils.pl' || $file eq basename($0);
    
    my $content;
    {
        local $/;
        open(my $fh, "<", $file) or die "can't open $file: $!";
        $content = <$fh>;
        close($fh);
    }

    my $changed = 0;

    # 1. Wrap config loading in BEGIN block if not already
    if ($content =~ s/require "\.\/utils\.pl";\s+my \$config_data = load_config\(\);/my \$config_data;\nBEGIN {\n    require ".\/utils.pl";\n    \$config_data = load_config();\n}/g) {
        $changed = 1;
    }

    # 2. Replace use lib
    if ($content =~ s/use lib "\/work1\/asrdictt\/taoyu\/sbin";/use lib \$config_data->{dir_sbin};/g) {
        $changed = 1;
    }

    # 3. Replace sbin paths in system calls or other strings
    if ($content =~ s/\/work1\/asrdictt\/taoyu\/sbin\//\$config_data->{dir_sbin}\//g) {
        $changed = 1;
    }

    # 4. Replace the specific large path variables
    if ($content =~ s/my \$config\s+=\s+"\/work1\/asrdictt\/taoyu\/bin\/atom-v20151016b\/atom_hadoop_dnnfa\.fb72\.cfg";/my \$config            = \$config_data->{atom_config};/g) { $changed = 1; }
    if ($content =~ s/my \$dir_ac\s+=\s+"\/work1\/asrdictt\/taoyu\/mlg\/korean\/am\/1_mle_mfc\/final_s9k";/my \$dir_ac            = \$config_data->{dir_ac};/g) { $changed = 1; }
    if ($content =~ s/my \$dir_lm\s+=\s+"\/work1\/asrdictt\/taoyu\/mlg\/korean\/res\/res_fa\/out_package_1gram";/my \$dir_lm            = \$config_data->{dir_lm};/g) { $changed = 1; }
    if ($content =~ s/my \$wts_file\s+=\s+"\/work1\/asrdictt\/taoyu\/mlg\/korean\/am\/2_dnn_fb72\/3_train\/mlp-ring-h2048_2048_2048_2048_2048_2048_512-cw11-targ9004-step1_b4096_jumpframe3-2-5_discard0\/mlp\.99\.wts\.merge";/my \$wts_file          = \$config_data->{wts_file};/g) { $changed = 1; }
    if ($content =~ s/my \$fea_norm\s+=\s+"\/work1\/asrdictt\/taoyu\/mlg\/korean\/am\/2_dnn_fb72\/1_down_pfile\/lib_fb72\/fea\.norm";/my \$fea_norm          = \$config_data->{fea_norm};/g) { $changed = 1; }
    if ($content =~ s/my \$state_count\s+=\s+"\/work1\/asrdictt\/taoyu\/mlg\/korean\/am\/2_dnn_fb72\/1_down_pfile\/lib_fb72\/states\.count\.low100\.txt";/my \$state_count       = \$config_data->{state_count};/g) { $changed = 1; }

    # 5. Replace other /work1/asrdictt/taoyu/bin/ paths
    if ($content =~ s/\/work1\/asrdictt\/taoyu\/bin\//\$config_data->{dir_bin}\//g) {
        $changed = 1;
    }

    if ($changed) {
        open(my $fh_out, ">", $file) or die "can't write to $file: $!";
        print $fh_out $content;
        close($fh_out);
        print "Updated $file\n";
    }
}
