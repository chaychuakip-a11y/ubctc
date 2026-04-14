use strict;

sub load_config {
    my $config_file = "config.json";
    if (!-e $config_file) {
        die "Error: config.json not found in current directory.\n";
    }

    my %config;
    open(my $fh, "<", $config_file) or die "can't open $config_file: $!";
    while (my $line = <$fh>) {
        # 移除换行符和首尾空格
        $line =~ s/[\r\n]//g;
        $line =~ s/^\s+|\s+$//g;
        
        # 匹配 "key" : "value"
        if ($line =~ /"([^"]+)"\s*:\s*"([^"]+)"/) {
            $config{$1} = $2;
        }
        # 匹配 "key" : 123
        elsif ($line =~ /"([^"]+)"\s*:\s*([\d\.]+)/) {
            $config{$1} = $2;
        }
    }
    close($fh);

    if (!%config) {
        die "Error: Failed to parse any configuration from $config_file. Please check file format.\n";
    }

    return \%config;
}

1;
