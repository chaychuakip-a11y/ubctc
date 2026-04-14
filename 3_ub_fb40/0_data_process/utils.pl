use strict;

sub load_config {
    my $config_file = "config.json";
    if (!-e $config_file) {
        die "Error: config.json not found in current directory.\n";
    }

    my %config;
    open(my $fh, "<", $config_file) or die "can't open $config_file: $!";
    while (my $line = <$fh>) {
        # Match "key": "value" or "key": numeric_value
        if ($line =~ /"([^"]+)"\s*:\s*"([^"]+)"/ || $line =~ /"([^"]+)"\s*:\s*([\d\.]+)/) {
            $config{$1} = $2;
        }
    }
    close($fh);

    if (!%config) {
        die "Error: Failed to parse any configuration from $config_file. Please check the file format.\n";
    }

    return \%config;
}

1;
