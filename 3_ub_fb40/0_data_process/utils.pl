use strict;
use JSON::PP;

sub load_config {
    my $config_file = "config.json";
    if (!-e $config_file) {
        die "Error: config.json not found in current directory.\n";
    }
    my $json_text = do {
        open(my $fh, "<", $config_file) or die "can't open $config_file: $!";
        local $/;
        <$fh>
    };
    return decode_json($json_text);
}

1;
