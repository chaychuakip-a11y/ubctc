import glob
import re
import os

files = glob.glob("*.pl")
for file_path in files:
    if file_path == 'utils.pl' or file_path == 'replace_paths.py' or file_path == 'replace_paths.pl':
        continue
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            content = f.read()

    changed = False

    # 1. Wrap config loading in BEGIN block
    pattern1 = r'require "\./utils\.pl";\s+my \$config_data = load_config\(\);'
    replacement1 = 'require "./utils.pl";\nmy $config_data;\nBEGIN {\n    $config_data = load_config();\n}'
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        changed = True

    # 2. Replace use lib
    pattern2 = r'use lib "/work1/asrdictt/taoyu/sbin";'
    replacement2 = 'use lib $config_data->{dir_sbin};'
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        changed = True

    # 3. Replace sbin paths
    if '/work1/asrdictt/taoyu/sbin/' in content:
        content = content.replace('/work1/asrdictt/taoyu/sbin/', '$config_data->{dir_sbin}/')
        changed = True

    # 4. Replace specific variables (handle different spacings)
    # my $config            = "/work1/asrdictt/taoyu/bin/atom-v20151016b/atom_hadoop_dnnfa.fb72.cfg";
    repl_vars = [
        (r'my \$config\s+=\s+"/work1/asrdictt/taoyu/bin/atom-v20151016b/atom_hadoop_dnnfa\.fb72\.cfg";', 'my $config            = $config_data->{atom_config};'),
        (r'my \$dir_ac\s+=\s+"/work1/asrdictt/taoyu/mlg/korean/am/1_mle_mfc/final_s9k";', 'my $dir_ac            = $config_data->{dir_ac};'),
        (r'my \$dir_lm\s+=\s+"/work1/asrdictt/taoyu/mlg/korean/res/res_fa/out_package_1gram";', 'my $dir_lm            = $config_data->{dir_lm};'),
        (r'my \$wts_file\s+=\s+"/work1/asrdictt/taoyu/mlg/korean/am/2_dnn_fb72/3_train/mlp-ring-h2048_2048_2048_2048_2048_2048_512-cw11-targ9004-step1_b4096_jumpframe3-2-5_discard0/mlp\.99\.wts\.merge";', 'my $wts_file          = $config_data->{wts_file};'),
        (r'my \$fea_norm\s+=\s+"/work1/asrdictt/taoyu/mlg/korean/am/2_dnn_fb72/1_down_pfile/lib_fb72/fea\.norm";', 'my $fea_norm          = $config_data->{fea_norm};'),
        (r'my \$state_count\s+=\s+"/work1/asrdictt/taoyu/mlg/korean/am/2_dnn_fb72/1_down_pfile/lib_fb72/states\.count\.low100\.txt";', 'my $state_count       = $config_data->{state_count};')
    ]

    for pattern, replacement in repl_vars:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changed = True

    # 5. Replace other /work1/asrdictt/taoyu/bin/ paths
    if '/work1/asrdictt/taoyu/bin/' in content:
        content = content.replace('/work1/asrdictt/taoyu/bin/', '$config_data->{dir_bin}/')
        changed = True

    if changed:
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print(f"Updated {file_path}")
