import os
import re

# 目标替换块：更稳健的加载逻辑
new_header = '''my $config_data;
BEGIN {
    require "./utils.pl";
    $config_data = load_config();
    if (defined $config_data->{dir_sbin}) {
        unshift @INC, $config_data->{dir_sbin};
    }
}
use share_hadoop;'''

# 通用的旧块正则（匹配之前的各种 BEGIN/require 变体）
old_block_pattern = re.compile(
    r'my \$config_data;.*?BEGIN \{.*?require "\./utils\.pl";.*?\}.*?use share_hadoop;',
    re.DOTALL
)

pl_files = [f for f in os.listdir('.') if f.endswith('.pl') and f != 'utils.pl' and f != 'fix_inc.pl']

for file_path in pl_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 修复可能被误改的 @INC
    content = content.replace('@fix_inc.pl', '@INC')
    
    # 2. 替换为新 Header
    new_content = old_block_pattern.sub(new_header, content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(new_content)
        print(f"Updated {file_path}")
    else:
        # 如果正则没匹配到（可能是结构略有不同），直接检查 share_hadoop 前的内容
        print(f"Warning: Regex didn't match in {file_path}, checking manually.")
