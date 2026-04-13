# 数据处理脚本配置化说明 (2026-04-13)

本目录下的数据处理脚本（Perl）已完成配置化改造，不再需要手动在每个脚本中修改硬编码路径。

## 1. 核心文件说明

- **`config.json`**: 集中管理 HDFS 路径、Hadoop 队列、二进制工具路径及模型文件路径。
- **`utils.pl`**: 通用工具脚本，负责加载并解析 `config.json`。
- **`*.pl`**: 业务脚本，已修改为通过 `$config_data` 获取配置。

## 2. 修改配置的方法

### 全局修改
如果需要更改 HDFS 根目录或 Hadoop 队列，请直接编辑 `config.json`：
```json
{
  "hdfs_src_root": "/new/path/to/src",
  "hdfs_out_root": "/new/path/to/out",
  "jobqueue": "nlp",
  ...
}
```

### 运行时指定输入目录
现在所有脚本都支持通过第一个命令行参数指定输入目录。如果指定了参数，脚本将忽略 `config.json` 中的默认输入路径。

**示例：**
```bash
# 使用 config.json 中的默认路径运行
perl 1_dnnfa.pl

# 运行时指定特定的 HDFS 输入目录
perl 1_dnnfa.pl /workdir/asrdictt/custom_input/korean_data
```

## 3. 注意事项
- 脚本依赖 Perl 的 `JSON::PP` 模块（Perl 5.14+ 标准库自带）。
- 确保 `utils.pl` 和 `config.json` 始终与业务脚本处于同一目录下。
- 若内网环境无法访问 HDFS 默认根目录，请优先检查 `config.json` 中的 `hdfs_src_root` 和 `hdfs_out_root`。
