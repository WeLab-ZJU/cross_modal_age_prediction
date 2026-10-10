# 历史实验配置

本目录的 `.sh.txt` 文件仅作为配置参考，不是可直接运行的启动脚本。

- `human/` 包含人类眼球实验配置。
- `mouse/` 包含小鼠眼球及多模态实验配置。
- `<HUMAN_DATA_ROOT>`、`<HUMAN_OUTPUT_ROOT>`、`<PRETRAINED_ROOT>` 等为路径占位符。
- 同一变量多次赋值时，以最后一次为准。复用配置前请核对 GPU 数量、
  图像尺寸和支持的参数。

运行命令见根 README、`run.py`、`scripts/` 和 `mouse/demo/`。
配置注意事项见 `docs/USAGE_NOTES.md`。
