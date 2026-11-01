# BEPC

从 `/data/wbb/vispec-clean-benchmark-20261031` 保持原样的官方 ViSpec、外置 BEPC 控制器、五任务基准框架，以及长输出和 mismatch 预算消融的独立版本。

## 下载

```bash
git clone --recurse-submodules https://github.com/wangbanbu/BEPC.git
cd BEPC
python tools/verify_snapshot.py
```

已经下载但未初始化子模块时，执行 `git submodule update --init --recursive`。

官方 ViSpec 固定为 `49a33ff0ca31e281b9dc8803fd6f3041c6c0e135`；LVSpec 参考实现固定为 `84dde16ea16b39402552eb4eb95d26f9c5101d15`。两者通过 Git 子模块保留独立来源与原始提交，BEPC 实现在 `harness/` 中。

## 代码入口

| 路径 | 内容 |
|---|---|
| `upstream/ViSpec/` | 官方 ViSpec 原始框架、模型实现及依赖清单 |
| `harness/refined_math.py` | BEPC 子空间、分量与分数计算 |
| `harness/refined_observer.py` | 原生隐藏状态及视觉/文本上下文观察器 |
| `harness/refined_controller.py` | BEPC 连续前缀验证、mismatch 预算和路径提交 |
| `harness/speed_controller.py` | 使用相同算法参数的快速控制器 |
| `harness/compact_logits.py`、`harness/terminal_update.py` | 公共计算优化 |
| `harness/run_benchmark.py` | 短答案五任务 AR / ViSpec / Strict / BEPC / CCVE / LVSpec 比较 |
| `configs/`、`tests/` | 原始冻结配置与 CPU 测试 |
| `long_output/harness/`、`long_output/single_pass/` | 长输出框架与单次正式运行入口 |
| `long_output/bepc_optimization_20261101/` | 原版、等价快速版和 rescue=0.625 实验版本 |
| `long_output/bepc_vs_margin_20261101/` | BEPC 与 Margin 对照 |
| `long_output/bepc_response_uncapped_20261101/` | 取消整题累计 mismatch 上限，保留每轮上限 |
| `long_output/bepc_round_uncapped_20261101/` | 进一步取消每轮 mismatch 上限 |
| `long_output/bepc_margin_token_audit_20261101/source/` | token 变化与共同前缀诊断工具 |
| `reference/LVSpec/` | 官方 LVSpec 参考实现 |

原版 BEPC 的固定设置为 U8/P4、文本窗口32、视觉池64、gamma=0.5、kappa=1、eta=1e-4、tau=1、双向修正、每轮2次/整题4次真实 mismatch 预算。不同预算版本各自保留协议和实现，使用时按对应目录选择。

## 环境与运行

`requirements.txt` 逐字复制自固定版本的官方 ViSpec，其中包含 CUDA 版本相关的 PyTorch 依赖；安装时根据实际 CUDA 环境处理。源实验使用 Python 环境 `/data/wbb/miniconda3/envs/spec/bin/python`。

已有 CPU 测试可以统一运行（部分评测测试需要原工作区使用的官方评分器）：

```bash
python tools/run_cpu_tests.py
```

原始实验说明见 [README_BENCHMARK_ORIGINAL.md](README_BENCHMARK_ORIGINAL.md)，短输出协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。各长输出版本的协议位于各自目录。原始阶段命令包括：

```bash
CUDA_VISIBLE_DEVICES=0 python harness/run_benchmark.py --phase parity --shard 0
CUDA_VISIBLE_DEVICES=0 python harness/run_benchmark.py --phase development --shard 0
python harness/select_lvspec.py
python harness/prepare_experiment.py --stage freeze
CUDA_VISIBLE_DEVICES=0 python harness/run_benchmark.py --phase main --shard 0
```

正式实验使用两个 shard，另一个在 GPU 1 上以 `--shard 1` 运行。配置保留原机器的模型、图片、输入及官方评分工具路径和 SHA；其他机器需要准备相应资源、适配路径，并为新实验重新建立身份与配置记录。`prepare_experiment.py` 也依赖原工作区的模型身份文件。原配置属于历史冻结证据，修改后的配置需要作为新实验保存。

上传前已通过全部160项 CPU 测试，覆盖21个测试模块；验证记录见 [SNAPSHOT_VALIDATION.json](docs/SNAPSHOT_VALIDATION.json)。这些检查在原工作区依赖可用的环境中执行，未重跑 GPU 实验。

## 快照与实验记录

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) 记录每份原始文件的 SHA-256、官方提交与上传范围。原来16个绝对路径软链接已经保存为实际文件，源码字节保持一致。`tools/verify_snapshot.py` 可检查源码摘要、Python 语法和两个子模块的提交及干净状态。

`reports/` 和各实验的 `reports/`、`audit/` 保留已有报告与审计记录；报告中的数值和结论没有重新计算。输入 JSON 与标签清单保留，模型权重、图片、逐请求 `results/`、官方评分 `metrics/` 缓存、运行日志和生成的 token 审核数据未包含在本快照内。完整结果复核需要原始实验归档。

五任务原始完整报告：[FULL_EXPERIMENT_REPORT.md](reports/FULL_EXPERIMENT_REPORT.md)。该实验为每任务100组测试的小样本比较；历史样本曝光、自动质量指标和具体运行口径以各原始协议及报告为准。
