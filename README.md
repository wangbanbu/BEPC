# BEPC

ViSpec 框架与 BEPC 算法实现。默认算法**同时取消每轮最多2次、整题最多4次真实不匹配预算约束**。

BEPC 使用 U8/P4、文本窗口32、视觉池64、gamma=0.5、kappa=1、eta=1e-4、tau=1 和双向修正。每个不匹配仍须满足 BEPC 分数条件；EOS、padding、输出边界和最长连续前缀选择继续生效。

## BEPC 代码位置

| 功能 | 文件 | 主要类或函数 |
|---|---|---|
| 默认入口与参数 | [bepc.py](bepc.py) | `generate()`、`default_variant()`、`DEFAULT_VARIANT`、`BEPCController` |
| 取消每轮不匹配次数上限，作为默认控制器 | [harness/round_uncapped_controller.py](harness/round_uncapped_controller.py) | `NoRoundCapBEPC` |
| 取消整题累计不匹配次数上限 | [harness/uncapped_controller.py](harness/uncapped_controller.py) | `NoResponseCapBEPC` |
| 无上限控制器继承的执行优化 | [harness/bepc_experimental.py](harness/bepc_experimental.py) | `ExperimentalBEPC` |
| 视觉/文本子空间构建与 BEPC 分数计算 | [harness/refined_math.py](harness/refined_math.py) | `RefinedSpaces`、`pool_visual_rows()`、`score_components()` |
| 连续前缀扫描与实际选中路径提交 | [harness/refined_controller.py](harness/refined_controller.py) | `budgeted_scan()`、`RefinedController.evaluate()`、`RefinedController.update()` |
| 快速候选验证与视觉优先剪枝 | [harness/speed_controller.py](harness/speed_controller.py) | `SpeedController.allowed()`、`SpeedController.evaluate()` |
| 原生隐藏状态与视觉/文本上下文获取 | [harness/refined_observer.py](harness/refined_observer.py)、[harness/speed_observer.py](harness/speed_observer.py) | `RefinedContextObserver`、`SpeedContextObserver` |
| 推理中的 logits 与末轮构树优化 | [harness/compact_logits.py](harness/compact_logits.py)、[harness/terminal_update.py](harness/terminal_update.py) | `CompactTreeLogits`、`tree_decoding()`、`TerminalUpdate` |

`bepc.py` 中的 `BEPCController` 指向 `NoRoundCapBEPC`。该类继承 `NoResponseCapBEPC`，先取消整题累计上限，再把每轮预算设为 `math.inf`，因此实际解码不受这两类次数上限约束。

## ViSpec 框架位置

原生模型入口为 [vispec/model/spec_model_ours.py](vispec/model/spec_model_ours.py) 中的 `SpecModel`。Qwen2.5-VL 模型实现位于 [vispec/model/modeling_qwen2_5_vl_kv.py](vispec/model/modeling_qwen2_5_vl_kv.py)，KV cache 位于 [vispec/model/kv_cache.py](vispec/model/kv_cache.py)，树解码辅助函数位于 [vispec/model/utils.py](vispec/model/utils.py)。

单元测试位于 [tests/](tests/)，其中 [test_round_uncapped.py](tests/test_round_uncapped.py) 检查无次数上限控制器，[test_refined_math.py](tests/test_refined_math.py) 检查子空间和评分计算。

## 使用

安装 `requirements.txt` 中的依赖，并根据实际 CUDA 环境安装对应 PyTorch。自行准备 Qwen2.5-VL Target 与 ViSpec Draft 权重。

```python
from vispec.model.spec_model_ours import SpecModel
from bepc import generate

model = SpecModel.from_pretrained(
    base_model_path="/path/to/Qwen2.5-VL-7B-Instruct",
    spec_model_path="/path/to/ViSpec-Qwen2.5-VL-7B-Instruct",
    total_token=30, depth=3, top_k=8, num_q=2,
    torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
).eval()

# inputs 由对应 AutoProcessor 生成并放到模型设备上，
# 包含 input_ids、attention_mask、pixel_values、image_grid_thw。
output_ids = generate(model, inputs, max_new_tokens=512)
answer = model.tokenizer.decode(
    output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

生成入口支持单样本、T=0，通过原生模型模块安装控制器，并在请求结束后恢复函数。两类不匹配预算在默认参数中均为 `None`；内部以正无穷表示无上限，统计输出使用 `null`。

运行 CPU 单元测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=harness python -m unittest discover -s tests
```

ViSpec 源码来自 [KangJialiang/ViSpec](https://github.com/KangJialiang/ViSpec)，固定提交 `49a33ff0ca31e281b9dc8803fd6f3041c6c0e135`。许可见 `LICENSE`。
