# 官方任务评价实现审计

审计日期：2026-10-31。实现：`harness/evaluate_tasks.py`；测试：`tests/test_task_metrics.py`。本文件说明评测规则与适配，不包含尚未完成的新基准结果。评测不调用模型、VLM Judge或学习式答案提取器。

## 统一接口与数据边界

```python
per_question, aggregate = score_task(task, samples, predictions, output_dir)
```

一次调用仅包含一个方法、一个split及一个确定性quality repeat，建议repeat0。`samples`使用准备器的runtime records，`question_id`为唯一字符串；返回的`sample_id`与之相同。MME保留两条独立问题及同一`group_id`、`category`。`predictions`可为ID到文本的mapping、与samples对齐的文本列表，或含`question_id/output_text`的记录；重复、遗漏、多余ID都会失败，不能静默剔除问题或把3次测速当3个质量样本。

模型输入与标签分开保存。若samples没有标签，评测器仅在CPU评分时，从`data/<task>/dev_labels.json`或`test_labels.json`匹配唯一split；`coco_caption`别名映射到COCO scorer。标签图像SHA必须与runtime record一致。ScienceQA的`answer_index/choices`、TextVQA的十个`answers`、COCO的`sentences`仅在此时加入评测副本，原runtime records不修改；输出保存标签文件SHA。

每个问题保存原文本、解析结果、`score`、`parse_status`、`parse_invalid`和`uncertain`。二者并非同义词：例如ScienceQA两个矛盾显式答案标为uncertain；未识别答案标为invalid。二者都按0计入任务Accuracy分母。COCO的主`score`是CIDEr，其他指标独立保存。任何原始Java/数据错误会保存`metric_failure.json`后抛出，不替换评价算法。

所有输出与新缓存限定在本次新项目内。旧评价代码、模型、数据和历史结果均只读；没有复用历史Caption分数。`per_question_scores.json`、`aggregate.json`及`metric_inputs.sha256`保证当前调用可追溯。

## 原始规则、采用方式与局限

| 任务/指标 | 原始定义/实现 | 本轮采用方式 | 必要适配和边界 |
|---|---|---|---|
| MME | 每类`100*(acc+acc+)`，acc+要求同图两问均对；[作者项目](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)、[官方eval_tool.zip](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/tools/eval_tool.zip) | 原样执行作者`parse_pred_ans`；acc/acc+使用相同数学公式 | 对100/20个图像组子集按真实group_id配对，不依赖文件相邻行；不是完整MME leaderboard成绩 |
| GQA | [官方评价](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)与原`eval.py`使用`predicted == gold` | 原样采用exact answer Accuracy | 新数据准备器使用`val_balanced`，不是test-dev或challenge；不另行lowercase、去标点或语义抽取 |
| TextVQA | [官方MMF](https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py)的`EvalAIAnswerProcessor`及`TextVQAAccuracyEvaluator` | 直接执行原实现；逐题结果与其`eval_pred_list`交叉核验 | 原始10个人工答案全部保留，不能用裸字符串匹配或简单`min(count/3,1)`代替 |
| ScienceQA | [作者evaluate_acc.py](https://github.com/lupantech/ScienceQA/blob/main/tools/evaluate_acc.py)比较原始答案索引，计多选Accuracy | 原样采用选项正确性定义 | 官方完整driver硬编码4241题；本轮是图像题子集driver。固定文本到选项解析属于本研究适配，不归因于原论文 |
| CIDEr-D | [CVPR2015论文](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)、[官方COCO工具](https://github.com/tylin/coco-caption)、[Python3移植](https://github.com/salaniz/pycocoevalcap) | 已核验的pycocoevalcap `Cider`类；原PTB tokenizer、clipping和sigma=6长度惩罚 | 当前split完整参考集合决定DF；dev20和test100分别计算，绝对分数不能跨两者比较 |
| SPICE | [ECCV2016论文](https://arxiv.org/abs/1607.08822)、[作者实现](https://github.com/peteanderson80/SPICE) | 原jar与Stanford CoreNLP3.6.0模型，原语义tuple F1 | Java内存/线程/写入目录属于执行配置；不改匹配算法。undefined保留null，不能填0或删题 |
| CHAIR | [EMNLP2018论文](https://aclanthology.org/D18-1437/)、[作者代码](https://github.com/LisaAnne/Hallucination) | 原作者对象/同义词映射，通过已核验Python3适配器；逐句和聚合结果与作者`compute_chair`核验 | 范围是80个COCO对象类别及同义词，不是属性/动作/关系/绑定或所有事实错误的Ground Truth |

## MME：成对问题与子集报告

`acc`是问题正确比例；`acc+`是图像组中两问均正确的比例。对每类分别计算并相加，不能把整套问题的一个全局Accuracy当MME官方总分。

实现返回：

- `MME_subset_score`：本次出现类别的分数之和，主指标；
- `Perception_subset_score`、`Cognition_subset_score`；
- 每类acc、acc+、score、问题数、组数和other数；
- 逐组acc/acc+及每问`group_pair_correct`，便于保留两问进行统计；
- 全局问题Accuracy与全局acc+，仅作辅助描述。

14类均出现时，perception上界2000、cognition上界800、合计2800；本轮子集即使覆盖14类，分数也不是完整官方数据成绩，更不能称为“完整3000分评测”。缺失类别明确列出，不补0掩盖样本缺失。

原解析器将输出小写后，先接受精确yes/no，否则只检查前4字符中是否有yes或no；例如`nobody`按原代码会被解析为no。这是原工具局限，不能为了改善结果修改parser。生产输出已由冻结生成策略去首尾空白；评测层不再修改字符串。

数据准备器`choose_mme`按RGB去重所有已选组，跨category和dev/test避免同一底图重复使用。推断时仍应核对最终data manifest。统计建议在category内按image/group重采样，保留每组两问，并固定14类加总规则；直接重采样单题会破坏acc+，忽略分层会改变类别权重。

本地环境没有sklearn。实现从原`calculation.py`的AST执行完整原解析类，调用其未修改的parser；acc/acc+按原公式直接计算，不导入未使用的precision/confusion-matrix依赖。这是依赖适配，不是新MME指标。

## TextVQA：十答案归一化与连续得分

预测和全部10个答案都通过原`EvalAIAnswerProcessor`。对每个留出的人工答案，统计其余9个答案的匹配数，取`min(matches/3,1)`，最后平均10次。只有1、2、3、至少4个匹配答案时，通常对应0.3、0.6、0.9、1.0；这与不留一的简化公式不同。

保留连续逐题得分。空答案按官方归一化后空串处理，仍留在分母；不把模型长句中某个单词、OCR片段或最终结论用LLM提取出来。短答案Prompt是本轮适配，原ViSpec解释型Prompt与这类评测不完全匹配。

## ScienceQA：固定解析扩展

仅评价原始图像题。选择集合及原答案索引来自官方标签；题目hint和choices进入原始Prompt，GT lecture、solution、caption不用于答案生成。

解析顺序在测试前固定：

1. 显式`answer is/:/=`的选项字母；多个互相矛盾的显式答案或`A or B`为uncertain。
2. 单独字母，可带括号及结尾标点。
3. 句首有明确分隔符的选项标签，如`(B) ...`或`B. ...`；同时输出多行不同选项为uncertain。
4. 越界字母、只有选项文本、没有明确选项均invalid；不随机猜测，不用GT答案帮助解析。

Accuracy分母保留全部问题，invalid与uncertain均0。答案解析是必要研究扩展；不能把格式失败全部解释为模型视觉知识错误，也不能用删除格式失败样本提高Accuracy。主Prompt要求只输出选项字母，因此上述宽格式是固定兼容规则。

## COCO：共享参考集合及只读依赖

每种方法必须提交当前split全部图像的唯一Caption，缺失ID失败。COCO参考由本次单独标签文件读取；每次Cider对象用完整当前split建立DF，不从旧运行导入DF或CIDEr结果。不同方法在同一split使用相同参考；20图开发与100图测试的绝对CIDEr不可直接比较。

原PTB tokenizer会在自身目录写临时文件，因此本轮仅复制其小Python wrapper到新的输出目录，并以symlink引用原jar；临时文件不会落入旧依赖目录。SPICE缓存、输入、日志、输出及文件锁都在本项目；相同当前输入结果可以按SHA复用，不导入旧Caption评分memo。SPICE任何undefined逐项保留null，aggregate同样undefined。

CHAIR读取原captions/instances两类COCO标注和作者同义词。复制的作者源与pattern singularization材料仅作为新输出内评价依赖，不涉及新模型训练。逐Caption CHAIRi沿用作者无对象时0的规则；如果整个集合没有任何对象提及，聚合CHAIRi分母为0，保存null，不能把它称为完美事实精确性。

CIDEr衡量参考共识相似性，SPICE依赖句法/语义解析，CHAIR覆盖固定对象词汇。三者独立报告，不投票生成Harmful或人工事实标签，也不证明事实错误率/显著遗漏率非劣效。

## 本轮原始实现核验

读取本地原代码，并在2026-10-31对公开原始来源核验：MMF当前raw文件与本地文件逐字节一致；MME作者发布的eval_tool.zip内calculation.py与本地文件逐字节一致。MME官方zip SHA256为`b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797d3d8acee`。GQA完整evaluation archive很大，本轮核对已有作者脚本的原始Accuracy代码与Stanford官方定义，没有重新下载整个archive。

| 实现 | SHA256 |
|---|---|
| MME calculation.py | `66f154c0d3568cdd0c95b7d3e6b5a4b10cd393c71463328352c8c417c20a0607` |
| MMF m4c_evaluators.py | `e10a99e15c4658f8d0d83a05cc536c835922e701171fce0016718bec021b8c0a` |
| GQA作者eval.py | `787ca40d8c06d09e4e4a053259f84a4ca3ca431a6fdd8d6fb3003109f013fbdf` |
| ScienceQA evaluate_acc.py | `b33525c2387646c03a8f6f690ea6fbd7a05bcc7457919c6511a478d10fea4d25` |
| CIDEr scorer | `299037838373391c3335e066c867ba2d6dd8617d2c8fb2c4193f32620a774adc` |
| SPICE jar | `6c6a73ae6dd108667047c1818707590413934dcb9f1a62cef6469059b41b3d59` |
| 作者CHAIR source | `b05f701ce6ec0125dd700ffb80ed58bfcf2686df8c9bd62f70169cb928de4627` |

## 测试与尚待完成的运行验证

CPU测试9项通过：GQA严格匹配、TextVQA四级软分数和官方归一化、十答案要求、ScienceQA无随机回退/矛盾/越界处理、MME组配对acc+、预测ID完整性、分离标签join、COCO官方PTB/CIDEr及undefined保留。最后一项仅为合成fixture的管线测试，SPICE/CHAIR结果被显式mock，不能当作真实基准质量结果。

本轮没有启动GPU、Judge或正式任务评分。真实SPICE/CHAIR集成应在root协调CPU/Java资源后、生成结果完成时运行；失败会保留并中断评分。方法的速度、接受长度与显存由runner统计，本评测层不读这些变量决定任务分数。所有指标的代码和配置应在实际测试前封印，后续不得因结果好坏调整答案parser或评价定义。
