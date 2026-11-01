# 长输出单次测试生成质量评价协议

协议日期：2026-11-01，Asia/Shanghai。本文件在单次测试正式评分前固定，沿用先前已封印的长输出解析规则；只修订测量次数和输出目录，不修改任何评分规则。主接口是`single_pass/harness/evaluate_long.py::score_task(task, samples, predictions, output_dir)`。

## 任务与解析边界

沿用同一批图像、问题、官方答案与20组开发/100组测试划分，按用户新指令每种设置只测1次。MME保留每组原始两问。已成功的开发repeat0直接导入，保留原始时间、原始token、原始出处和字节哈希；仅补缺失repeat0，不选择更优的repeat1/2。答案标签只在CPU评分时连接，不能加入生成输入。质量仅使用repeat0。单次测量不提供重复计时或主测确定性证据，图像cluster bootstrap置信区间不包含重复运行波动。

QA移除原短答案后缀，追加统一要求：`Provide a detailed explanation of about 100 to 160 words, using the image and the question to support your answer. Do not invent details that are not visible or supported by the question. End with exactly one final line in the format Final answer: <...>.`其中格式说明根据任务固定为yes/no、single word or short phrase或one option letter。COCO要求：`Please provide a detailed description of the given image in about 100 to 160 words. Describe visible objects, attributes, actions, quantities, and spatial relations, and distinguish uncertainty from visible facts.`词数是提示要求，不是样本有效性或质量筛选阈值；不强制最小生成长度。最大新token预算512，所有方法同一任务共享相同Prompt与停止条件，实际边界及允许的整块停止超出按运行协议记录。

QA长输出由解释段落和最后单独一行`Final answer: ...`构成。解释内容不参与答案搜索。研究扩展`terminal_final_answer_v1`仅执行以下格式操作：

1. 匹配独立行、大小写不敏感的`Final answer:`标记，并去除该行答案两端空白。
2. 必须只有一个这样的标记，且标记位于最后一个非空行。
3. 缺失、空标记、非终止标记为invalid；重复标记即使答案相同也保留uncertain。
4. MME仅接受`yes/no`，可有单个句末`.!?`；SQA仅接受一个选项字母，可带配对括号和单个句末`.!?`，之后沿用冻结的ScienceQA选项解析。
5. GQA和TextVQA提取内容不再移除引号、Markdown、标点、大小写或附带解释。TextVQA之后执行原官方归一化；GQA仍使用exact equality。
6. 格式失败、不确定、越界答案均保留在完整分母中，按0计。不存在gold辅助抽取、关键词猜测、随机回退或Judge抽取。

该格式解析是本研究为了长解释输出进行的必要扩展，不是原始基准论文定义。截断可能使末尾答案缺失，必须分别报告生成截断率和答案提取失败率，不能把格式失败全部解释为知识/视觉错误，也不能删除后重算主Accuracy。原始完整文本、提取结果、原因、官方解析状态均保存。

## 原样采用的官方规则

| 任务 | 原样采用的评价定义及实现 | 长输出必要扩展与局限 |
|---|---|---|
| MME | [作者评价代码](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)：每类`100*(acc+acc+)`，后者要求同图两问皆对，再按类求和 | 先提取末行yes/no；这是解释输出适配。保留14类分层及完整问答对；仍是子集成绩，满分2800，不能称完整leaderboard结果 |
| GQA | [官方评价](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)：预测字符串与原答案完全一致 | 只提取末行，不以语义相似替代exact。Accuracy不评价解释本身是否正确 |
| TextVQA | [原MMF评测器](https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py)：官方`EvalAIAnswerProcessor`及十答案leave-one-out VQA Accuracy | 提取答案后原样调用，不使用未经归一化的裸字符串匹配。解释中的OCR引用和错误不会单独扣分 |
| ScienceQA | [作者evaluate_acc.py](https://github.com/lupantech/ScienceQA/blob/main/tools/evaluate_acc.py)：原多选索引Accuracy | 固定末行字母解析；无随机猜测。不评价解释逻辑，不能宣称推理可靠性提升 |
| COCO CIDEr-D | [官方COCO工具](https://github.com/tylin/coco-caption)、[Python3移植](https://github.com/salaniz/pycocoevalcap)：原PTB tokenizer、TF-IDF、clipping和sigma6长度惩罚 | 对完整长描述原样评分；参考是短Caption，长度惩罚和参考覆盖不足使绝对分数不适合单独评价详细描述是否更好 |
| COCO SPICE | [作者实现](https://github.com/peteanderson80/SPICE)：原Stanford场景图tuple F1 | 整段长描述解析；参考事实不全、解析错误及跨句结构均有限。undefined保持null，不填0或删样本 |
| COCO CHAIR | [原论文](https://aclanthology.org/D18-1437/)、[作者代码](https://github.com/LisaAnne/Hallucination)：80类COCO对象及同义词的对象幻觉分析 | 完整长描述包含更多对象提及机会；CHAIRs随描述长度变化，CHAIRi及对象数量一起报告。不涵盖全部属性、关系、动作、绑定错误 |

COCO主视图始终是完整输出，不按得分选择首句、摘要或删除句子。CIDEr下降不等于事实伤害；任何自动指标都不称人工Ground Truth。此次未配置新的人类盲评或自动图像Judge，因此不能回答“哪种方法的人类整体描述质量更好”。QA主结论限于最终答案任务正确性，无法评价解释是否忠实或推理是否正确。

## 封印实现与写入隔离

只读复用`../harness/evaluate_tasks.py`，SHA256：

`1f491b0f1dba0a148b5a65ddc574c73c76878c098c967fc1f45fc08ba2a83fc3`

该原模块的源码和所有官方依赖不修改；动态载入后，仅将模块的输出根目录`ROOT`改为`long_output/single_pass/`，隔离其输出检查及SPICE锁。依赖常量已在原模块载入时解析为原只读路径。原封印`configs/frozen_config.json`（SHA256 `5c582799d4b0098ed8334fc5e80f753c73896783f406c93f07c3c6c57e47b188`）指定的标签文件逐个校验SHA，按question_id连接，图像SHA也必须一致；不更改任何原输入行。

原官方评测完整输出保存在每个方法目录的`official_scoring/`；长答案适配后的`per_question_scores.json`、`aggregate.json`和`answer_extraction.json`保存在该方法目录。适配签名包含完整原始输出，不能用相同末行答案覆盖不同解释。Java/官方依赖失败保存原错误并抛出，不以替代指标继续。单元测试使用合成文本或只读官方标签连接，不能将测试fixture计入实验结果。

## 单次适配的来源验证

父长输出评价器固定为`../harness/evaluate_long.py`，SHA256 `c5e17666135bc1d4263ca5e5d132418d8a721fd35d5c8b792557b691bf3d2072`。新入口的`extract_final_answer`函数与父实现逐AST相同，依赖原官方代码和标签配置的绝对身份不变。新入口写出目录、父源码校验和single-pass provenance不同；不是新的质量指标。10项合成质量解析/官方评价边界单元测试全部通过，选择器及单次MAT/配对统计自检通过。
