# BEPC 与 Margin：生成 token 变化及关键语义保护审计

结论：BEPC 确实改变了实际候选接受决策，并能在共同前缀下保留 Target token；已有局部属性精度改善和最终答案恢复案例。但“保留 Target”与“保护正确的关键事实”不能等同。本轮也确认了任务相关性被破坏、等价改写被误当数字纠错，以及后续续写新增幻觉的反例。目前证据支持机制可追溯，尚不支持 BEPC 稳定优于 Margin 的关键 token 保护结论。

## 1. 实验范围与冻结边界

- 对照来源：`../bepc_vs_margin_20261101/`，ScienceQA、COCO Caption、TextVQA 各100道冻结测试题，共300对/600条原始请求。
- 这是长输出设置：任务指令要求约100–160词，T=0，最大512新 token。两种方法均取消整题 mismatch 上限，保留每轮最多2个 mismatch。此报告不混入后续取消每轮上限的独立消融。
- BEPC=`bepc_no_response_cap`；Margin=`margin_no_response_cap`；阈值均固定为1，原始树、Prompt、KV、停止规则、原始质量评分、计时和全部历史输出不变。
- 读取全部300对进行CPU token对齐。对全部276对 token 有变化的样本，额外执行首次分叉前缀的原生树诊断；该轮选路后、KV提交前停止，没有重新生成完整答案或Caption。
- 没有训练、信号融合、修改模型/控制器、调整阈值、重抽样或更新冻结标签。模型辅助原图审查作为解释材料保存，不作为双人人工Ground Truth。

可追溯输入见 `audit/SOURCE_FREEZE.json`、`data/paired_records.json`。全部300对图像SHA、Prompt token、图像grid及输入张量SHA一致。600条输出的选中路径展平后，逐token等于原始生成结果。独立检查还核对了原实验14项输入冻结、29项交付封存、12项评分封存及262项历史保全文件，均未改变。

## 2. 如何区分“变化”与“保护”

原生最终层 margin 为 \(g=z_a-z_d\)，其中 \(a\) 是相同合法前缀的Target greedy token，\(d\) 是该前缀下的Draft候选。Margin接受条件为 \(g\le1\)。BEPC使用既有、冻结的修正 \(s=g-\delta\)，接受条件为 \(s\le1\)，且 \(|\delta|\le0.5\)。

本报告使用以下不同证据层次：

1. **输出token变化**：两条已保存输出的token ID不同。保存最长共同前缀和所有差异片段。
2. **机械 protection**：同一合法前缀、同一候选上，Margin会接受而BEPC实际拒绝。这个名称仅表示门控方向。
3. **机械 rescue**：Margin会拒绝而BEPC实际接受。
4. **直接保留Target**：首次可见输出差异处，BEPC确实拒绝Margin选中的Draft token，且BEPC输出该前缀下的Target token。
5. **语义保护**：还需证明被拒绝/改选的完整语义单元涉及错误事实、错误答案绑定或关键推理错误。Target top1本身可能错误。
6. **任务损害因果结论**：还需单事件干预、后续相同策略的反事实对照。这里比较两套完整策略的既有输出，首次分叉后的所有门控与前缀也会变化，不能把最终质量差归因于单个token。

`token_ledger.parquet` 的 `target_root/draft_node` 仅表示token来源，不代表该token是否mismatch。差异片段由确定性的 `difflib.SequenceMatcher(autojunk=False)` 给出，不称为最小编辑距离或干预次数。全部276对在首次分叉后又出现相同措辞；这些对齐的相同片段不再具有共同因果前缀。

## 3. 全部300对输出统计

| 任务 | 题数 | 生成文本变化 | 最终答案表面变化 | 官方评分改善 | 官方评分下降 | 官方评分不变 |
|---|---:|---:|---:|---:|---:|---:|
| ScienceQA | 100 | 87 | 4 | 2 | 1 | 97 |
| TextVQA | 100 | 91 | 19 | 6 | 5 | 89 |
| COCO | 100 | 98 | 不适用 | 依指标分别报告 | 依指标分别报告 | 依指标分别报告 |

共276/300题文本变化，24题不变；三任务所有输出均未触及512-token截断。ScienceQA Accuracy为85%→86%；TextVQA为73.1%→73.3%。这些是既有官方评分，不能把所有生成变化记为成功保护，也不构成显著优越性证明。

TextVQA的19例表面答案变化中，17例官方归一化答案变化；`Stop`→`STOP`、`YDR2108 YDR3110`→`YDR2108, YDR3110`不改变归一化结果。所有23个QA最终答案表面变化都先在解释正文发生token分叉，没有一个首次输出分叉直接发生在最终答案行。

完整数据含83,236个原始生成token、3,337个变化片段。可直接打开 `reports/TOKEN_DIFF_BROWSER.html`，按任务、题号、QA评分改善/下降筛选；原图缩略图、原始Prompt和双方完整输出均可查看。

## 4. 原生树诊断：首次token差异如何发生

对276例逐一保持封存原生Target/Draft forward及tree mask实现，重放相同输入到首次选路分叉树。核验全部前置共同路径、该轮BEPC历史路径和同一树上plain Margin shadow扫描所得历史路径完全相同。全部276例有效；没有丢弃失败样本，本次没有失败。

| 首次可见差异的机制关系 | ScienceQA | TextVQA | COCO | 合计 |
|---|---:|---:|---:|---:|
| 直接拒绝Draft并保留Target token | 40 | 49 | 37 | 126 |
| 直接拒绝Draft后改选另一个非Target token | 9 | 8 | 9 | 26 |
| 直接rescue并提交Draft，另一策略保留Target | 30 | 17 | 26 | 73 |
| 直接rescue，另一策略改选第三个token | 1 | 4 | 4 | 9 |
| 同位置protection/rescue混合 | 0 | 1 | 0 | 1 |
| 更深gate改变较早祖先分支选择 | 7 | 12 | 22 | 41 |
| 合计 | 87 | 91 | 98 | 276 |

126例说明BEPC具有实际保留Target token的机制效果；它不是“126例正确关键token被保护”。73例说明BEPC也会主动放宽，而非仅仅更保守。这些数字只覆盖首次选路/输出分叉，不是完整生成中的全部protection/rescue次数；24个输出不变样本未做门控诊断，不能据此断言其未选候选上的门控都相同。

41例尤其体现ViSpec的树结构：更深节点的门控会改变最长可接受路径或同长度的首次优先路径，因此最终输出可以在更早的祖先处变化。例如COCO358664可见首次变化为 `shows`→`depicts`，实际改变的门却比较更深处的 `two` 与 `a`；TextVQA39338可见首次变化为 `with`→`,`，实际改变的更深门比较 `with` 与 `including`。不能把较早显示的token说成直接被拒绝的对象。

原始附加读出包含311条候选决策差异，其中7条仅为shadow路径的补充假设计算，生产BEPC未访问，已从正式门控分类剔除。严格v2保留304条实际BEPC访问的差异（190 protection、114 rescue）。这些是首次分叉树内的候选记录，含多个祖先前缀和非最终选中分支，不是304个独立自然on-policy事件。首次可见差异分类不受7条剔除影响。正式引用 `diagnostics/allchanged_actual_visited_v2/`，保留原始v1用于审计。

## 5. 有意义的改善与必须同时保留的反例

### 5.1 关键属性的局部改善：COCO410428

共同正文为 `...a thick, white wool coat and a`。

- Margin接受 ` patch`，后续写 `a patch of gray fur on its back`；还在下一句写侧面有红色paint/brand。
- BEPC拒绝 ` patch`、保留 ` pink`，后续写 `a pink marking on its side`。
- 原图可见羊侧面的粉/红色标记。BEPC避免了较不准确的灰色背部斑块描述，属于局部属性精度改善；Margin本来也提到红色标记，因此不能说该标记在Margin中完全丢失。
- 实际门控：\(g=1.0,m_U=1.873,m_P=-2.089,\delta=-0.5,s=1.5\)。Margin接受，BEPC拒绝；同前缀下BEPC确实保留Target的 ` pink`。

这是目前可供人工复核的“有直接门控证据、且局部属性更准”的正例。它不证明整段Caption无其他错误，也不是单token续写干预的任务级因果证据。

### 5.2 OCR恢复来自rescue：TextVQA38417

最终答案 `Davisson's`→`Davison's`，官方分数0→0.3；模型原图检查可读到DAVISON'S。首次共同前缀为 `The image depicts a business named "`。

- Target候选为token `Davis`，Draft候选为token `D`。
- \(g=1.375,m_U=-3.072,m_P=1.233,\delta=0.5,s=0.875\)：Margin拒绝Draft `D`，BEPC接受，开始另一条subword续写路径。
- 这是rescue，不是“拦住错误姓名token”。`Davis`也可以继续组成正确的`Davison`；不能把一个子词当作完整错误姓名。完整姓名改善是观察到的后续结果，不能直接归因于拦住多出的`s`。

### 5.3 保护了错误任务焦点：TextVQA35696

问题明确问 `what are the letters on the jersey?`。共同前缀为 `The jersey features the`。

- Margin接受Draft ` letters`，最终答 `NC`，官方分数1.0。
- BEPC拒绝 ` letters`、保留Target ` number`，最终答 `6`，官方分数0.3。
- \(g=0.75,m_U=2.396,m_P=-1.024,\delta=-0.5,s=1.25\)。这是严格核验过的实际protection，但保护的是错误任务焦点。

它说明视觉子空间支持某个更显眼内容，并不保证该内容回答了问题。证据强度、视觉显著性与任务相关性必须区分。不能只根据`m_U>0`或“Target被保护”断言有益。

### 5.4 直接保留动词后，仍可能产生新动作幻觉：COCO30925

共同前缀为 `...The giraffe is`。Margin接受 ` leaning`，BEPC保留Target ` reaching`：\(g=0.9375,\delta=-0.4245,s=1.3620\)。

两种动词在图中都可合理理解；BEPC后续增加 `as it stands on its hind legs`。原图显示四脚落地，该后续断言是明确的姿态错误。门控确实保护了Target，但没有保证后续动作描述更可靠。这里只证明完整策略的后续错误，不能声称该动词单独造成错误。

### 5.5 数字token变化不是数字纠错：ScienceQA16731

首次数字token从Margin的`1`变为BEPC的`0`，完整公式分别为 `1/2 * m * v^2` 与 `0.5 * m * v^2`，数学等价。该门为 \(g=1,\delta=-0.1431,s=1.1431\)，机械保护真实，但语义上没有纠正公式。

以双方共同声明的质量32/36 u、速度1000 m/s为条件，后续Margin计算16,000,000与18,000,000，BEPC计算160,000与180,000，后者相差100倍；双方最终多选答案B都正确。本例补充审核没有独立核验图示数值，因此这里是共同文本条件下的算术检查。答案Accuracy不变掩盖了推理中的数字退化，说明task outcome与factual/reasoning错误应独立保存。

### 5.6 最终答案改善不能倒推关键实体被直接保护

ScienceQA13461从Africa/B错误恢复为Asia/C正确；但首次实际门控是 ` colored`→` in`，不是Africa→Asia。ScienceQA17228双方正文已正确叙述热量流向，BEPC仅最终选项与解释更一致。TextVQA38240双方正文均已写出`End of bus lane`，最终从`Bus`补全为`bus lane`。这些是有用的答案恢复，不能一概称为直接视觉识别保护。

## 6. QA全部答案变化与格式影响

ScienceQA共2例Wrong→Correct、1例Correct→Wrong、1例Wrong→Wrong；其余96例答案表面未变。TextVQA6例评分改善、5例下降、8例表面变化但评分不变，连续分数沿用官方VQA评测。

TextVQA三个格式相关案例必须单列：

| 题号 | Margin→BEPC | 官方得分变化 | 内容核查 |
|---|---|---:|---|
| 34741 | 缺少最终答案行→Olivetti | 0→0.9 | 双方正文已经包含Olivetti |
| 37671 | people→缺少规范最终答案行 | 1→0 | BEPC正文已有people |
| 38948 | Vader→缺少规范最终答案行 | 1→0 | BEPC正文已有Vader |

这三例净变化为−1.1原始分；其他TextVQA内容变化净+1.3，合计+0.2/100题。三个格式案例均正常EOS结束，并非512-token截断。不能只展示Olivetti恢复而隐藏两次答案交付失败，也不能将它称为OCR识别能力提高。

全部23例答案表面变化如下。更细的原始问题、解释、官方答案、解析状态和图像引用见 `reports/QA_CASE_REVIEW.md` 与结构化JSON。

| 任务/题号 | Margin→BEPC答案 | 官方分数 | 解释类别 |
|---|---|---|---|
| scienceqa/11022 | B → C | 0.0→0.0 | quantity_numeric_readout |
| scienceqa/11060 | B → A | 1.0→0.0 | entity_attribute_binding_and_reasoning |
| scienceqa/13461 | B → C | 0.0→1.0 | visual_entity_identification |
| scienceqa/17228 | A → B | 0.0→1.0 | final_option_reasoning_consistency |
| textvqa/34741 | [缺少规范答案行] → Olivetti | 0.0→0.9 | terminal_answer_format |
| textvqa/34760 | Mike → Lee | 0.0→0.0 | entity_answer_completeness |
| textvqa/34951 | Participants → Skylar and Jaw | 0.0→0.0 | entity_specificity_with_unsupported_addition |
| textvqa/35020 | Construction → None | 0.0→0.0 | ambiguous_question_abstention |
| textvqa/35085 | 1.5 → 1 | 1.0→0.0 | quantity_numeric_readout |
| textvqa/35696 | NC → 6 | 1.0→0.3 | answer_target_binding_letters_vs_number |
| textvqa/36619 | BARRA → Gracie Barra | 0.3→1.0 | named_entity_phrase_completion |
| textvqa/37229 | Stop → STOP | 1.0→1.0 | normalization_equivalent_case |
| textvqa/37315 | Phone → open | 0.0→1.0 | answer_target_binding_and_visual_text |
| textvqa/37577 | Thea Klivans → TheaKlivans | 0.0→0.0 | incorrect_name_and_whitespace |
| textvqa/37671 | people → [缺少规范答案行] | 1.0→0.0 | terminal_answer_format |
| textvqa/37816 | 12:05 → 12:15 | 0.0→0.0 | quantity_time_readout |
| textvqa/38240 | Bus → bus lane | 0.0→1.0 | answer_phrase_completeness |
| textvqa/38301 | 0171 → oscar | 0.0→1.0 | answer_target_binding_name_vs_identifier |
| textvqa/38417 | Davisson's → Davison's | 0.0→0.3 | named_entity_spelling |
| textvqa/38598 | Imagine → Banner | 0.0→0.0 | answer_target_binding_phrase_vs_object |
| textvqa/38686 | YDR2108 YDR3110 → YDR2108, YDR3110 | 1.0→1.0 | normalization_equivalent_punctuation |
| textvqa/38948 | Vader → [缺少规范答案行] | 1.0→0.0 | terminal_answer_format |
| textvqa/39338 | M → N/A | 1.0→0.0 | visual_letter_to_abstention |

## 7. COCO原图核查与指标冲突

全100图：CHAIRi改善9、退化17、不变74；CHAIRs改善7、退化8、不变85。既有总体指标为SPICE 12.66%→13.09%，但CHAIRs19%→20%、CHAIRi4.19%→5.37%。不能因SPICE均值上升宣称事实保护已被建立。

为核查具体事实，选取**全部26对CHAIRi不相等案例**，同时包含9个改善、17个退化，未按改善符号筛选。先随机A/B盲化，单个模型辅助审查者逐张查看原图后封存结果，最后unblind。结果为BEPC较好6、较差5、相当2、不确定13。该集合按指标变化富集，不能估计自然有害率；单个模型审查也不能充当人工Ground Truth或人际一致性验证。

局部改善包括：202913更准确表达蛋糕已有缺口及铲子在托盘上的位置；298699更好保持双建筑数量与Encore标牌绑定；358664避免无依据的grind-on-edge动作；578314补回图中左侧垃圾桶。相反，497438新增无依据的cat-ear headband及视线断言；28377丢失左右建筑灰/黄的颜色绑定；537574把可见项圈写成harness。后者虽有局部退化，双方又存在其他难比较的内容，所以整体保留uncertain。

指标冲突：6个模型审查较好案例中，4个CHAIRi反而升高；5个较差案例的CHAIRi均升高；2个tie中，一个CHAIRi升高、另一个降低；13个uncertain中，CHAIRi改善6、退化7。这说明对象参考覆盖、提及分母、属性/关系错误及信息覆盖之间存在差别，不能让CHAIRi替代总体事实判断。

额外4图的关键token审查与原26盲审单独保存：

- 410428：`patch`→`pink`是局部属性精度改善，非首次发现标记。
- 525373：`window`→`door`不能自动当纠错，原图同时有玻璃门与窗；BEPC后续还可能改变左侧儿童衣服颜色。
- 30925：保留`reaching`后增加四脚落地图像不支持的“后腿站立”。
- 71929：`small`→`fork`只是叉子更早出现，Margin后面同样提到fork，不能记为对象保护。

补充审查明确为unblinded、按token案例选择；它不加入26例盲审关系分布。见 `reports/ADDITIONAL_KEY_TOKEN_CASE_REVIEW.md`。

## 8. 阈值边界敏感性

304条实际访问的门控差异中，190条为protection，其中84条原生\(g\)恰等于阈值1；114条rescue的\(g\)均大于1。原生g来自BF16 logits，附加分量读出用原先FP32规则，并未重新定义数值精度。

例如TextVQA36619的实际门比较Target `patch` 与Draft `red`，\(g=1,m_U=0.000289,m_P=-0.1931\)，只有\(\delta=-0.0001445\)也能让得分由1变为1.0001445，从而拒绝；最终BEPC改选`white`而非该Target `patch`。这同时提醒：修正幅度不能直接当作语义重要程度，阈值边界与最长树路径可能放大小门控变化。这里仅作数值敏感性描述，未据此改阈值或重测，亦未证明这些差异都是BF16伪影。

## 9. 可复现性、成本和证据边界

- 全300对CPU对齐：相同输入、600条路径展平、原始/可见token、停止/截断、评分与解析状态均核验。语义候选字面串到原始token跨度见 `data/qa_critical_semantic_units.csv`：324个匹配位置、41个在当前规则下未命中的候选/分支组合。数字不匹配小数或千位数的前半部分，单字母区分大小写，标点按字面匹配，保存匹配规则和上下文。选项字母、物理样本代号及同一字符串的其他角色仍需结合上下文确认；该索引不保证每条字符串命中都是完整关键语义单元，更不构成自动语义标签。这些是解释性后验注释，绝不进入在线信号。
- 全276例原生树诊断：历史BEPC共同路径/分叉路径、同树Margin历史路径逐例精确一致。独立审核重建552次整树分支扫描；3473项候选读出、3107个实际gate缓存决策一致。
- 机制诊断总前缀运行时间284.16秒，不含三次模型加载；实际重做3426次Target、12600次Draft、276次视觉forward。门控附加分解本身无需再增加Target forward。不能把“附加读出0 forward”误说成本轮没有做模型推理；上述时间也不是新的方法吞吐benchmark。
- 全部诊断均在分叉轮KV提交前停止，不新生成完整Caption，不覆盖原计时。直接门控/路径差异可追溯；没有单事件后续任务损害因果实验。
- 原生KV/position/tree mask调用实现沿用封存代码，并由逐前缀和选路一致性作集成验证；本轮没有独立保存全部KV、mask、position、raw hidden和full logits张量，不能声称从独立原始张量重算了全部子空间或做了逐bit张量审计。分量数值复核是对序列化读出及冻结公式的检查。
- COCO原图审核为单模型解释性审查；QA额外推理错误也未替换官方Accuracy。双人人工可靠性仍待完成；不新增Harmful标签，不汇报新信号排名。

复现入口（使用原spec环境，避免Pillow版本改变输入RGB处理）：

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/data/wbb/miniconda3/envs/spec/bin/python source/analyze_token_changes.py
/data/wbb/miniconda3/envs/spec/bin/python source/map_critical_units.py
CUDA_VISIBLE_DEVICES=0 /data/wbb/miniconda3/envs/spec/bin/python source/prefix_tree_diagnostic.py --mode all
CUDA_VISIBLE_DEVICES=0 /data/wbb/miniconda3/envs/spec/bin/python source/prefix_tree_diagnostic_all_changed.py --mode changed
```

诊断脚本保留已有文件并跳过已完成条目；不会覆盖封存诊断。原始模型审查是已完成的人工可读产物，CPU脚本不会重新执行或伪造审查。首次全部诊断后的CPU汇总入口为 `source/summarize_prefix_diagnostics.py`，严格实际访问修正版为同目录v2入口。所有正式数字以 `diagnostics/allchanged_actual_visited_v2/` 为准。

## 10. 审稿人结论

**可以支持**：在本次相同配置300题的观测对照中，BEPC既有收紧也有救援，126例首次可见差异直接保留了共同前缀下Target token；原生树选择会把更深gate变化传导到较早输出；存在局部视觉属性改善与答案恢复案例，也存在明确的错误焦点、数值推理和动作幻觉反例。

**不能支持**：126例正确关键事实被保护；BEPC普遍知道哪些token关键；更高视觉支持就更符合问题；所有最终答案改善由首个protected token造成；官方答案分数相同就没有事实退化；自动模型审查已构成人工Ground Truth；BEPC已经稳定优于Margin。

因此当前结论是：**机械保护机制得到验证，稳定关键语义保护尚未得到验证。** 后续若检验后者，应在冻结完整语义单位和人工标准后，分别统计真正避免的错误、误拒的正确/有益候选、rescue收益和新增事实损害；在相同合法前缀上做单事件反事实，并保留树级选路影响。当前数据不能仅凭挑选正例构造优越性证明。

## 11. 交付索引

- `reports/TOKEN_DIFF_BROWSER.html`：全部300对的本地交互浏览器，包含原图缩略图、Prompt与完整差异。
- `data/token_changes.parquet` / `.csv`：一题一条的完整对齐和官方评分关系。
- `data/token_change_spans.parquet`：全部3,337个变化片段。
- `data/token_ledger.parquet`：83,236个原始token的选中轮次、树节点和来源。
- `data/qa_critical_semantic_units.csv`：QA语义候选到原始token跨度，非在线信号。
- `reports/QA_CASE_REVIEW.md`：全部23例QA答案变化解释。
- `blind_review/COCO_MODEL_REVIEW.jsonl`、`audit/COCO_UNBLINDED_REVIEW.json`：原26例盲态审查与解盲结果。
- `reports/ADDITIONAL_KEY_TOKEN_CASE_REVIEW.md`：补充关键token个案与等价改写/幻觉反例。
- `diagnostics/allchanged_actual_visited_v2/`：正式同前缀实际门控与首次可见差异分类。
- `audit/TOKEN_ALIGNMENT_INDEPENDENT_CHECK.md`、`audit/PREFIX_DIAGNOSTIC_INDEPENDENT_CHECK.md`：独立复核与局限。
