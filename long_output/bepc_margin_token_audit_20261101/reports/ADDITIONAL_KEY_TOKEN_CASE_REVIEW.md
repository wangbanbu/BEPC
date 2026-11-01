# 首次 token 差异的补充事实审核

**Unblinded · selectedfor_token_case · model-assisted · not Ground Truth。** 本文是原 26 对 A/B 盲审封存后的独立补充，不改原盲审文件、不将以下 4 个 COCO 个案或 1 个 ScienceQA 文本个案混入原 26 对的关系统计。选择是定向的，不能作为自然保护率估计。

逐张查看了 COCO 410428、525373、30925、71929 原图。ScienceQA 16731 按要求仅审查共同文本条件下的数学与后续解释，没有查看其图像。首次差异由保存的 returned_token_ids 逐一比较，再用冻结本地 Target tokenizer 解码；未重新生成、未调用外部 Judge、未改标签。

| 样本 | 首次差异（Margin → BEPC） | 可支持的解释 | 不能支持的解释 |
|---|---|---|---|
| coco 410428 | `patch` → `pink`，位置 32（0-based） | 局部属性/位置描述改善，但不是首次引入侧面标记，也不能据整句反推单个拒绝门因果作用。 | 不可说 Margin 完全漏掉标记、BEPC 新保护了标记对象，或把整段后果归因于 patch/pink 单 token。 |
| coco 525373 | `window` → `door`，位置 22（0-based） | 两种背景实体都可见；首次对象词差异不构成可确认的关键对象保护。 | 不可把 window→door 列为去除幻觉或重要对象保护成功。 |
| coco 30925 | `leaning` → `reaching`，位置 23（0-based） | 首个动词本身不是明确纠错；BEPC 后续新增四脚着地画面不支持的后腿站立动作。 | 不可把 reaching 视为关键动作被保护，不能声称 BEPC 的长颈鹿姿态更准确。 |
| coco 71929 | `small` → `fork`，位置 66（0-based） | 同一真实对象在 BEPC 更早出现；Margin 后文也明确有 fork，不是新增对象保护。 | 不可把 small→fork 直接记成 Margin 丢失餐叉、BEPC 保护餐叉的成功。 |
| scienceqa 16731 | `1` → `0`，位置 24（0-based） | 首个数词变化为数学等价改写；BEPC 后续计算结果小了 100 倍，官方最终答案准确率未捕捉此推理错误。 | 不可把 1→0 的 token 标签当成数字保护，也不可用最终答对排除新增推理错误。 |

## coco 410428

BEPC 的 pink marking on its side 与左侧成年羊可见粉红喷涂标记一致；Margin 的 patch of gray fur on its back 不准确，背上明显色块偏绿/青色，不能可靠称为灰毛补丁。

Margin 随后一整句已说 It has a distinctive marking on its side, possibly a red paint or brand，故两方法都描述了侧面识别标记，只是 BEPC 更早且颜色更准确。

允许的结论：可作为局部颜色/位置描述更贴近图像的探索案例。

不支持的结论：不可说 Margin 完全漏掉标记、BEPC 新保护了标记对象，或把整段后果归因于 patch/pink 单 token。

不确定性：喷涂用途只是可能的识别/品牌过程，不能由图像确定。 第一处局部表述改善不等于整个 Caption 已通过双人人工评估。

共同前缀结尾：` left is a larger adult, with a thick, white wool coat and a`

Margin 后续片段：` patch of gray fur on its back. It has a distinctive`

BEPC 后续片段：` pink marking on its side, possibly from a branding or identification`

## coco 525373

原图左侧明确有带把手的玻璃门，门上方遮光帘；孩子后方和右侧也有窗。Margin 的 window 和 BEPC 的 door 都可由图像支持。

BEPC 后文同样说 window with blinds，故不是用门纠正不存在的窗。BEPC 把左童穿着从 Margin 的 red outfit with a frilly collar 改成 also in a pink shirt；图中左童衣服为明显红/深红色、白边波浪领，Margin 的颜色/领口更准确。

允许的结论：可以说两方法选择了不同但存在的背景对象；BEPC 另有衣色误差风险。

不支持的结论：不可把 window→door 列为去除幻觉或重要对象保护成功。

不确定性：图像有偏暖色调，但两儿童衣色明显不同，不能不加说明地都写成同类粉色。 室外露台/阳台用途仍是推断；门外确可见栏杆与明亮室外。

共同前缀结尾：` indoors, likely in a home setting, with a neutral-colored wall and a`

Margin 后续片段：` window in the background. The child on the left is wearing`

BEPC 后续片段：` door in the background. The child on the right is wearing`

## coco 30925

长颈鹿向树伸颈，leaning its neck towards 和 reaching up to 都是可接受的概括，首个动词无充分证据判定一方有害。

BEPC 增加 as it stands on its hind legs。图像中前后腿均落地，未以后腿单独直立。该动作/姿态断言与图像冲突。Margin 未作此断言。

允许的结论：可作为 BEPC 轨迹新增动作幻觉的反例，说明不能只审第一个替换词。

不支持的结论：不可把 reaching 视为关键动作被保护，不能声称 BEPC 的长颈鹿姿态更准确。

不确定性：单张图不能证明正在吃叶子；两方都以 possibly 表述意图，须保留该推断层级。 没有单事件回放时，无法确定后腿句由首个动词干预单独引起。

共同前缀结尾：` a grassy area with a pond in the foreground. The giraffe is`

Margin 后续片段：` leaning its neck towards a tree, possibly to reach for leaves`

BEPC 后续片段：` reaching up to a tree, possibly to eat leaves, as`

## coco 71929

原图右上盘沿有餐叉，BEPC 的 fork resting on the plate 支持。Margin 在同一位置先描述另一份红色熟食/酱汁，但随后明确说 A fork is placed on the plate。

两者都保留叉子概念。Margin 较具体的 resting on the rice and the sauce 与餐叉位于盘右上、叉齿邻近米饭的画面只部分吻合；BEPC 的 on the plate 更宽泛。更简短不自动等于补回关键事实。

允许的结论：可以报告信息出现位置、具体程度和输出长度变化。

不支持的结论：不可把 small→fork 直接记成 Margin 丢失餐叉、BEPC 保护餐叉的成功。

不确定性：食物具体种类、味道/辣度、靠近水体或海边餐厅都不能由当前画面完全确认。

共同前缀结尾：` are also a few pieces of green vegetables, possibly pickled, and a`

Margin 后续片段：` small portion of what looks like a cooked, red, and`

BEPC 后续片段：` fork resting on the plate. To the left of the plate`

## scienceqa 16731

1/2 = 0.5，均是 KE = (1/2) m v^2 的同一系数，不能把 token ID 从 1 变 0 当作数值事实纠错。

仅依据两输出共同声明 m_A=32 u、m_B=36 u、v=1000 m/s，正确结果为 16,000,000 和 18,000,000 u·m²/s²。Margin 写出这两个数；BEPC 写 160,000 和 180,000，分别是正确值的 1/100。两者最后 Final answer: B 均被官方计为正确。

允许的结论：可作为答案 Outcome tie 但解释/算术 factual regression 的例子；量纲未换算成 SI，本审查只比较共同单位下算术。

不支持的结论：不可把 1→0 的 token 标签当成数字保护，也不可用最终答对排除新增推理错误。

不确定性：按任务要求未查看此例图片，未独立确认图示质量/速度；算术判断以文本共同给定的量为条件。 后续算术错误不能在无单事件干预的情况下唯一归因于最初等价系数改写。

共同前缀结尾：` in each sample, we use the formula for kinetic energy: KE = `

Margin 后续片段：`1/2 * m * v^2, where m`

BEPC 后续片段：`0.5 * m * v^2, where m`

## 对关键 token 保护的证据边界

一个生成 token 带数字、颜色、对象或动作标签，不能自动成为保护成功。需要先恢复完整语义单元，再核对原图，检查另一方法是否在稍后同样表达该信息，并检查新轨迹有无新增错误。这里存在局部颜色描述改善，也存在等价改写、提前表达和后续动作/算术退化。保留这些反例，避免只截取首个有利 token。

已保存的控制器/树路径日志可以支持“决策或路径在哪一轮发生差异”；完整 Caption 对只能支持轨迹层面的结果差异。若要证明某个拒绝决策保护了某事实，仍需在相同前缀固定其他设置的一次干预续写及其可靠审核。

结构化附件：`audit/ADDITIONAL_KEY_TOKEN_CASE_REVIEW.json`。原输入 SHA256：`71439def16542ef8f6c28c5ed392402efd690e77361e6b111ec3cad7bc0e20aa`。本次独立 JSON SHA256：`7505f45924b9565fc31cd07bec4673b9f2ac52741c5e5dcbd21194caa31a66dd`。
