# COCO开发集指标独立复核

结论：未发现图像—参考描述错位、错误split、双重缩放或输出截断造成的评测bug。原生Strict ViSpec开发集CIDEr-D为 **0.3732156211768144**，独立执行作者原始scorer得到相同均值；20条逐图最大绝对误差为 **0.000e+00**。本检查仅访问已完成的20图开发集预测和开发参考，不访问主测试质量；不改源码、冻结输入、指标结果或阈值。

## 证据范围与身份关联

- 读取`results/development/development_coco_shard{0,1}.jsonl`与完成标记，校验结果SHA；只选`method=vispec, repeat=0`作为质量输入。60条Strict开发请求用于停止规则和三次重复核验。
- 20条预测以真实COCO image_id重新建字典；独立读取官方`captions_val2014.json`，按每条参考的原始annotation sentid核对image_id与文字。20图各5条、共100条参考全部精确匹配，官方文件名与真实COCO ID一致。
- `score_task`先按question_id对齐预测，再唯一识别dev_labels文件并核对image SHA；内部0..19序号同时用于对应预测与参考。独立复算使用真实COCO ID，结果完全一致，排除仅由内部序号重排造成的错配。
- 独立调用Stanford原PTB类，经stdin输入100条官方参考和20条冻结预测；全部120条分词结果与已保存PTB输入逐条一致。未向原依赖目录或既有metric目录写临时文件。

## 原始实现与数值复算

| 项目 | 结果 |
|---|---:|
| 已保存Strict CIDEr-D | 0.3732156211768144 |
| 作者原始scorer独立调用 | 0.3732156211768144 |
| 生产依赖独立直接调用 | 0.3732156211768144 |
| 逐图最大绝对误差 | 0.000e+00 |
| DF图像数 / 每图参考数 | 20 / 5 |
| 1–4gram阶数 / Gaussian sigma | 4 / 6.0 |
| PTB预测词数均值 / 参考词数均值 | 20.45 / 10.56 |
| 100个预测—参考对的平均既定长度惩罚因子 | 0.342335 |
| Strict repeat0达到128-token截断的图像 | 0/20 |
| 三次Strict可见token与文字完全一致 | 20/20 |

调用[作者原始CIDEr scorer](https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/cider/cider_scorer.py)，其原始Python2源码仅在内存中把`xrange`换成`range`、`.iteritems()`换成`.items()`，计算语义、默认n和sigma均保持原样。本地作者源码SHA256为`13e222f84e82d74b606d6afdb97d225bebddc6d1c7ee063f196d6d1ed931c6b2`；本次从作者raw URL读取的内容SHA与之相同。原始文件、生产依赖和已保存结果均未修改。

生产实现由[官方Cider入口](https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/cider/cider.py)传入n=4、sigma=6；scorer自身把四阶相似度均值乘10，并在图像间取均值。`score_task`直接保留原生结果，报告层若写×100则为37.32156211768144；未在计算中额外除100或重复乘100。该官方coco-caption入口虽然类名是Cider，内部含clip与Gaussian长度惩罚，审计使用CIDEr-D名称说明其计算。

DF是当前完整20图开发参考的图像级频数：每张图的5条参考先合并该图出现的ngram，再累计图像数；不是把100条参考当100个document。`ref_len=log(20)`，最大document_frequency不超过20。正式100图测试会重新以其100图参考计算DF，开发与测试的绝对CIDEr值不应直接横比；本次不计算或查看测试值。

低绝对值与模型较长、含额外细节的Caption相容：预测词数均值高于参考，既定sigma6的长度项会明显抑制部分样例。小DF会改变IDF权重，但本检查没有改变DF语料或Prompt，因此**不能把低分因果归结于小DF或Prompt**。未为提高分数调节长度、分词、参考、sigma或样本。

## 固定5例图像与五参考复核

按冻结开发顺序预先取ordinal0、1、2、4、6，覆盖五张不同图像；下列观察仅为身份核验，不产生人工事实Ground Truth或新Harmful标签。

### ordinal0 / COCO 109434

- 图像：`data/assets/coco/109434.jpg`；image SHA256 `c0cf6896dbbf66ca535a6262cb34808b9927cabee044506545993be2129cd488`。
- Strict：A modern bathroom with a green accent wall, a double sink vanity, and a bathtub with a shower, featuring decorative elements like a leaf mirror and plants.
- CIDEr-D：0.030835974439。
- 图像复核：图像确为绿色墙面浴室、单洗手池和浴缸；Caption出现double sink，与参考和图像有差别，身份对应正确。
- 官方五参考：
  1. A bathroom with a large mirror above a white sink.
  2. A birdcage is near the sink in the bathroom.
  3. A three piece bathroom with a single vanity.
  4. A bathroom with a green wall and a sink encased in dark wood cabinets.
  5. a bathroom with a sink and a bath tub in it.

### ordinal1 / COCO 556545

- 图像：`data/assets/coco/556545.jpg`；image SHA256 `ad7babbaed7723e47185c81cf1aec45619719ef233456c2c4b7c2e0d80afefe6`。
- Strict：A skateboard leans against a wall next to a framed photograph, set against a red-toned room with a wooden door.
- CIDEr-D：0.525904780193。
- 图像复核：图像是滑板旁的相框，红色地面/木门背景；Caption与该图及参考对应。
- 官方五参考：
  1. A skateboard resting against a wall with a photo sitting next to it. 
  2. a framed picture leaning against the wall next to a skateboard.
  3. A skateboard and portrait photograph of a young man, on the floor.
  4. a skateboard and a framed portrait leaning against a wall
  5. A photo and a skateboard sitting on the floor.

### ordinal2 / COCO 214184

- 图像：`data/assets/coco/214184.jpg`；image SHA256 `523a3d7f0536b1cca85fc110e43c6dde1e1afa9d6ed9f04828498626ae5c9c37`。
- Strict：A black cat lounges on a shelf filled with books and vintage cameras.
- CIDEr-D：1.195829917626。
- 图像复核：黑猫躺在书上，上一层有旧相机；Caption与该图对应，参考主要描述猫和书。
- 官方五参考：
  1. The black cat is laying on top of books on the shelf. 
  2. The cat is lying on top of the books on the shelf.
  3. A cat is lying on top of books on the bookshelf.
  4. A cat lays down on a row of books
  5. A black cat laying on top of a set of books.

### ordinal4 / COCO 156282

- 图像：`data/assets/coco/156282.jpg`；image SHA256 `b234e64e7775834be4660f25d605f3cc7da38695d64ed0152b01015619aa2fe6`。
- Strict：A person is flying a colorful kite on a sunny beach with a clear blue sky and a hill in the background.
- CIDEr-D：0.273495984939。
- 图像复核：沙滩人物放风筝、蓝天与山坡；Caption与参考对应，含更多背景细节。
- 官方五参考：
  1. A woman standing in a field flying a kite.
  2. The woman is flying a kite at the beach.
  3. A person on a beach holding a kite string and a kite is in the air.
  4. A person is flying a kite on a beach.
  5. A woman flys a kite against a blue clear sky.

### ordinal6 / COCO 528905

- 图像：`data/assets/coco/528905.jpg`；image SHA256 `59f5905212064cd2452ec101d27e780a96ed8faf57b3633e83d7764e28c2a930`。
- Strict：A person is mid-air throwing a frisbee while two others run on a grassy field, with a car and trees in the background.
- CIDEr-D：0.142167028415。
- 图像复核：三人在草地玩飞盘，有汽车和树；Caption的throwing与参考catching存在动作措辞差异，未见跨图错配。
- 官方五参考：
  1. A GROUP OF PEOPLE PLAYING FRISBEE IN A GRASSY FIELD
  2. Two players watch as a team member catches the frisbee mid-air.
  3. a man jumping in the air catching a frisbee in a park
  4. People playing a game in a wood gated field.
  5. Players are on a grassy field and one is catching a frisbee.

## 停止、截断与重复检查

对60条Strict开发请求从raw_token_ids重新应用128-token预算与tokenizer的EOS=151645，得到的首EOS位置、returned_token_ids和decode后文字全部与记录一致。评分使用首EOS前的可见Caption；树形一步可能提交的EOS后尾部token没有进入评分。20条repeat0均未触及128-token上限，因此本开发低分不是预算截断的结果。三次Strict结果逐图一致；没有重新生成Caption。

## 输入与实现哈希

| 证据 | SHA256 |
|---|---|
| `data/coco/dev.json` | `59208115e991c6b2eb90fea923b14bf0be116fac5c39266ebbbbdc4184384ebf` |
| `data/coco/dev_labels.json` | `546eb8612e45dfba6feca6f674f7f6dc8c8adb79a8a5667e7fdd9e67acd992d9` |
| `harness/evaluate_tasks.py` | `1f491b0f1dba0a148b5a65ddc574c73c76878c098c967fc1f45fc08ba2a83fc3` |
| `harness/run_benchmark.py` | `439be31ad88f0d46e07bbad17dce432024e7ef1069d9e4e2b4f69c7acc5ffcb0` |
| `metrics/development/coco/vispec/aggregate.json` | `c86bd888dab28dcfd2aa7963b7097dacfb35f2f01773cd63830a18c5bcda8b7c` |
| `metrics/development/coco/vispec/per_question_scores.json` | `113eca497a965b71bae8eaa130633a2d0270231eabfdf9757d1f72d13c4d560f` |
| `results/development/development_coco_shard0.jsonl` | `6a267962882fbb979b14f8e46e81b83554882ab84bb99a1065368f465d11ff59` |
| `results/development/development_coco_shard1.jsonl` | `910c762e90a07eace6f078c2b03462a38948721a86c18fc2226fd5c14dcee994` |
| 官方captions_val2014.json | `7da4cc45cb76ee1642d163e846e1e0a33feffac2e841c9c426d585ea499165a4` |
| 原Stanford PTB jar | `2fcb91bb7a111f93d71e264f4ee0e3afd19ba0dde6d21b38605088df9e940399` |

复核完成后，上述冻结输入、源码、原始开发结果和既有评分文件SHA全部保持不变。该检查验证当前开发指标的身份与实现一致性，不证明Caption人工事实安全，也不提供任何主测试结论。
