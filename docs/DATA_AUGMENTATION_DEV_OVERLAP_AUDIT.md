# 人工增强与Dev文本重合核对

核对日期：2026-10-27。读取项目中现存的实际训练CSV，只做文本统计，没有重新生成增强、修改标签或训练模型。

## 核对规则

1. 文本先做 Unicode NFKC、casefold 和连续空白合并。
2. “相同词”按 `[^\W_]+` 切分字母／数字串，不删除冠词、介词等功能词；这不是模型的子词分词。
3. “词干”用 NLTK PorterStemmer，检查词形变化后的重合；它不是完整词义还原。
4. “完整短语”比较单侧全文；“无序文本对”把两侧排序后比较，因此同时检查正向和反向重复。
5. 词、词干、短语和文本对均计去重后的交集种类数，不是匹配行数。该规则与 `prepare_manual_data.py` 的历史审计一致。
6. 对人工新增／复制批次，先确认训练文件的前3042行与官方Train逐项一致，再仅统计该批新增的末尾行；原Train、旧scope和新批次不混在一起算。
7. 本次Dev只使用instance_id、text1、text2，不用Dev标签接受或拒绝样本。

## 汇总

<!-- audit-summary:start -->
| 核对部分 | 行数 | 相同词种类 | 相同Porter词干 | 相同完整短语 | 相同无序文本对 |
|---|---:|---:|---:|---:|---:|
| 官方Train | 3042 | 509 | 516 | 155 | 7 |
| scope200 | 200 | 23 | 24 | 2 | 0 |
| relation48 | 48 | 0 | 0 | 0 | 0 |
| 人工80 | 80 | 0 | 0 | 0 | 0 |
| 事件200 | 200 | 0 | 0 | 0 | 0 |
| 边界120 | 120 | 0 | 0 | 0 | 0 |
| 局部128 | 128 | 0 | 0 | 0 | 0 |
| semantic64 | 64 | 0 | 0 | 0 | 0 |
| 家族768 | 768 | 0 | 0 | 0 | 0 |
| E改写88 | 88 | 58 | 66 | 9 | 0 |
| 复制200 | 200 | 166 | 170 | 23 | 0 |
| 复制80 | 80 | 96 | 98 | 7 | 0 |
| 复制120 | 120 | 129 | 136 | 12 | 0 |
| 复制128 | 128 | 137 | 144 | 15 | 0 |
| 复制768 | 768 | 296 | 305 | 63 | 0 |
<!-- audit-summary:end -->

## 如何理解结果

- relation48、人工80、事件200、边界120、局部128、semantic64、家族768：新增训练行在以上四种检查中均为0。
- scope200：无重复文本对，但共享23种词、24种词干和2条完整短语，不能称为词汇零重合。
- E改写88：无重复文本对，但共享词、词干和整短语，不能套用后期人工批次的严格隔离结论。
- 官方Train本身与Dev有7个规范化无序文本对重合。各训练版本保留了官方Train，所以新增批次零交集不等于完整训练集零交集。
- copy直接复制官方Train，词汇与Dev共用是预期现象；本次检查的复制批次均没有重复的Dev整对。
- 以上都是文本重合检查。增强类型曾参考Dev错误分析，零重合不能证明开发过程没有Dev影响，也不能证明标签已正确。隐藏Test没有提供，未对Test作零重合承诺。

## 实际重合的文本

### scope200与Dev共有的完整短语

- `a brown horse`
- `a horse`

### E改写88与Dev共有的完整短语

- `a brown dog`
- `a close-up view`
- `are playing`
- `immersed in water`
- `is jumping`
- `is playing`
- `philippines`
- `play`
- `run`

### 官方Train与Dev共有的7个无序文本对

| 一侧文本（规范化） | 另一侧文本（规范化） | Train涉及行数 | Dev涉及行数 |
|---|---|---:|---:|
| a baby | a small baby | 2 | 2 |
| a black and white cow | a black and white horned cow | 2 | 2 |
| are playing | play | 2 | 2 |
| at the camera | into the camera | 1 | 1 |
| driving | driving down | 2 | 2 |
| stocks | us stocks | 2 | 4 |
| suicide bomber | suicide bombers | 2 | 2 |

### 对应的原始行与ID

| Train ID | Train text1 | Train text2 | Dev ID | Dev text1 | Dev text2 |
|---|---|---|---|---|---|
| dico_phrasis_0000113__en-en__flipped | a small baby | a baby | dico_phrasis_0001755__en-en__flipped | a baby | a small baby |
| dico_phrasis_0000113__en-en__flipped | a small baby | a baby | dico_phrasis_0001755__en-en__original | a small baby | a baby |
| dico_phrasis_0000113__en-en__original | a baby | a small baby | dico_phrasis_0001755__en-en__flipped | a baby | a small baby |
| dico_phrasis_0000113__en-en__original | a baby | a small baby | dico_phrasis_0001755__en-en__original | a small baby | a baby |
| dico_phrasis_0000270__en-en__flipped | Suicide Bombers | Suicide bomber | dico_phrasis_0000329__en-en__flipped | Suicide bombers | Suicide bomber |
| dico_phrasis_0000270__en-en__flipped | Suicide Bombers | Suicide bomber | dico_phrasis_0000329__en-en__original | Suicide bomber | Suicide bombers |
| dico_phrasis_0000270__en-en__original | Suicide bomber | Suicide Bombers | dico_phrasis_0000329__en-en__flipped | Suicide bombers | Suicide bomber |
| dico_phrasis_0000270__en-en__original | Suicide bomber | Suicide Bombers | dico_phrasis_0000329__en-en__original | Suicide bomber | Suicide bombers |
| dico_phrasis_0000536__en-en__flipped | driving down | driving | dico_phrasis_0001848__en-en__flipped | driving | driving down |
| dico_phrasis_0000536__en-en__flipped | driving down | driving | dico_phrasis_0001848__en-en__original | driving down | driving |
| dico_phrasis_0000536__en-en__original | driving | driving down | dico_phrasis_0001848__en-en__flipped | driving | driving down |
| dico_phrasis_0000536__en-en__original | driving | driving down | dico_phrasis_0001848__en-en__original | driving down | driving |
| dico_phrasis_0000583__en-en__flipped | A black and white horned cow | A black and white cow | dico_phrasis_0001880__en-en__flipped | a black and white cow | A black and white horned cow |
| dico_phrasis_0000583__en-en__flipped | A black and white horned cow | A black and white cow | dico_phrasis_0001880__en-en__original | A black and white horned cow | a black and white cow |
| dico_phrasis_0000583__en-en__original | A black and white cow | A black and white horned cow | dico_phrasis_0001880__en-en__flipped | a black and white cow | A black and white horned cow |
| dico_phrasis_0000583__en-en__original | A black and white cow | A black and white horned cow | dico_phrasis_0001880__en-en__original | A black and white horned cow | a black and white cow |
| dico_phrasis_0000780__en-en__flipped | are playing | play | dico_phrasis_0000768__en-en__flipped | play | are playing |
| dico_phrasis_0000780__en-en__flipped | are playing | play | dico_phrasis_0000768__en-en__original | are playing | play |
| dico_phrasis_0000780__en-en__original | play | are playing | dico_phrasis_0000768__en-en__flipped | play | are playing |
| dico_phrasis_0000780__en-en__original | play | are playing | dico_phrasis_0000768__en-en__original | are playing | play |
| dico_phrasis_0001615__en-en__flipped | stocks | US stocks | dico_phrasis_0000271__en-en__flipped | US stocks | Stocks |
| dico_phrasis_0001615__en-en__flipped | stocks | US stocks | dico_phrasis_0000271__en-en__original | Stocks | US stocks |
| dico_phrasis_0001615__en-en__flipped | stocks | US stocks | dico_phrasis_0001691__en-en__flipped | Stocks | US stocks |
| dico_phrasis_0001615__en-en__flipped | stocks | US stocks | dico_phrasis_0001691__en-en__original | US stocks | Stocks |
| dico_phrasis_0001615__en-en__original | US stocks | stocks | dico_phrasis_0000271__en-en__flipped | US stocks | Stocks |
| dico_phrasis_0001615__en-en__original | US stocks | stocks | dico_phrasis_0000271__en-en__original | Stocks | US stocks |
| dico_phrasis_0001615__en-en__original | US stocks | stocks | dico_phrasis_0001691__en-en__flipped | Stocks | US stocks |
| dico_phrasis_0001615__en-en__original | US stocks | stocks | dico_phrasis_0001691__en-en__original | US stocks | Stocks |
| dico_phrasis_0003071__en-en__original | into the camera | at the camera | dico_phrasis_0003245__en-en__original | into the camera | at the camera |

## 每批数据的来源、示例与明细

下列示例直接取自训练快照；F/B/E/N为文件中的训练标签，不代表本次重新审核过标签。

### 官方Train

- 来源：[训练CSV](../data/train/dico_nli_train_track1_participant_labeled.csv)。
- 本次统计：从第1行数据起取3042行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 881, 'BACKWARD_ENTAILMENT': 881, 'EQUIVALENCE': 802, 'NEGATIVE_OTHER': 478}。
- 无序文本对交集：7。

| text1 | text2 | 标签 |
|---|---|---|
| Conservative opposition | Conservatives | FORWARD_ENTAILMENT |
| Conservatives | Conservative opposition | BACKWARD_ENTAILMENT |
| to Philippines | for the Philippine nation | EQUIVALENCE |
| off Mexico | on Mexico | NEGATIVE_OTHER |

共有词：`10`、`11`、`12`、`13`、`2`、`23`、`3`、`30`、`38`、`39`、`56`、`6`、`7`、`8`、`a`、`about`、`abuse`、`accompanied`、`across`、`adult`、`afghan`、`afghanistan`、`after`、`against`、`aged`、`ailing`、`air`、`airline`、`airlines`、`airplane`、`airport`、`al`、`all`、`along`、`an`、`and`、`animal`、`animals`、`another`、`anti`、`apple`、`are`、`area`、`arms`、`army`、`around`、`arrest`、`as`、`asian`、`asylum`、`at`、`atop`、`attack`、`australia`、`australian`、`baby`、`back`、`ball`、`bangladesh`、`bank`、`bar`、`baseball`、`be`、`beach`、`before`、`begins`、`behind`、`beside`、`bicycle`、`big`、`bill`、`bird`、`birds`、`black`、`blast`、`blonde`、`boat`、`bomb`、`bomber`、`bombers`、`bombs`、`boston`、`bovine`、`boy`、`bradley`、`branch`、`branches`、`brazil`、`british`、`brotherhood`、`brown`、`brownish`、`bubble`、`building`、`bull`、`bus`、`by`、`called`、`cambodia`、`camera`、`canada`、`canine`、`car`、`cardinals`、`carving`、`cat`、`catch`、`catches`、`cats`、`center`、`chemical`、`chief`、`child`、`children`、`china`、`city`、`claims`、`clash`、`clashes`、`close`、`coast`、`collapse`、`coloring`、`computer`、`constitution`、`control`、`convicted`、`couch`、`couple`、`covered`、`cow`、`cows`、`crash`、`creek`、`crisis`、`crowded`、`cruise`、`cuts`、`cyclone`、`dark`、`daughter`、`dead`、`deadly`、`deal`、`death`、`deaths`、`decker`、`deep`、`despite`、`dining`、`disaster`、`district`、`dog`、`dogs`、`domestic`、`door`、`double`、`down`、`dress`、`dressed`、`driving`、`drone`、`earnings`、`earthquake`、`east`、`egypt`、`egyptian`、`election`、`elections`、`elevated`、`enclosure`、`engine`、`england`、`every`、`experts`、`family`、`feeder`、`feline`、`female`、`fence`、`field`、`fire`、`first`、`five`、`flat`、`for`、`forces`、`forest`、`four`、`francis`、`french`、`fresh`、`friday`、`frisbee`、`from`、`front`、`fur`、`gang`、`gay`、`german`、`germany`、`get`、`ginger`、`girl`、`girls`、`goats`、`government`、`governor`、`grass`、`grassy`、`gray`、`green`、`ground`、`group`、`guide`、`hakimullah`、`hat`、`hay`、`head`、`health`、`held`、`high`、`hill`、`his`、`holding`、`home`、`horned`、`horse`、`hospital`、`huge`、`human`、`immersed`、`in`、`infant`、`inside`、`into`、`iran`、`iranian`、`is`、`israeli`、`italy`、`its`、`jail`、`japan`、`judge`、`jumping`、`killed`、`large`、`law`、`laws`、`laying`、`leader`、`leaping`、`leather`、`little`、`london`、`long`、`looking`、`lush`、`magnitude`、`male`、`mali`、`man`、`mandela`、`marathon`、`march`、`marriage`、`massive`、`may`、`medium`、`mehsud`、`memorial`、`men`、`messy`、`militants`、`military`、`million`、`monochrome`、`morsi`、`moscow`、`most`、`mountain`、`mounted`、`moving`、`mubarak`、`murder`、`muslim`、`national`、`near`、`new`、`newborn`、`next`、`nigerian`、`nine`、`north`、`northern`、`northwest`、`not`、`nuclear`、`of`、`off`、`office`、`oil`、`old`、`on`、`one`、`opposition`、`orange`、`other`、`out`、`over`、`pak`、`pakistan`、`pakistani`、`palestinian`、`palestinians`、`parked`、`parrot`、`party`、`path`、`pay`、`pen`、`people`、`perched`、`person`、`phailin`、`philippines`、`photo`、`photograph`、`picture`、`pit`、`plane`、`plant`、`play`、`playing`、`police`、`political`、`pool`、`pope`、`posing`、`post`、`powerful`、`powers`、`prayer`、`president`、`presidential`、`pro`、`protest`、`protesters`、`protests`、`public`、`puppy`、`purple`、`qaeda`、`quake`、`rally`、`rape`、`rear`、`rebel`、`record`、`red`、`release`、`remaining`、`resting`、`reunions`、`river`、`rocky`、`room`、`ruins`、`run`、`runner`、`running`、`runway`、`russia`、`russian`、`s`、`same`、`sanctions`、`saudi`、`schools`、`scooter`、`screen`、`sea`、`search`、`seated`、`security`、`set`、`several`、`sex`、`sheep`、`shepherd`、`ship`、`shop`、`shopping`、`shore`、`shoreline`、`side`、`sign`、`silver`、`six`、`slide`、`small`、`snow`、`snowy`、`sofas`、`software`、`soldier`、`some`、`song`、`southern`、`spots`、`stand`、`standing`、`station`、`steam`、`steps`、`stocks`、`stop`、`street`、`strike`、`strikes`、`structure`、`suicide`、`supporters`、`supreme`、`surface`、`swing`、`syria`、`syrian`、`table`、`taliban`、`talks`、`tall`、`tan`、`television`、`terminal`、`texas`、`thai`、`that`、`the`、`their`、`them`、`three`、`through`、`thursday`、`to`、`toll`、`top`、`towards`、`town`、`track`、`tracks`、`trade`、`train`、`transport`、`trapped`、`tree`、`truck`、`tuesday`、`tunnel`、`tv`、`two`、`typhoon`、`u`、`ukraine`、`underwater`、`unions`、`united`、`up`、`upon`、`upright`、`us`、`use`、`van`、`vatican`、`vehicle`、`venezuela`、`venezuelan`、`victims`、`victory`、`video`、`view`、`violence`、`visit`、`wall`、`war`、`washington`、`water`、`wave`、`waves`、`weapons`、`wearing`、`west`、`wet`、`while`、`white`、`win`、`window`、`wins`、`wire`、`with`、`within`、`woman`、`women`、`wooden`、`world`、`yard`、`yellow`、`young`。
共有Porter词干：`10`、`11`、`12`、`13`、`2`、`23`、`3`、`30`、`38`、`39`、`56`、`6`、`7`、`8`、`a`、`about`、`abus`、`accompani`、`across`、`adopt`、`adult`、`afghan`、`afghanistan`、`after`、`against`、`age`、`ail`、`air`、`airlin`、`airplan`、`airport`、`al`、`all`、`along`、`an`、`and`、`anim`、`announc`、`anoth`、`anti`、`appl`、`are`、`area`、`arm`、`armi`、`around`、`arrest`、`as`、`asian`、`asylum`、`at`、`atop`、`attack`、`australia`、`australian`、`babi`、`back`、`ball`、`bangladesh`、`bank`、`bar`、`basebal`、`be`、`beach`、`befor`、`begin`、`behind`、`besid`、`bicycl`、`big`、`bill`、`bird`、`black`、`blast`、`blond`、`boat`、`bomb`、`bomber`、`boston`、`bovin`、`bowl`、`boy`、`bradley`、`branch`、`brazil`、`bribe`、`british`、`brotherhood`、`brown`、`brownish`、`bu`、`bubbl`、`build`、`bull`、`by`、`call`、`cambodia`、`camera`、`canada`、`canin`、`car`、`cardin`、`carv`、`cat`、`catch`、`center`、`chemic`、`chief`、`child`、`children`、`china`、`citi`、`claim`、`clash`、`close`、`coast`、`collaps`、`color`、`comput`、`constitut`、`control`、`convict`、`couch`、`coupl`、`cover`、`cow`、`crash`、`creek`、`crisi`、`cross`、`crowd`、`cruis`、`cut`、`cyclon`、`dark`、`daughter`、`dead`、`deadli`、`deal`、`death`、`decker`、`deep`、`despit`、`dine`、`direct`、`disast`、`district`、`do`、`dog`、`domest`、`door`、`doubl`、`down`、`dress`、`drive`、`drone`、`earn`、`earthquak`、`east`、`eat`、`egypt`、`egyptian`、`elect`、`elev`、`enclosur`、`engin`、`england`、`everi`、`expert`、`explos`、`fall`、`famili`、`feeder`、`felin`、`femal`、`fenc`、`field`、`fire`、`first`、`five`、`flat`、`for`、`forc`、`forest`、`four`、`franci`、`french`、`fresh`、`friday`、`frisbe`、`from`、`front`、`fur`、`game`、`gang`、`gate`、`gay`、`german`、`germani`、`get`、`ginger`、`girl`、`goat`、`govern`、`governor`、`grant`、`grass`、`grassi`、`gray`、`green`、`ground`、`group`、`guid`、`hakimullah`、`hat`、`hay`、`head`、`health`、`held`、`hi`、`high`、`hill`、`hold`、`home`、`horn`、`hors`、`hospit`、`hostag`、`hous`、`huge`、`human`、`immers`、`in`、`infant`、`insid`、`into`、`iran`、`iranian`、`is`、`isra`、`issu`、`it`、`itali`、`jail`、`japan`、`judg`、`jump`、`kill`、`land`、`larg`、`law`、`lay`、`leader`、`leap`、`leather`、`link`、`littl`、`london`、`long`、`look`、`lush`、`magnitud`、`male`、`mali`、`man`、`mandela`、`marathon`、`march`、`marriag`、`massiv`、`may`、`medium`、`mehsud`、`memori`、`men`、`messi`、`milit`、`militari`、`million`、`minist`、`monochrom`、`morsi`、`moscow`、`most`、`mount`、`mountain`、`move`、`mubarak`、`murder`、`muslim`、`nation`、`near`、`network`、`new`、`newborn`、`next`、`nigerian`、`nine`、`north`、`northern`、`northwest`、`not`、`nuclear`、`of`、`off`、`offic`、`oil`、`old`、`on`、`one`、`open`、`oper`、`opposit`、`orang`、`other`、`out`、`over`、`pak`、`pakistan`、`pakistani`、`palestinian`、`park`、`parrot`、`parti`、`path`、`pay`、`pen`、`peopl`、`perch`、`perform`、`person`、`phailin`、`philippin`、`photo`、`photograph`、`pictur`、`pilot`、`pit`、`plane`、`plant`、`play`、`polic`、`polit`、`pool`、`pope`、`pose`、`post`、`power`、`prayer`、`presid`、`presidenti`、`pro`、`propos`、`protest`、`public`、`pull`、`puppi`、`purpl`、`qaeda`、`quak`、`raft`、`ralli`、`rape`、`rear`、`rebel`、`record`、`red`、`releas`、`remain`、`rememb`、`rest`、`return`、`reunion`、`river`、`rock`、`rocki`、`room`、`round`、`ruin`、`run`、`runner`、`runway`、`russia`、`russian`、`s`、`same`、`sanction`、`saudi`、`school`、`scooter`、`screen`、`sea`、`search`、`seat`、`secur`、`senat`、`set`、`sever`、`sex`、`sheep`、`shepherd`、`ship`、`shop`、`shore`、`shorelin`、`side`、`sign`、`silver`、`six`、`slide`、`small`、`snow`、`snowi`、`sofa`、`softwar`、`soldier`、`some`、`song`、`southern`、`speak`、`spot`、`stand`、`state`、`station`、`steam`、`step`、`stock`、`stop`、`street`、`strike`、`structur`、`suicid`、`suit`、`support`、`suprem`、`surfac`、`swing`、`syria`、`syrian`、`tabl`、`taliban`、`talk`、`tall`、`tan`、`televis`、`termin`、`texa`、`thai`、`that`、`the`、`their`、`them`、`three`、`through`、`thursday`、`to`、`toll`、`top`、`toward`、`town`、`track`、`trade`、`train`、`transport`、`trap`、`tree`、`truck`、`tuesday`、`tunnel`、`tv`、`two`、`typhoon`、`u`、`ukrain`、`underwat`、`union`、`unit`、`up`、`upon`、`upright`、`us`、`use`、`van`、`vatican`、`vehicl`、`venezuela`、`venezuelan`、`victim`、`victori`、`video`、`view`、`violenc`、`visit`、`wall`、`war`、`washington`、`water`、`wave`、`weapon`、`wear`、`west`、`wet`、`while`、`white`、`win`、`window`、`wire`、`with`、`within`、`woman`、`women`、`wooden`、`world`、`yard`、`yellow`、`young`。
共有完整短语：`a baby`；`a black and white cow`；`a black and white horned cow`；`a black and white photo`；`a black dog`；`a black steam train`；`a black train`；`a boat`；`a boy`；`a brown and white dog`；`a brown dog`；`a brown horse`；`a cat`；`a child`；`a close-up view`；`a gray , black , and white cat`；`a group of people`；`a long train`；`a man`；`a orange and white cat`；`a person`；`a red double decker bus`；`a sheep`；`a small baby`；`a small bird`；`a small boy`；`a small white dog`；`a street`；`a train`；`a white bird`；`a white cat`；`a white dog`；`a woman`；`a wooden table`；`a young boy`；`a young girl`；`a young man`；`air strikes`；`an adult female`；`an apple computer`；`are playing`；`are posing`；`as one`；`at the camera`；`at the train station`；`bangladesh collapse deaths`；`black train`；`brazil`；`by a tree`；`catches fire`；`china`；`clash`；`close up`；`dog`；`domestic cat`；`down the tracks`；`dressed in`；`driving`；`driving down`；`egypt`；`family reunions`；`for boston victims`；`four people`；`from a bird feeder`；`from a tree`；`gay marriage`；`immersed in water`；`in a field`；`in a field of green grass`；`in a lush green field`；`in afghanistan`；`in hay`；`in hospital`；`in prayer and song`；`in syria`；`in the field`；`in the grass`；`in the waves`；`in ukraine`；`into the camera`；`is jumping`；`is playing`；`is running`；`is standing`；`moving at a run`；`muslim brotherhood supporters`；`of a silver train engine`；`on a beach`；`on a crowded street`；`on a runway`；`on a swing`；`on a track`；`on branches`；`on couch`；`on friday`；`on new constitution`；`on the back`；`on the branch`；`on the brown horse`；`on the grass`；`on the ground`；`on the shore`；`on the street`；`on the tracks`；`on the water`；`on thursday`；`one`；`one dead`；`pakistani taliban chief hakimullah mehsud`；`people`；`perched`；`philippines`；`play`；`pope francis`；`presidential elections`；`protesters`；`protests`；`resting on elevated spots`；`run`；`sanctions`；`saudi women`；`stand`；`stocks`；`suicide bomber`；`suicide bombers`；`syria chemical weapons`；`taliban`；`thai protesters`；`the black dog`；`the girl`；`the tracks`；`the train`；`the window`；`three dogs`；`through the deep snow`；`through the grass`；`through the snow`；`through the water`；`thursday`；`trapped`；`two animals`；`two cats`；`two cows`；`two dogs`；`two girls`；`two people`；`two women`；`us`；`us stocks`；`vatican`；`washington post`；`wearing`；`with a baby`；`with a ball`；`with large sofas and a dining table`。

### scope200

- 来源：[训练CSV](../experiments/deberta_original_aug/outputs/run01_original_scope/training_data.csv)。
- 本次统计：从第3043行数据起取200行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 40, 'BACKWARD_ENTAILMENT': 40, 'EQUIVALENCE': 80, 'NEGATIVE_OTHER': 40}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| a red truck | a truck | FORWARD_ENTAILMENT |
| a truck | a red truck | BACKWARD_ENTAILMENT |
| a red truck | a truck that is red | EQUIVALENCE |
| a red truck | a red bridge | NEGATIVE_OTHER |

共有词：`a`、`and`、`black`、`brown`、`building`、`couch`、`cow`、`door`、`dress`、`fence`、`gray`、`green`、`hat`、`horse`、`is`、`red`、`river`、`that`、`tree`、`truck`、`white`、`window`、`yellow`。
共有Porter词干：`a`、`and`、`black`、`brown`、`build`、`couch`、`cow`、`door`、`dress`、`fenc`、`gray`、`green`、`hat`、`hors`、`is`、`red`、`river`、`sofa`、`that`、`tree`、`truck`、`white`、`window`、`yellow`。
共有完整短语：`a brown horse`；`a horse`。

### relation48

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- 本次统计：从第3243行数据起取48行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 12, 'BACKWARD_ENTAILMENT': 12, 'EQUIVALENCE': 12, 'NEGATIVE_OTHER': 12}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| under metal bridge | under bridge | FORWARD_ENTAILMENT |
| under bridge | under metal bridge | BACKWARD_ENTAILMENT |
| under metal bridge | underneath metal bridge | EQUIVALENCE |
| under metal bridge | above metal bridge | NEGATIVE_OTHER |

共有词：无。
共有Porter词干：无。
共有完整短语：无。

### 人工80

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run01_original_manual/training_data.csv)。
- 本次统计：从第3243行数据起取80行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 20, 'BACKWARD_ENTAILMENT': 20, 'EQUIVALENCE': 20, 'NEGATIVE_OTHER': 20}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| hexagonal flasks | flasks | FORWARD_ENTAILMENT |
| flasks | hexagonal flasks | BACKWARD_ENTAILMENT |
| aubergine | eggplant | EQUIVALENCE |
| crimson flasks | ceramic flasks | NEGATIVE_OTHER |

共有词：无。
共有Porter词干：无。
共有完整短语：无。

### 事件200

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run03_event_200/training_data.csv)。
- 本次统计：从第3243行数据起取200行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 40, 'BACKWARD_ENTAILMENT': 40, 'EQUIVALENCE': 80, 'NEGATIVE_OTHER': 40}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| consuming rice | consuming | FORWARD_ENTAILMENT |
| consuming | consuming rice | BACKWARD_ENTAILMENT |
| slowly consuming lamb | consuming lamb slowly | EQUIVALENCE |
| slowly consuming rice | slowly consuming lamb | NEGATIVE_OTHER |

共有词：无。
共有Porter词干：无。
共有完整短语：无。

### 边界120

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run04_boundary_120/training_data.csv)。
- 本次统计：从第3243行数据起取120行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 24, 'BACKWARD_ENTAILMENT': 24, 'EQUIVALENCE': 48, 'NEGATIVE_OTHER': 24}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| frying dumplings | cooking dumplings | FORWARD_ENTAILMENT |
| cooking dumplings | frying dumplings | BACKWARD_ENTAILMENT |
| biologists | biological scientists | EQUIVALENCE |
| completed repairs | scheduled repairs | NEGATIVE_OTHER |

共有词：无。
共有Porter词干：无。
共有完整短语：无。

### 局部128

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run11_local128_manual/training_data.csv)。
- 本次统计：从第3291行数据起取128行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 32, 'BACKWARD_ENTAILMENT': 32, 'EQUIVALENCE': 32, 'NEGATIVE_OTHER': 32}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| Kelvori-branded portable radios | portable radios | FORWARD_ENTAILMENT |
| portable radios | Kelvori-branded portable radios | BACKWARD_ENTAILMENT |
| Kelvori-branded portable radios | portable radios branded Kelvori | EQUIVALENCE |
| Kelvori-branded radios | Kelvorin-branded radios | NEGATIVE_OTHER |

共有词：无。
共有Porter词干：无。
共有完整短语：无。

### semantic64

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv)。
- 本次统计：从第3291行数据起取64行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 16, 'BACKWARD_ENTAILMENT': 16, 'EQUIVALENCE': 16, 'NEGATIVE_OTHER': 16}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| workers drinking coffee | workers drinking | FORWARD_ENTAILMENT |
| workers drinking | workers drinking coffee | BACKWARD_ENTAILMENT |
| quickly drinking coffee | drinking coffee quickly | EQUIVALENCE |
| drinking coffee | cooking rice during June | NEGATIVE_OTHER |

共有词：无。
共有Porter词干：无。
共有完整短语：无。

### 家族768

- 来源：[训练CSV](../experiments/deberta_family_aug/prepared/training_data_manual.csv)。
- 本次统计：从第3291行数据起取768行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 192, 'BACKWARD_ENTAILMENT': 192, 'EQUIVALENCE': 192, 'NEGATIVE_OTHER': 192}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| beige cabinets | cabinets | FORWARD_ENTAILMENT |
| cabinets | beige cabinets | BACKWARD_ENTAILMENT |
| beige cabinets or stools | stools or beige cabinets | EQUIVALENCE |
| beige cabinets | metal cabinets | NEGATIVE_OTHER |

共有词：无。
共有Porter词干：无。
共有完整短语：无。

### E改写88

- 来源：[训练CSV](../experiments/deberta_closure/closure88_train.csv)。
- 本次统计：从第3043行数据起取88行（不计表头）。
- 标签数量：{'EQUIVALENCE': 88}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| play | plays | EQUIVALENCE |

共有词：`a`、`across`、`adult`、`an`、`and`、`are`、`at`、`atop`、`black`、`brown`、`canine`、`close`、`dog`、`dogs`、`field`、`fur`、`grassy`、`ground`、`holding`、`immersed`、`in`、`into`、`is`、`its`、`jumping`、`leaping`、`male`、`man`、`monochrome`、`mounted`、`moving`、`of`、`on`、`philippines`、`photograph`、`picture`、`play`、`playing`、`run`、`s`、`seated`、`snow`、`surface`、`terminal`、`the`、`to`、`tracks`、`transport`、`up`、`upon`、`upright`、`vehicle`、`view`、`water`、`while`、`white`、`with`、`within`。
共有Porter词干：`a`、`across`、`adult`、`an`、`and`、`announc`、`are`、`at`、`atop`、`black`、`brown`、`canin`、`close`、`color`、`crowd`、`direct`、`dog`、`eat`、`field`、`fur`、`grassi`、`ground`、`hold`、`immers`、`in`、`into`、`is`、`it`、`jump`、`leap`、`male`、`man`、`monochrom`、`mount`、`move`、`of`、`on`、`open`、`perch`、`philippin`、`photograph`、`pictur`、`play`、`pose`、`remain`、`run`、`s`、`seat`、`snow`、`surfac`、`termin`、`the`、`to`、`track`、`transport`、`up`、`upon`、`upright`、`vehicl`、`view`、`water`、`while`、`white`、`win`、`with`、`within`。
共有完整短语：`a brown dog`；`a close-up view`；`are playing`；`immersed in water`；`is jumping`；`is playing`；`philippines`；`play`；`run`。

### 复制200

- 来源：[训练CSV](../experiments/deberta_original_aug/outputs/run02_original_copy/training_data.csv)。
- 本次统计：从第3043行数据起取200行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 40, 'BACKWARD_ENTAILMENT': 40, 'EQUIVALENCE': 80, 'NEGATIVE_OTHER': 40}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| of Its Chemical Weapons Stockpile | Chemical Weapons | FORWARD_ENTAILMENT |
| Chemical Weapons | of Its Chemical Weapons Stockpile | BACKWARD_ENTAILMENT |
| are running | moving at a run | EQUIVALENCE |
| DPRK leader | North Korea | NEGATIVE_OTHER |

共有词：`11`、`2`、`38`、`6`、`a`、`adult`、`afghan`、`afghanistan`、`after`、`airline`、`airport`、`al`、`all`、`along`、`an`、`and`、`another`、`are`、`as`、`at`、`atop`、`attack`、`baby`、`back`、`beside`、`big`、`bird`、`black`、`bomb`、`bomber`、`boston`、`boy`、`brazil`、`british`、`brotherhood`、`brown`、`building`、`bus`、`by`、`camera`、`car`、`cat`、`chemical`、`china`、`city`、`claims`、`clash`、`clashes`、`collapse`、`computer`、`couple`、`covered`、`dark`、`decker`、`dog`、`double`、`down`、`dressed`、`egypt`、`every`、`feline`、`fence`、`field`、`five`、`for`、`four`、`french`、`from`、`front`、`girl`、`girls`、`grass`、`grassy`、`green`、`ground`、`guide`、`hat`、`in`、`inside`、`is`、`its`、`jumping`、`large`、`leader`、`leaping`、`male`、`man`、`marriage`、`men`、`militants`、`military`、`most`、`mountain`、`moving`、`mubarak`、`muslim`、`national`、`near`、`new`、`newborn`、`nine`、`north`、`of`、`off`、`on`、`other`、`over`、`pakistan`、`pakistani`、`palestinian`、`palestinians`、`people`、`person`、`play`、`playing`、`police`、`pool`、`posing`、`president`、`qaeda`、`red`、`run`、`running`、`runway`、`s`、`same`、`schools`、`set`、`sex`、`sheep`、`side`、`six`、`snow`、`snowy`、`some`、`standing`、`steps`、`stocks`、`street`、`syria`、`table`、`taliban`、`talks`、`that`、`the`、`three`、`to`、`top`、`tracks`、`train`、`tree`、`tunnel`、`two`、`u`、`up`、`upon`、`us`、`water`、`weapons`、`while`、`white`、`with`、`woman`、`women`、`wooden`、`young`。
共有Porter词干：`11`、`2`、`38`、`6`、`a`、`adult`、`afghan`、`afghanistan`、`after`、`airlin`、`airport`、`al`、`all`、`along`、`an`、`and`、`anoth`、`are`、`arrest`、`as`、`at`、`atop`、`attack`、`babi`、`back`、`besid`、`big`、`bird`、`black`、`bomb`、`bomber`、`boston`、`boy`、`brazil`、`british`、`brotherhood`、`brown`、`bu`、`build`、`by`、`camera`、`car`、`cat`、`chemic`、`china`、`citi`、`claim`、`clash`、`collaps`、`comput`、`coupl`、`cover`、`dark`、`decker`、`dog`、`doubl`、`down`、`dress`、`egypt`、`everi`、`felin`、`fenc`、`field`、`five`、`for`、`four`、`french`、`from`、`front`、`girl`、`grass`、`grassi`、`green`、`ground`、`guid`、`hat`、`in`、`insid`、`is`、`it`、`jump`、`larg`、`leader`、`leap`、`male`、`man`、`marriag`、`men`、`milit`、`militari`、`most`、`mountain`、`move`、`mubarak`、`muslim`、`nation`、`near`、`new`、`newborn`、`nine`、`north`、`of`、`off`、`on`、`open`、`other`、`over`、`pakistan`、`pakistani`、`palestinian`、`park`、`peopl`、`person`、`play`、`polic`、`pool`、`pose`、`presid`、`qaeda`、`red`、`return`、`run`、`runway`、`s`、`same`、`school`、`set`、`sex`、`sheep`、`side`、`six`、`snow`、`snowi`、`some`、`speak`、`stand`、`step`、`stock`、`street`、`syria`、`tabl`、`taliban`、`talk`、`that`、`the`、`three`、`to`、`top`、`toward`、`track`、`train`、`tree`、`tunnel`、`two`、`u`、`union`、`up`、`upon`、`us`、`use`、`wall`、`water`、`weapon`、`while`、`white`、`with`、`woman`、`women`、`wooden`、`young`。
共有完整短语：`a black dog`；`a man`；`a person`；`a sheep`；`a street`；`a young boy`；`are playing`；`are posing`；`at the camera`；`down the tracks`；`from a tree`；`in a field`；`in a field of green grass`；`in afghanistan`；`in syria`；`is standing`；`moving at a run`；`on the grass`；`on the water`；`taliban`；`the girl`；`the train`；`two women`。

### 复制80

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run02_original_copy/training_data.csv)。
- 本次统计：从第3243行数据起取80行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 20, 'BACKWARD_ENTAILMENT': 20, 'EQUIVALENCE': 20, 'NEGATIVE_OTHER': 20}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| US Senate | Senate | FORWARD_ENTAILMENT |
| Senate | US Senate | BACKWARD_ENTAILMENT |
| is trying | attempting | EQUIVALENCE |
| Two large black dogs | Two white dogs | NEGATIVE_OTHER |

共有词：`12`、`2`、`6`、`a`、`adult`、`after`、`against`、`along`、`an`、`and`、`animals`、`arms`、`at`、`attack`、`baby`、`big`、`birds`、`black`、`bovine`、`boy`、`brown`、`bus`、`cat`、`chemical`、`city`、`claims`、`collapse`、`coloring`、`computer`、`cow`、`crowded`、`decker`、`dog`、`dogs`、`double`、`drone`、`england`、`experts`、`field`、`five`、`for`、`four`、`girl`、`girls`、`government`、`green`、`group`、`his`、`horse`、`in`、`is`、`its`、`large`、`little`、`male`、`man`、`military`、`northern`、`of`、`on`、`opposition`、`over`、`pakistan`、`pakistani`、`party`、`people`、`protesters`、`red`、`runway`、`s`、`set`、`snow`、`snowy`、`southern`、`steps`、`street`、`strike`、`swing`、`thai`、`the`、`three`、`to`、`train`、`two`、`typhoon`、`united`、`upon`、`us`、`use`、`vehicle`、`weapons`、`white`、`with`、`within`、`women`、`young`。
共有Porter词干：`12`、`2`、`6`、`a`、`adult`、`after`、`against`、`along`、`an`、`and`、`anim`、`arm`、`at`、`attack`、`babi`、`big`、`bird`、`black`、`bovin`、`boy`、`brown`、`bu`、`cat`、`chemic`、`citi`、`claim`、`collaps`、`color`、`comput`、`cow`、`crowd`、`decker`、`dog`、`doubl`、`drone`、`england`、`expert`、`field`、`five`、`for`、`four`、`girl`、`govern`、`green`、`group`、`hi`、`hold`、`hors`、`in`、`is`、`it`、`larg`、`littl`、`male`、`man`、`militari`、`northern`、`of`、`on`、`open`、`opposit`、`over`、`pakistan`、`pakistani`、`park`、`parti`、`peopl`、`protest`、`red`、`runway`、`s`、`senat`、`set`、`snow`、`snowi`、`southern`、`step`、`street`、`strike`、`swing`、`thai`、`the`、`three`、`to`、`train`、`two`、`typhoon`、`unit`、`upon`、`us`、`use`、`vehicl`、`weapon`、`white`、`with`、`within`、`women`、`young`。
共有完整短语：`a black and white cow`；`a black dog`；`on a crowded street`；`on a swing`；`protesters`；`the train`；`two people`。

### 复制120

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run05_boundary_copy120/training_data.csv)。
- 本次统计：从第3243行数据起取120行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 24, 'BACKWARD_ENTAILMENT': 24, 'EQUIVALENCE': 48, 'NEGATIVE_OTHER': 24}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| US Senate | Senate | FORWARD_ENTAILMENT |
| Senate | US Senate | BACKWARD_ENTAILMENT |
| Birds | the avian animals | EQUIVALENCE |
| with a gemmed collar | with a blue collar | NEGATIVE_OTHER |

共有词：`11`、`3`、`6`、`a`、`about`、`adult`、`after`、`against`、`an`、`and`、`animals`、`are`、`arms`、`as`、`at`、`atop`、`baby`、`bangladesh`、`birds`、`black`、`bomb`、`bombs`、`boston`、`bovine`、`boy`、`brotherhood`、`bus`、`by`、`camera`、`car`、`chemical`、`china`、`collapse`、`coloring`、`computer`、`cow`、`death`、`deaths`、`decker`、`dog`、`dogs`、`double`、`drone`、`election`、`elections`、`england`、`experts`、`fence`、`field`、`five`、`for`、`four`、`french`、`from`、`girl`、`government`、`grass`、`grassy`、`green`、`ground`、`group`、`hat`、`high`、`horned`、`horse`、`in`、`inside`、`into`、`is`、`its`、`large`、`laws`、`leaping`、`leather`、`little`、`long`、`male`、`man`、`militants`、`military`、`most`、`muslim`、`near`、`newborn`、`northern`、`of`、`office`、`old`、`on`、`one`、`opposition`、`pakistan`、`pakistani`、`party`、`people`、`person`、`play`、`playing`、`president`、`protesters`、`red`、`running`、`s`、`seated`、`shop`、`six`、`slide`、`snow`、`snowy`、`street`、`strike`、`suicide`、`swing`、`thai`、`the`、`three`、`to`、`toll`、`two`、`u`、`united`、`up`、`upon`、`us`、`war`、`water`、`white`、`with`、`within`。
共有Porter词干：`11`、`3`、`6`、`a`、`about`、`adopt`、`adult`、`after`、`against`、`an`、`and`、`anim`、`are`、`arm`、`as`、`at`、`atop`、`attack`、`babi`、`bangladesh`、`bird`、`black`、`bomb`、`boston`、`bovin`、`boy`、`brotherhood`、`bu`、`by`、`camera`、`car`、`chemic`、`china`、`close`、`collaps`、`color`、`comput`、`cow`、`death`、`decker`、`dog`、`doubl`、`drone`、`elect`、`england`、`expert`、`explos`、`fenc`、`field`、`five`、`for`、`four`、`french`、`from`、`girl`、`govern`、`grass`、`grassi`、`green`、`ground`、`group`、`hat`、`high`、`horn`、`hors`、`hostag`、`hous`、`in`、`insid`、`into`、`is`、`it`、`larg`、`law`、`leap`、`leather`、`littl`、`long`、`male`、`man`、`milit`、`militari`、`most`、`muslim`、`near`、`newborn`、`northern`、`of`、`offic`、`old`、`on`、`one`、`open`、`opposit`、`pakistan`、`pakistani`、`parti`、`peopl`、`person`、`play`、`presid`、`protest`、`pull`、`red`、`run`、`s`、`school`、`seat`、`senat`、`shop`、`six`、`slide`、`snow`、`snowi`、`speak`、`street`、`strike`、`suicid`、`swing`、`thai`、`the`、`three`、`to`、`toll`、`toward`、`two`、`u`、`unit`、`up`、`upon`、`us`、`war`、`water`、`white`、`with`、`within`。
共有完整短语：`a black and white cow`；`a black and white horned cow`；`a black dog`；`are playing`；`as one`；`at the camera`；`bangladesh collapse deaths`；`is running`；`on a swing`；`on the grass`；`protesters`；`two people`。

### 复制128

- 来源：[训练CSV](../experiments/deberta_manual_aug/outputs/run12_local128_copy/training_data.csv)。
- 本次统计：从第3291行数据起取128行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 32, 'BACKWARD_ENTAILMENT': 32, 'EQUIVALENCE': 32, 'NEGATIVE_OTHER': 32}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| with tan furniture and flat screen television | with large screen television | FORWARD_ENTAILMENT |
| with large screen television | with tan furniture and flat screen television | BACKWARD_ENTAILMENT |
| set | sets off | EQUIVALENCE |
| on the ground | parked | NEGATIVE_OTHER |

共有词：`2`、`3`、`7`、`a`、`against`、`air`、`airport`、`along`、`an`、`and`、`are`、`area`、`around`、`asian`、`at`、`baby`、`ball`、`bangladesh`、`beside`、`bill`、`bird`、`black`、`brown`、`by`、`camera`、`canada`、`car`、`chemical`、`child`、`city`、`clash`、`clashes`、`control`、`creek`、`dead`、`death`、`deaths`、`dog`、`dogs`、`door`、`down`、`driving`、`earthquake`、`egypt`、`election`、`fence`、`field`、`fire`、`five`、`flat`、`for`、`from`、`front`、`girl`、`grass`、`gray`、`green`、`ground`、`hat`、`hill`、`his`、`home`、`in`、`inside`、`iranian`、`is`、`israeli`、`killed`、`large`、`leaping`、`long`、`magnitude`、`mali`、`man`、`morsi`、`of`、`off`、`office`、`oil`、`old`、`on`、`one`、`other`、`over`、`parked`、`people`、`philippines`、`pit`、`pool`、`posing`、`president`、`presidential`、`pro`、`protesters`、`quake`、`red`、`river`、`russia`、`russian`、`s`、`screen`、`set`、`sheep`、`side`、`small`、`standing`、`stocks`、`street`、`strike`、`strikes`、`supporters`、`syria`、`table`、`taliban`、`talks`、`tan`、`television`、`that`、`the`、`three`、`through`、`to`、`tracks`、`train`、`truck`、`tv`、`two`、`typhoon`、`us`、`weapons`、`white`、`with`、`within`、`woman`、`wooden`、`yellow`、`young`。
共有Porter词干：`2`、`3`、`7`、`a`、`against`、`air`、`airport`、`along`、`an`、`and`、`are`、`area`、`around`、`asian`、`at`、`babi`、`ball`、`bangladesh`、`besid`、`bill`、`bird`、`black`、`brown`、`by`、`camera`、`canada`、`car`、`chemic`、`child`、`citi`、`claim`、`clash`、`close`、`control`、`crash`、`creek`、`dead`、`death`、`dog`、`door`、`down`、`drive`、`earthquak`、`eat`、`egypt`、`elect`、`fenc`、`field`、`fire`、`five`、`flat`、`for`、`from`、`front`、`girl`、`grass`、`gray`、`green`、`ground`、`hat`、`hi`、`hill`、`home`、`hous`、`in`、`insid`、`iranian`、`is`、`isra`、`kill`、`larg`、`leap`、`long`、`magnitud`、`mali`、`man`、`minist`、`morsi`、`nation`、`of`、`off`、`offic`、`oil`、`old`、`on`、`one`、`open`、`other`、`over`、`park`、`peopl`、`philippin`、`pit`、`pool`、`pose`、`presid`、`presidenti`、`pro`、`protest`、`quak`、`rebel`、`red`、`river`、`russia`、`russian`、`s`、`screen`、`senat`、`set`、`sheep`、`side`、`small`、`stand`、`stock`、`street`、`strike`、`support`、`syria`、`tabl`、`taliban`、`talk`、`tan`、`televis`、`that`、`the`、`three`、`through`、`to`、`toward`、`track`、`train`、`truck`、`tv`、`two`、`typhoon`、`us`、`weapon`、`white`、`with`、`within`、`woman`、`wooden`、`yellow`、`young`。
共有完整短语：`a man`；`a white dog`；`a woman`；`a young girl`；`at the camera`；`driving`；`in the field`；`is standing`；`on the ground`；`philippines`；`protesters`；`syria chemical weapons`；`the tracks`；`two dogs`；`two people`。

### 复制768

- 来源：[训练CSV](../experiments/deberta_family_aug/prepared/training_data_copy.csv)。
- 本次统计：从第3291行数据起取768行（不计表头）。
- 标签数量：{'FORWARD_ENTAILMENT': 192, 'BACKWARD_ENTAILMENT': 192, 'EQUIVALENCE': 192, 'NEGATIVE_OTHER': 192}。
- 无序文本对交集：0。

| text1 | text2 | 标签 |
|---|---|---|
| after U.S.-Russia Deal | US-Russia Deal | FORWARD_ENTAILMENT |
| US-Russia Deal | after U.S.-Russia Deal | BACKWARD_ENTAILMENT |
| Allowed | to permit | EQUIVALENCE |
| In Oklahoma City | Oklahoma City Airport | NEGATIVE_OTHER |

共有词：`12`、`2`、`3`、`6`、`8`、`a`、`about`、`abuse`、`accompanied`、`across`、`adult`、`after`、`against`、`air`、`airline`、`airlines`、`airport`、`al`、`all`、`along`、`an`、`and`、`animals`、`another`、`are`、`area`、`arms`、`army`、`around`、`as`、`asian`、`at`、`atop`、`australia`、`australian`、`baby`、`ball`、`bangladesh`、`bank`、`bar`、`beach`、`before`、`begins`、`beside`、`bicycle`、`big`、`bill`、`bird`、`birds`、`black`、`boat`、`boy`、`brazil`、`british`、`brotherhood`、`brown`、`brownish`、`building`、`bus`、`by`、`camera`、`canine`、`car`、`cat`、`catches`、`chemical`、`chief`、`child`、`children`、`china`、`city`、`claims`、`clashes`、`close`、`collapse`、`constitution`、`control`、`couple`、`cow`、`cows`、`crash`、`crisis`、`crowded`、`dark`、`dead`、`deal`、`death`、`deaths`、`decker`、`dining`、`dog`、`dogs`、`double`、`down`、`dressed`、`driving`、`drone`、`earnings`、`earthquake`、`egypt`、`election`、`elections`、`elevated`、`engine`、`every`、`experts`、`feeder`、`feline`、`fence`、`field`、`fire`、`five`、`for`、`forces`、`four`、`francis`、`friday`、`frisbee`、`from`、`front`、`fur`、`german`、`germany`、`girl`、`girls`、`government`、`grass`、`grassy`、`gray`、`green`、`ground`、`group`、`guide`、`hat`、`hay`、`hill`、`holding`、`home`、`horned`、`horse`、`hospital`、`in`、`inside`、`into`、`is`、`israeli`、`italy`、`japan`、`jumping`、`killed`、`large`、`laws`、`little`、`long`、`magnitude`、`male`、`mali`、`man`、`mandela`、`march`、`massive`、`may`、`men`、`militants`、`morsi`、`mountain`、`moving`、`murder`、`muslim`、`national`、`near`、`new`、`next`、`nigerian`、`nine`、`north`、`northwest`、`of`、`office`、`oil`、`on`、`one`、`opposition`、`over`、`pak`、`pakistan`、`pakistani`、`palestinian`、`palestinians`、`party`、`people`、`perched`、`person`、`photo`、`picture`、`plane`、`play`、`playing`、`police`、`political`、`pope`、`posing`、`post`、`president`、`presidential`、`pro`、`protesters`、`protests`、`purple`、`rally`、`red`、`release`、`remaining`、`resting`、`room`、`run`、`runner`、`running`、`russia`、`russian`、`s`、`sanctions`、`sea`、`search`、`security`、`set`、`several`、`sex`、`sheep`、`shepherd`、`ship`、`shop`、`shore`、`shoreline`、`six`、`slide`、`small`、`snow`、`snowy`、`stand`、`standing`、`station`、`steam`、`steps`、`stocks`、`street`、`strike`、`strikes`、`supporters`、`supreme`、`swing`、`syria`、`table`、`tall`、`tan`、`terminal`、`thai`、`the`、`three`、`through`、`thursday`、`to`、`toll`、`top`、`town`、`track`、`tracks`、`trade`、`train`、`transport`、`tree`、`truck`、`tv`、`two`、`typhoon`、`u`、`ukraine`、`up`、`upon`、`us`、`use`、`view`、`war`、`water`、`weapons`、`wearing`、`wet`、`while`、`white`、`with`、`within`、`woman`、`women`、`wooden`、`yellow`、`young`。
共有Porter词干：`12`、`2`、`3`、`6`、`8`、`a`、`about`、`abus`、`accompani`、`across`、`adult`、`after`、`against`、`air`、`airlin`、`airport`、`al`、`all`、`along`、`an`、`and`、`anim`、`anoth`、`are`、`area`、`arm`、`armi`、`around`、`as`、`asian`、`at`、`atop`、`attack`、`australia`、`australian`、`babi`、`ball`、`bangladesh`、`bank`、`bar`、`beach`、`befor`、`begin`、`besid`、`bicycl`、`big`、`bill`、`bird`、`black`、`boat`、`bomb`、`boy`、`brazil`、`bribe`、`british`、`brotherhood`、`brown`、`brownish`、`bu`、`build`、`by`、`call`、`camera`、`canin`、`car`、`cat`、`catch`、`chemic`、`chief`、`child`、`children`、`china`、`citi`、`claim`、`clash`、`close`、`collaps`、`color`、`constitut`、`control`、`coupl`、`cow`、`crash`、`crisi`、`cross`、`crowd`、`dark`、`dead`、`deal`、`death`、`decker`、`dine`、`dog`、`doubl`、`down`、`dress`、`drive`、`drone`、`earn`、`earthquak`、`egypt`、`elect`、`elev`、`engin`、`everi`、`expert`、`explos`、`feeder`、`felin`、`fenc`、`field`、`fire`、`five`、`for`、`forc`、`four`、`franci`、`friday`、`frisbe`、`from`、`front`、`fur`、`german`、`germani`、`girl`、`govern`、`grass`、`grassi`、`gray`、`green`、`ground`、`group`、`guid`、`hat`、`hay`、`hill`、`hold`、`home`、`horn`、`hors`、`hospit`、`hous`、`in`、`insid`、`into`、`is`、`isra`、`it`、`itali`、`japan`、`jump`、`kill`、`larg`、`law`、`littl`、`long`、`magnitud`、`male`、`mali`、`man`、`mandela`、`march`、`massiv`、`may`、`men`、`milit`、`minist`、`morsi`、`mountain`、`move`、`murder`、`muslim`、`nation`、`near`、`new`、`next`、`nigerian`、`nine`、`north`、`northwest`、`of`、`offic`、`oil`、`on`、`one`、`open`、`oper`、`opposit`、`over`、`pak`、`pakistan`、`pakistani`、`palestinian`、`park`、`parti`、`peopl`、`perch`、`person`、`photo`、`photograph`、`pictur`、`pilot`、`plane`、`play`、`polic`、`polit`、`pope`、`pose`、`post`、`presid`、`presidenti`、`pro`、`protest`、`purpl`、`ralli`、`red`、`releas`、`remain`、`rest`、`room`、`run`、`runner`、`russia`、`russian`、`s`、`sanction`、`school`、`sea`、`search`、`secur`、`senat`、`set`、`sever`、`sex`、`sheep`、`shepherd`、`ship`、`shop`、`shore`、`shorelin`、`six`、`slide`、`small`、`snow`、`snowi`、`spot`、`stand`、`state`、`station`、`steam`、`step`、`stock`、`street`、`strike`、`support`、`suprem`、`swing`、`syria`、`tabl`、`talk`、`tall`、`tan`、`termin`、`thai`、`the`、`three`、`through`、`thursday`、`to`、`toll`、`top`、`toward`、`town`、`track`、`trade`、`train`、`transport`、`tree`、`truck`、`tv`、`two`、`typhoon`、`u`、`ukrain`、`union`、`up`、`upon`、`us`、`use`、`view`、`war`、`water`、`weapon`、`wear`、`wet`、`while`、`white`、`win`、`with`、`within`、`woman`、`women`、`wooden`、`yellow`、`young`。
共有完整短语：`a black and white horned cow`；`a black and white photo`；`a black dog`；`a black steam train`；`a boy`；`a brown and white dog`；`a child`；`a group of people`；`a man`；`a person`；`a white dog`；`a woman`；`a young boy`；`a young girl`；`are playing`；`are posing`；`as one`；`at the camera`；`at the train station`；`bangladesh collapse deaths`；`by a tree`；`catches fire`；`close up`；`dressed in`；`driving`；`four people`；`from a bird feeder`；`from a tree`；`in a field`；`in a field of green grass`；`in hay`；`in hospital`；`in syria`；`in the field`；`is playing`；`is running`；`is standing`；`moving at a run`；`on a crowded street`；`on a track`；`on friday`；`on new constitution`；`on the grass`；`on the water`；`on thursday`；`one dead`；`people`；`perched`；`play`；`pope francis`；`protesters`；`protests`；`run`；`sanctions`；`the black dog`；`the tracks`；`three dogs`；`thursday`；`two cows`；`two dogs`；`two people`；`two women`；`us`。

## 输入文件指纹

SHA-256用于标识这次实际检查的文件版本。

| 文件 | SHA-256 |
|---|---|
| [data/dev/dico_nli_dev_track1_participant_labeled.csv](../data/dev/dico_nli_dev_track1_participant_labeled.csv) | `80b7152f7ccf0d349d42ad2f098988106689b918b36337e5756b121d17513511` |
| [data/train/dico_nli_train_track1_participant_labeled.csv](../data/train/dico_nli_train_track1_participant_labeled.csv) | `5e8dd9f3a7e000933fec3c9e64d064f9a25586f141f58d6e6afb47c4fc797211` |
| [experiments/deberta_original_aug/outputs/run01_original_scope/training_data.csv](../experiments/deberta_original_aug/outputs/run01_original_scope/training_data.csv) | `7845cacebdd517664809c2acd15af394a200ed970b07f4cb55a9343274bf90a2` |
| [experiments/deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv](../experiments/deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv) | `b6a1ace5674efd71a21ef7142b656e36680c3d93defb0beaed26c70bacc247d3` |
| [experiments/deberta_manual_aug/outputs/run01_original_manual/training_data.csv](../experiments/deberta_manual_aug/outputs/run01_original_manual/training_data.csv) | `24ef0c86444db8d44959ef33bd46fb918d06398ea1729725697662ea189d0ecb` |
| [experiments/deberta_manual_aug/outputs/run03_event_200/training_data.csv](../experiments/deberta_manual_aug/outputs/run03_event_200/training_data.csv) | `3bab25014d1156af0ab6aed38b2566416848c7181d176e627ad7d3aa91a649ab` |
| [experiments/deberta_manual_aug/outputs/run04_boundary_120/training_data.csv](../experiments/deberta_manual_aug/outputs/run04_boundary_120/training_data.csv) | `a615e4f03bc7eded5d540a6ef87b2980fcf1b816a05cbf85a583c0b7731c4741` |
| [experiments/deberta_manual_aug/outputs/run11_local128_manual/training_data.csv](../experiments/deberta_manual_aug/outputs/run11_local128_manual/training_data.csv) | `3edf99317720b745e99f50d36df8dd4abf448e9fa39fa9930e95b778b4c66711` |
| [experiments/deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv](../experiments/deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv) | `d7ce908b9069dc032b33a928111143df7cd105dd6c93d320274de9ea9908bbfa` |
| [experiments/deberta_family_aug/prepared/training_data_manual.csv](../experiments/deberta_family_aug/prepared/training_data_manual.csv) | `c905efefef5740d4eb05555c04957947e9823fdaee130640b886ea8853d2fa60` |
| [experiments/deberta_closure/closure88_train.csv](../experiments/deberta_closure/closure88_train.csv) | `e6a82efdd3c5afeb378d48c1033e2a3e1464abab431b160eb6189ffddfe63422` |
| [experiments/deberta_original_aug/outputs/run02_original_copy/training_data.csv](../experiments/deberta_original_aug/outputs/run02_original_copy/training_data.csv) | `40bb15fbd46e924c9cbacceb1f6045cad7b884cda9419bd990290cdf454b4c24` |
| [experiments/deberta_manual_aug/outputs/run02_original_copy/training_data.csv](../experiments/deberta_manual_aug/outputs/run02_original_copy/training_data.csv) | `6a69939402ba3b279dfa2c9d22ac289976594663f56ad134c12b8d8e103aca8d` |
| [experiments/deberta_manual_aug/outputs/run05_boundary_copy120/training_data.csv](../experiments/deberta_manual_aug/outputs/run05_boundary_copy120/training_data.csv) | `2c6d647ea1a2cce96ff0e914cc1df54f24e1dfce6ff1be23e309f2366e1b4237` |
| [experiments/deberta_manual_aug/outputs/run12_local128_copy/training_data.csv](../experiments/deberta_manual_aug/outputs/run12_local128_copy/training_data.csv) | `32608de1c1e0d6b0a43d62479fa90326a86c32f1330c3415846582421562eb86` |
| [experiments/deberta_family_aug/prepared/training_data_copy.csv](../experiments/deberta_family_aug/prepared/training_data_copy.csv) | `853be417d69109225c1785f090607be71b3b0a29285f24f18b10b971e5d57d0a` |
