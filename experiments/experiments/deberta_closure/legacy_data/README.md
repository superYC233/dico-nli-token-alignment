# 闭包实验的历史CSV

这里把原先散落在项目根目录和experiments根目录的两批文件分开归档，移动时没有修改CSV内容。

| 目录 | 原位置 | 内容 |
|---|---|---|
| [reviewed](reviewed) | 项目根目录的closure_pairs.csv、closure_rows.csv | 后期留下的较小样本表；文件名不能单独证明每条标签正确 |
| [unfiltered](unfiltered) | experiments根目录的同名CSV | 较早留下的较大样本表，与reviewed不是同一份数据 |

生成脚本归到[generate_legacy_closure.py](../generate_legacy_closure.py)。运行后写入`../outputs/closure_generated/`，不会覆盖本目录归档的两批CSV。脚本仍使用原来的Train关系图和Dev文本排除规则，生成数据需要另行审核。

训练入口现有的[closure88_train.csv](../closure88_train.csv)是另一份完整训练快照，其位置和内容没有改变。关于E改写88的实验结果见[项目README第8阶段](../../../README.md#8-数据增强每批数据从哪里来检查了什么)。
