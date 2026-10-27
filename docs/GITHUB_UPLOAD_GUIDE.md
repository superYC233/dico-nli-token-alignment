# 本次GitHub更新

目标仓库：[superYC233/dico-nli-token-alignment](https://github.com/superYC233/dico-nli-token-alignment)。

2026-10-27已通过Git读取远端提交 `c13efa4986b1da954d37fdf14548e85e0c49827a`，当时远端有150个文件。本次在更新实验内容后，又整理了根目录散文件和文档链接；具体新增、修改文件以更新包旁的`upload_manifest.csv`为准。

## 本次放入哪些内容

- 最新README、数据说明、方法说明和Dev重合审计。
- Lexical及后续实验代码，以及[分类索引](../experiments/README.md)；S01、S11、S12、S14、S16都包含在内。根目录散文件的新位置见[移动记录](DIRECTORY_ORGANIZATION_20261027.json)。
- 人工增强源表、理由标注，以及上述五个融合入口直接依赖的训练快照。完整快照含官方Train副本，原始数据出处见 [data/README](../data/README.md)。
- 精选历史成绩和融合成员预测，用于核对报告中的历史CSV计算结果。
- 本次S14的实际结果及提交包：independent 0.545861，pair-aware 0.962381。S16也保留本次实际预测记录。

更新包不包含模型权重、Hugging Face缓存、Python缓存、本机IDE配置、备份、官方data目录中的CSV，以及64MB的全组合搜索表。历史权重路径只是原运行记录，不代表GitHub提供对应权重下载。

## 怎么使用更新包

1. 从GitHub下载或克隆这个仓库，得到正常的本地仓库目录。
2. 解压本次更新ZIP，把里面的文件和文件夹按整理后的路径复制到仓库根目录，覆盖同名文档。
3. 远端旧根目录的`DICO_NLI_FINAL_EXPERIMENT_RECORD.md`已移到[docs/history](history/DICO_NLI_FINAL_EXPERIMENT_RECORD.md)，`train.py`和`evaluate.py`已移到[原0.9248独立目录](../experiments/deberta_original/README.md)。确认新文件已复制后，从克隆目录移除这三个旧根目录文件，以免留下两份；其他远端文件不要随意删除。更新包旁的`remote_deletions.csv`记录这次需要移除的旧路径。
4. 用GitHub Desktop或Git查看新增、修改和移动，再提交并推送。**不要把整个ZIP当成项目文件上传，也不要把模型和缓存目录拖进去。**
5. 从官方渠道下载Train、Dev及Dev模板，按 [data/README](../data/README.md) 放置；正式训练仍由使用者自行运行。

本次整理只准备本地文件，没有推送远端。更新包旁的文件清单区分新增、修改和排除内容；清单中的远端版本用于说明对比基准，不能代替未来再次拉取远端更新。
