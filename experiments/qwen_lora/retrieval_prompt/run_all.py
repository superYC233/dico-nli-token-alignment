"""在 PyCharm 右键运行这个文件：准备示例 → 三组预测 → 汇总。"""
from prepare_examples import prepare
from evaluate_retrieval import evaluate
from summarize_results import summarize


if __name__ == '__main__':
    prepare()
    evaluate()
    summarize()

