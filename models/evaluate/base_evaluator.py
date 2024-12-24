from dataclasses import dataclass

# Helpfulness、Honesty、Harmlessness
@dataclass
class BaseEvaluator:
    def evaluate(self, text: str) -> float:
        """
        Evaluate the text and return a score.
        """
        raise NotImplementedError



# 指令跟随难度（Instruction-Following Difficulty，IFD）是一个用于筛选具有增强LLM指令调优潜力的数据样例的指标
# 评估SFT数据质量的一些指标包括Length、Rewardscore、Perplexity、MTLD、KNN-i、Unieval-naturalness、Unieval-coherence、Unieval-understandability等
# https://github.com/maszhongming/UniEval
# 1. 使用打分模型对数据进行打分