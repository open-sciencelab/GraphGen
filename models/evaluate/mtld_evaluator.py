from dataclasses import  dataclass
from base_evaluator import BaseEvaluator
from utils import detect_main_language, NLTKHelper

nltk_helper = NLTKHelper()

@dataclass
class MTLDEvaluator(BaseEvaluator):
    """
    衡量文本词汇多样性的指标
    """
    def evaluate(self, text: str) -> float:
        return self._calculate_mtld_score(text)

    def _calculate_mtld_score(self, text: str, threshold=0.72) -> float:
        """
        计算MTLD (向前和向后的平均值)

        min is 1.0
        higher is better
        """
        if not text or not text.strip():
            return 0.0

        lang = "chinese" if detect_main_language(text) == "zh" else "english"
        tokens = nltk_helper.word_tokenize(text, lang)

        stopwords = nltk_helper.get_stopwords(lang)
        filtered_tokens = [word for word in tokens if word not in stopwords]
        filtered_tokens = [word for word in filtered_tokens if word.isalnum()]

        if not filtered_tokens:
            return 0

        # 计算向前的MTLD
        forward_factors = self._compute_factors(filtered_tokens, threshold)

        # 计算向后的MTLD
        backward_factors = self._compute_factors(filtered_tokens[::-1], threshold)

        # 取平均值
        return (forward_factors + backward_factors) / 2

    def _compute_factors(self, tokens: list, threshold: float) -> float:
        factors = 0
        current_segment = []
        unique_words = set()

        for token in tokens:
            current_segment.append(token)
            unique_words.add(token)
            ttr = len(unique_words) / len(current_segment)

            if ttr <= threshold:
                factors += 1
                current_segment = []
                unique_words = set()

        # 处理最后一个不完整片段
        if current_segment:
            ttr = len(unique_words) / len(current_segment)
            if ttr <= threshold:
                factors += 1
            else:
                factors += (1 - (ttr - threshold) / (1 - threshold))

        return len(tokens) / factors if factors > 0 else len(tokens)
