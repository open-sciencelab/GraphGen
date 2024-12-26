import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from models import LengthEvaluator, MTLDEvaluator, RewardEvaluator, TextPair, UniEvaluator
from utils import logger, set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))
set_logger(os.path.join(sys_path, "cache", "evaluate.log"))

load_dotenv()

length_evaluator = LengthEvaluator(
    tokenizer_name=os.getenv("EVALUATE_TOKENIZER", "cl100k_base")
)
mtld_evaluator = MTLDEvaluator()

reward_evaluator = RewardEvaluator(
    reward_name=os.getenv("EVALUATE_REWARD", "OpenAssistant/reward-model-deberta-v3-large-v2")
)

uni_evaluator = UniEvaluator(
    model_name=os.getenv("EVALUATE_UNI", "MingZhong/unieval-sum")
)

logger.info("Evaluators loaded")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='cache/data', help='folder to load data')
    parser.add_argument('--output', type=str, default='cache/output', help='path to save output')

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        raise ValueError(f"Folder {args.folder} does not exist")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    results = []

    logger.info(f"Data loaded from {args.folder}")
    for file in os.listdir(args.folder):
        if file.endswith('.json'):
            logger.info(f"Processing {file}")
            data = json.load(open(os.path.join(args.folder, file)))

            data = [TextPair(
                question=data[key]['question'],
                answer=data[key]['answer']
            ) for key in data]
            length_scores = length_evaluator.get_average_score(data)
            logger.info(f"Length scores: {length_scores}")

            mtld_scores = mtld_evaluator.get_average_score(data)
            logger.info(f"MTLD scores: {mtld_scores}")

            reward_scores = reward_evaluator.get_average_score(data)
            logger.info(f"Reward scores: {reward_scores}")

            uni_naturelness_scores = uni_evaluator.get_average_score(data, 'naturalness')
            logger.info(f"Uni naturalness scores: {uni_naturelness_scores}")

            uni_coherence_scores = uni_evaluator.get_average_score(data, 'coherence')
            logger.info(f"Uni coherence scores: {uni_coherence_scores}")

            uni_understandability_scores = uni_evaluator.get_average_score(data, 'understandability')
            logger.info(f"Uni understandability scores: {uni_understandability_scores}")

            results.append({
                'file': file,
                'length': length_scores,
                'mtld': mtld_scores,
                'reward': reward_scores,
                'uni_naturalness': uni_naturelness_scores,
                'uni_coherence': uni_coherence_scores,
                'uni_understandability': uni_understandability_scores
            })
        
    results = pd.DataFrame(results)
        
    results.to_csv(os.path.join(args.output, 'evaluation.csv'), index=False)
