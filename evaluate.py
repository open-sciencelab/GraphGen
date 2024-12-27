import os
import json
import argparse
import torch
import pandas as pd
from dotenv import load_dotenv
from models import LengthEvaluator, MTLDEvaluator, RewardEvaluator, TextPair, UniEvaluator
from utils import logger, set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))
set_logger(os.path.join(sys_path, "cache", "logs", "evaluate.log"))

load_dotenv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default='cache/data', help='folder to load data')
    parser.add_argument('--output', type=str, default='cache/output', help='path to save output')

    parser.add_argument('--tokenizer', type=str, default='cl100k_base', help='tokenizer name')
    parser.add_argument('--reward', type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2', help='Comma-separated list of reward models')
    parser.add_argument('--uni', type=str, default='MingZhong/unieval-sum', help='uni model name')

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        raise ValueError(f"Folder {args.folder} does not exist")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    reward_models = args.reward.split(',')

    logger.info("Evaluators loading")

    length_evaluator = LengthEvaluator(
        tokenizer_name=args.tokenizer
    )
    mtld_evaluator = MTLDEvaluator()

    reward_evaluators = []
    for reward_name in reward_models:
        reward_evaluators.append({
            'reward_name': reward_name.split('/')[-1],
            'evaluator': RewardEvaluator(
                reward_name=reward_name
            )
        })

    uni_evaluator = UniEvaluator(
        model_name=args.uni
    )

    logger.info("Evaluators loaded")

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

            reward_scores = []
            for reward_evaluator in reward_evaluators:
                reward_scores.append({
                    'reward_name': reward_evaluator['reward_name'],
                    'score': reward_evaluator['evaluator'].get_average_score(data)
                })
                logger.info(f"{reward_evaluator['reward_name']} scores: {reward_scores[-1]['score']}")

            uni_naturelness_scores = uni_evaluator.get_average_score(data, 'naturalness')
            logger.info(f"Uni naturalness scores: {uni_naturelness_scores}")

            uni_coherence_scores = uni_evaluator.get_average_score(data, 'coherence')
            logger.info(f"Uni coherence scores: {uni_coherence_scores}")

            uni_understandability_scores = uni_evaluator.get_average_score(data, 'understandability')
            logger.info(f"Uni understandability scores: {uni_understandability_scores}")

            result = {
                'file': file,
                'number': len(data),
                'length': length_scores,
                'mtld': mtld_scores,
                'uni_naturalness': uni_naturelness_scores,
                'uni_coherence': uni_coherence_scores,
                'uni_understandability': uni_understandability_scores
            }
            for reward_score in reward_scores:
                result[reward_score['reward_name']] = reward_score['score']

            results.append(result)

            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        
    results = pd.DataFrame(results)
        
    results.to_csv(os.path.join(args.output, 'evaluation.csv'), index=False)
