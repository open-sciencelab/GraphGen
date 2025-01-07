import re

import pandas as pd
from tqdm import tqdm
from models import Tokenizer
from utils.log import parse_log
import plotly.express as px

def analyse_log(log_info: dict) -> dict:
    """
    Analyse the log information.

    :param log_info
    :return
    """
    logs = []
    for line in log_info:
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)', line)
        if not match:
            continue
        timestamp, logger_name, log_level, message = match.groups()
        logs.append({
            'timestamp': timestamp,
            'logger_name': logger_name,
            'log_level': log_level,
            'message': message
        })

    logs = [log_item for log_item in logs if log_item['log_level'] == 'INFO']

    for i in range(len(logs)):
        match = re.search(r'(\d+) nodes and (\d+) edges processed', logs[i]['message'])
        if match:
            break

    logs = logs[i:]
    assert len(logs) % 3 == 0

    # 每三个为一组
    results = []
    for i in range(0, len(logs), 3):
        results.append({
            'pre_length': int(logs[i + 1]['message'].split(': ')[1]),
            'question': logs[i + 2]['message'].split("Question: ")[1].split("Answer")[0].strip(),
            'answer': logs[i + 2]['message'].split("Answer: ")[1].strip()
        })

    return results

async def plot_rephrase_process(stats: list[dict]):
    """
    Plot a line revealing the length change of rephrasing.

    :return
    """

    df = pd.DataFrame(stats)

    df['count'] = 1
    df = df.groupby(['pre_length', 'post_length']).sum().reset_index()

    fig = px.scatter(df, x="pre_length", y="post_length", size="count", color="count", hover_name="count")
    fig.show()



if __name__ == "__main__":
    log = parse_log('/home/PJLAB/chenzihong/Project/graphgen/cache/logs/graphgen.log')
    data = analyse_log(log)

    tokenizer = Tokenizer(model_name='cl100k_base')

    for item in tqdm(data):
        item['post_length'] = len(tokenizer.encode_string(item['answer']))

    import asyncio
    asyncio.run(plot_rephrase_process(data))
