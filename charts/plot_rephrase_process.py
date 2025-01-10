import re

import pandas as pd
from tqdm import tqdm
from models import Tokenizer
from utils.log import parse_log
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

def analyse_log(log_info: dict) -> list:
    """
    Analyse the log information.

    :param log_info
    :return
    """
    logs = []
    current_message = None

    for line in log_info:
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)', line)
        if match:
            if current_message:
                logs.append(current_message)

            timestamp, logger_name, log_level, message = match.groups()
            current_message = {
                'timestamp': timestamp,
                'logger_name': logger_name,
                'log_level': log_level,
                'message': message
            }
        elif current_message:
            current_message['message'] += '\n' + line.strip()

    if current_message:
        logs.append(current_message)

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

def plot_pre_length_distribution(stats: list[dict]):
    """
    Plot the distribution of pre-length.

    :return fig
    """

    # 使用传入的stats参数而不是全局的data
    if not stats:
        return go.Figure()

    # 计算最大长度并确定区间
    max_length = max(item['pre_length'] for item in stats)
    bin_size = 50
    max_length = ((max_length // bin_size) + 1) * bin_size

    # 使用defaultdict避免键不存在的检查
    length_distribution = defaultdict(int)

    # 一次遍历完成所有统计
    for item in stats:
        bin_start = (item['pre_length'] // bin_size) * bin_size
        bin_key = f"{bin_start}-{bin_start + bin_size}"
        length_distribution[bin_key] += 1

    # 转换为排序后的列表以保持区间顺序
    sorted_bins = sorted(length_distribution.keys(),
                         key=lambda x: int(x.split('-')[0]))

    # 创建图表
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_bins,
            y=[length_distribution[bin_] for bin_ in sorted_bins],
            text=[length_distribution[bin_] for bin_ in sorted_bins],
            textposition='auto',
        )
    ])

    # 设置图表布局
    fig.update_layout(
        title='Distribution of Pre-Length',
        xaxis_title='Length Range',
        yaxis_title='Count',
        bargap=0.2,
        showlegend=False
    )

    # 如果数据点过多，优化x轴标签显示
    if len(sorted_bins) > 10:
        fig.update_layout(
            xaxis={
                'tickangle': 45,
                'tickmode': 'array',
                'ticktext': sorted_bins[::2],  # 每隔一个显示标签
                'tickvals': list(range(len(sorted_bins)))[::2]
            }
        )

    return fig

if __name__ == "__main__":
    log = parse_log('/home/PJLAB/chenzihong/Project/graphgen/cache/logs/graphgen.log')
    data = analyse_log(log)
    tokenizer = Tokenizer(model_name='cl100k_base')

    for item in tqdm(data):
        item['post_length'] = len(tokenizer.encode_string(item['answer']))

    import asyncio
    asyncio.run(plot_rephrase_process(data))
