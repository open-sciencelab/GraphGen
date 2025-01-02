from dataclasses import dataclass

from models.strategy.base_strategy import BaseStrategy


@dataclass
class TraverseStrategy(BaseStrategy):
    # 单向拓展还是双向拓展
    bidirectional: bool = False
    # 每个方向拓展的最大边数
    max_width: int = 5
    # 最长token数
    max_tokens: int = 1024
    # 最大边数和最大token数方法中选择一个生效
    expand_method: str = "max_width" # "max_width" or "max_tokens"
    # 每个方向拓展的最大深度
    max_depth: int = 1
    # 同一层中选边的策略（如果是双向拓展，同一层指的是两边连接的边的集合）
    edge_sampling: str = "max_loss" # "max_loss" or "min_loss" or "random"
    # 孤立节点的处理策略
    isolated_node_strategy: str = "add" # "add" or "ignore"
