from dataclasses import dataclass, field

from models.strategy.base_strategy import BaseStrategy


@dataclass
class TraverseStrategy(BaseStrategy):
    # 最大边数和最大token数方法中选择一个生效
    expand_method: str = "max_tokens" # "max_width" or "max_tokens"
    # 单向拓展还是双向拓展
    bidirectional: bool = True
    # 每个方向拓展的最大边数
    max_extra_edges: int = 5
    # 最长token数
    max_tokens: int = 256
    # 每个方向拓展的最大深度
    max_depth: int = 2
    # 同一层中选边的策略（如果是双向拓展，同一层指的是两边连接的边的集合）
    edge_sampling: str = "max_loss" # "max_loss" or "min_loss" or "random"
    # 孤立节点的处理策略
    isolated_node_strategy: str = "add" # "add" or "ignore"
    # 难度顺序 ["easy", "medium", "hard"], ["hard", "medium", "easy"], ["medium", "medium", "medium"]
    difficulty_order: list = field(default_factory=lambda: ["medium", "medium", "medium"])

    def to_yaml(self):
        return {
            "traverse_strategy": {
                "expand_method": self.expand_method,
                "bidirectional": self.bidirectional,
                "max_extra_edges": self.max_extra_edges,
                "max_tokens": self.max_tokens,
                "max_depth": self.max_depth,
                "edge_sampling": self.edge_sampling,
                "isolated_node_strategy": self.isolated_node_strategy,
                "difficulty_order": self.difficulty_order
            }
        }
