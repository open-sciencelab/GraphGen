from models import Chunk
from typing import List
from dataclasses import dataclass


@dataclass
class GraphRag:

    def insert(self, chunks: List[Chunk]):
        pass

    def async_insert(self):
        """

        :return:
        """

        # 对文档内容分块
        # 从新插入的块中提取实体和关系
        # 将文档和块分别存入各自的数据库中
        pass

    def query(self):
        pass

    def delete_by_entity(self):
        pass



if __name__ == "__main__":
    graph_rag = GraphRag()
    graph_rag.insert()
