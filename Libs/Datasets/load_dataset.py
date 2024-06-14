import math
import os.path as osp
from collections import deque
from typing import Any, Optional

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from Libs.Datasets.upfd import UPFD
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (from_networkx, get_self_loop_attr, scatter,
                                   to_edge_index, to_torch_csr_tensor)
from torch_geometric.utils.convert import to_networkx

import math
import numpy as np
import networkx as nx
from collections import deque

root = '/home/hwxu/Projects/Research/ECML-PKDD/'


class EchoChamberPartitioner(BaseTransform):
    def __init__(self, K=3) -> None:
        self.K = K

    def evaluate_addition(self, G, L, v, candidate_chamber):
        cohesion_score = 0
        separation_score = 0
        balance_score = 0

        chamber_nodes = [node for node, chamber in L.items() if chamber == candidate_chamber]
        external_nodes = [node for node, chamber in L.items() if chamber != candidate_chamber and chamber is not None]
        chamber_size = len(chamber_nodes)

        # 计算内聚性得分
        for u in chamber_nodes:
            if G.has_edge(u, v):
                cohesion_score += 1

        # 计算分离度得分
        for u in external_nodes:
            if G.has_edge(u, v):
                separation_score += 1

        # 计算平衡性得分
        balance_score = 1 / (chamber_size + 1) if chamber_size > 0 else 1

        # 综合得分
        total_score = cohesion_score - separation_score + balance_score
        return total_score

    def flood_fill_partition(self, G: nx.Graph):
        K = min(self.K, G.number_of_nodes())

        D = {node: False for node in G}  # 节点是否访问标记
        L = {node: None for node in G}  # 节点到回声室的映射
        Q = [deque() for _ in range(K)]  # 分区探索队列

        # 选择种子节点
        centralities = nx.degree_centrality(G)
        sorted_nodes = sorted(centralities, key=centralities.get, reverse=True)
        weighted_seeds = sorted_nodes[:K]

        for k, seed in enumerate(weighted_seeds):
            Q[k].append(seed)
            D[seed] = True
            L[seed] = k

        while any(Q):
            for k in range(K):
                if Q[k]:
                    u = Q[k].popleft()
                    for v in G.neighbors(u):
                        if not D[v]:
                            # 为v评估最佳回声室
                            best_chamber = k  # 默认为当前回声室
                            best_score = -float('inf')
                            for candidate_chamber in range(K):
                                score = self.evaluate_addition(G, L, v, candidate_chamber)
                                if score > best_score:
                                    best_score = score
                                    best_chamber = candidate_chamber
                            
                            # 将节点加入最佳回声室
                            Q[best_chamber].append(v)
                            L[v] = best_chamber
                            D[v] = True

        return L

    def naive_partition(self, G: nx.Graph) -> dict:
        K = min(self.K, G.number_of_nodes())
        nodes = list(G.nodes())
        degrees = np.array([G.degree(node) for node in nodes])

        exp_degrees = np.exp(degrees - np.max(degrees)) / degrees.sum()
        probabilities = exp_degrees / exp_degrees.sum()

        seeds = np.random.choice(
            nodes, size=K, replace=False, p=probabilities)

        # 初始化分区结果，节点访问状态和队列
        partition = {node: None for node in G}
        visited = {node: False for node in G}
        queues = [deque([seed]) for seed in seeds]

        for k, seed in enumerate(seeds):
            partition[seed] = k
            visited[seed] = True

        # 开始分割过程
        while any(queues):
            for k in range(K):
                if queues[k]:
                    current_node = queues[k].popleft()
                    for neighbor in G.neighbors(current_node):
                        if not visited[neighbor]:
                            queues[k].append(neighbor)
                            partition[neighbor] = k
                            visited[neighbor] = True

        # 返回分区结果
        return partition

    def create_subgraphs_from_partition_labels(self, G: nx.Graph, partition_labels: dict) -> Data:
        subgraphs = {}
        for node, partition_number in partition_labels.items():
            if partition_number not in subgraphs:
                subgraphs[partition_number] = nx.Graph()
            subgraphs[partition_number].add_node(node, **G.nodes[node])
            for neighbor in G.neighbors(node):
                if partition_labels[neighbor] == partition_number:
                    subgraphs[partition_number].add_edge(node, neighbor)
        # 执行1-Hop Overlapping处理
        for node, partition_number in partition_labels.items():
            for neighbor in G.neighbors(node):
                if partition_labels[neighbor] != partition_number:
                    # 将邻居节点添加到当前子图中，实现1-Hop Overlapping
                    subgraphs[partition_number].add_node(
                        neighbor, **G.nodes[neighbor])
                    # 添加连接当前节点和邻居节点的边
                    subgraphs[partition_number].add_edge(node, neighbor)

        # 确保所有子图都不为空
        non_empty_subgraphs = [
            subgraph for subgraph in subgraphs.values() if subgraph.number_of_nodes() > 0]
        # 转换为torch_geometric数据格式
        subgraphs = [from_networkx(subgraph)
                     for subgraph in non_empty_subgraphs]
        subgraphs = [T.Compose([T.AddSelfLoops(), T.ToUndirected(), T.AddRandomWalkPE(5)])(
            subgraph) for subgraph in subgraphs]
        graph = from_networkx(G)
        graph['partitions'] = subgraphs
        return graph

    def forward(self, data: Data) -> Data:
        G = to_networkx(data, node_attrs=['x'], graph_attrs=[
                        'y'], to_undirected=True)
        # partition_labels = self.flood_fill_partition(G)
        partition_labels = self.naive_partition(G)
        graph = self.create_subgraphs_from_partition_labels(
            G, partition_labels)
        return graph

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def load_UPFD(name='politifact', feature='bert', batch_size=64, K=3):

    sub_dataset = name
    feature = feature
    path = osp.join(root, 'Input', 'UPFD')
    pre_transform = T.Compose([
        T.ToUndirected(),
        T.AddSelfLoops(),
    ])
    transform = T.Compose([
        EchoChamberPartitioner(K=K),
    ])
    train_dataset = UPFD(path, sub_dataset, feature, 'train', transform=transform,
                         pre_transform=pre_transform,)
    val_dataset = UPFD(path, sub_dataset, feature, 'val', transform=transform,
                       pre_transform=pre_transform,)
    test_dataset = UPFD(path, sub_dataset, feature, 'test', transform=transform,
                        pre_transform=pre_transform,)
    # return train_dataset, val_dataset, test_dataset
    train_loader = DataLoader(train_dataset, batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size,
                             shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
