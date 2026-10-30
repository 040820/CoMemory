"""
查询结果可视化模块。

该模块将检索返回的三元组（triples）渲染成一张 PNG 图片，图片自上而下分为三块：
    1. 顶部：用户问题（Question）
    2. 中部左侧：知识图谱（节点 = 实体，边 = 关系，颜色深浅 = 相关性）
    3. 中部右侧：图例（Legend）
    4. 底部：三元组相关性排名表（Table）

典型用法::

    visualizer = QueryResultVisualizer(config)
    visualizer.create_visualization(question, triples, output_path)
"""

import sys
import os

# 将项目根目录加入模块搜索路径，保证在任意工作目录下都能 import 到项目包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd
from init.logger import logger

# 全局字体设置：优先使用衬线字体，避免中文/特殊字符显示为方块
plt.rcParams['font.family'] = ['Liberation Serif', 'DejaVu Serif', 'serif']
# 全局基础字号
plt.rcParams['font.size'] = 10


class QueryResultVisualizer:
    """查询结果可视化器：把检索到的三元组画成知识图谱 + 排名表。"""

    def __init__(self, config):
        """
        初始化可视化器。

        Args:
            config: 项目配置对象，需要包含 ``working_dir``（输出目录），
                    以及 ``config.retriever.top_visual``（最多可视化多少条三元组）。
        """
        self.config = config
        # 输出目录：优先取 config.working_dir，缺省为 ./results
        self.output_dir = getattr(config, 'working_dir', './results')
        logger.info("Query Result Visualizer initialized")

    def create_visualization(self, question: str, triples: List[Dict[str, Any]],
                             output_path: str = None) -> str:
        """
        生成查询结果的可视化图片。

        Args:
            question: 用户的查询问题，会显示在图片顶部。
            triples: 三元组列表，每项形如
                     ``{'src': ..., 'relation': ..., 'tgt': ..., 'final_score': ...}``。
            output_path: 输出图片路径；为 None 时自动生成到 ``self.output_dir``。

        Returns:
            生成的图片路径；若无数据可画则返回 None。
        """
        # 没有三元组时无法绘图，记录警告并直接返回
        if not triples:
            logger.warning("No triples provided for visualization")
            return None

        # 最多可视化的三元组数量，避免图过于拥挤（默认 10）
        top_visual = getattr(self.config.retriever, 'top_visual', 10)

        # 只取相关性最高的前 top_visual 条
        visualization_triples = triples[:top_visual]

        # 1) 构建有向图结构
        G = self._create_network_graph(visualization_triples)

        # 2) 绘制包含问题、图谱、图例和表格的整幅图
        fig = self._create_figure(question, G, visualization_triples)

        # 3) 未指定输出路径时，按节点数自动命名
        if output_path is None:
            output_path = os.path.join(
                self.output_dir,
                f"query_visualization_{len(visualization_triples)}_nodes.png"
            )

        # 保存为高分辨率 PNG（dpi=300），白色背景、无边框
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        # 关闭 figure，释放 matplotlib 内存，防止批量生成时内存泄漏
        plt.close()

        logger.info(f"Visualization saved to: {output_path}")
        return output_path

    def _create_network_graph(self, triples: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        将三元组列表转换为 networkx 有向图。

        Args:
            triples: 三元组列表。

        Returns:
            以实体为节点、关系为边的有向图 ``nx.DiGraph``。
        """
        G = nx.DiGraph()

        for i, triple in enumerate(triples):
            # 防御性检查：非字典元素无法解析，跳过并记录错误
            if not isinstance(triple, dict):
                logger.error(f"Triple {i} is not a dictionary: {type(triple)} - {triple}")
                continue

            # 取出源实体、目标实体、关系和得分，缺失时使用兜底默认值
            src = triple.get('src', f'Node_{i}_src')
            tgt = triple.get('tgt', f'Node_{i}_tgt')
            relation = triple.get('relation', 'relates_to')
            score = triple.get('final_score', 0.0)

            # numpy 标量无法直接参与某些序列化/比较，转换为原生 Python 类型
            if hasattr(score, 'item'):
                score = score.item()

            # 只有首次出现时才添加节点（避免覆盖已有节点的属性）
            if not G.has_node(src):
                G.add_node(src, node_type='entity', score=score)
            if not G.has_node(tgt):
                G.add_node(tgt, node_type='entity', score=score)

            # 添加带关系名和得分的边
            G.add_edge(src, tgt, relation=relation, score=score)

        return G

    def _create_figure(self, question: str, G: nx.DiGraph,
                       triples: List[Dict[str, Any]]) -> plt.Figure:
        """
        创建整幅布局：使用 GridSpec 将画布划分为 3 行 2 列。

        布局示意::

            +--------------------------+
            |        Question          |  第 0 行，跨两列
            +--------------------+-----+
            |       Graph        |Legend|  第 1 行，左图右图例
            +--------------------+-----+
            |         Table            |  第 2 行，跨两列
            +--------------------------+

        Args:
            question: 查询问题文本。
            G: 已构建的有向图。
            triples: 参与展示的三元组。

        Returns:
            matplotlib Figure 对象。
        """
        # 整幅图尺寸（英寸）
        fig = plt.figure(figsize=(20, 14))

        # 3 行 2 列网格：高度比 0.5:2:1，宽度比 3:1
        gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 2, 1], width_ratios=[3, 1],
                              hspace=0.3, wspace=0.2)

        # 顶部：问题文本，横跨两列
        ax_question = fig.add_subplot(gs[0, :])
        self._add_question_text(ax_question, question)

        # 中部左侧：主知识图谱
        ax_graph = fig.add_subplot(gs[1, 0])
        self._draw_graph(ax_graph, G, triples)

        # 中部右侧：图例
        ax_legend = fig.add_subplot(gs[1, 1])
        self._add_legend(ax_legend, G)

        # 底部：三元组排名表，横跨两列
        ax_table = fig.add_subplot(gs[2, :])
        self._add_triples_table(ax_table, triples)

        return fig

    def _add_question_text(self, ax, question: str):
        """在图片顶部绘制带圆角边框的问题文本框。"""
        # 关闭坐标轴刻度与边框
        ax.axis('off')

        # 圆角矩形框：(x, y) 起点、宽、高，以及边框样式
        box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                             boxstyle="round,pad=0.02",
                             facecolor='#f0f8ff',      # 淡蓝色填充
                             edgecolor='#4169e1',      # 皇家蓝边框
                             linewidth=2)
        ax.add_patch(box)

        # 居中显示问题文本（wrap=True 允许自动换行）
        ax.text(0.5, 0.5, f"Question: {question}",
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='#2c3e50', wrap=True)

    def _draw_graph(self, ax, G: nx.DiGraph, triples: List[Dict[str, Any]]):
        """
        在指定坐标轴上绘制主知识图谱：
        节点颜色按相关分数分三档，边上标注关系名称。
        """
        # 空图时显示占位文字
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, "No nodes to display", ha='center', va='center')
            ax.axis('off')
            return

        # 使用弹簧布局计算节点位置；seed 固定保证每次生成结果一致
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # 获取每个节点的相关性分数
        node_scores = [G.nodes[node].get('score', 0.0) for node in G.nodes()]

        # 将分数归一化到 [0, 1]，用于映射颜色；若所有分数相同则统一取 0.5
        if node_scores:
            min_score, max_score = min(node_scores), max(node_scores)
            if max_score > min_score:
                normalized_scores = [(score - min_score) / (max_score - min_score)
                                     for score in node_scores]
            else:
                normalized_scores = [0.5] * len(node_scores)
        else:
            normalized_scores = [0.5] * len(G.nodes())

        # 根据归一化分数将节点分为三档颜色
        node_colors = []
        for score in normalized_scores:
            if score < 0.33:
                node_colors.append('#ADD8E6')   # 浅蓝：低相关
            elif score < 0.66:
                node_colors.append('#90EE90')   # 浅绿：中相关
            else:
                node_colors.append('#FFB6C1')   # 浅粉：高相关

        # 绘制节点：尺寸、透明度、黑色描边
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=1500, alpha=0.8,
                               edgecolors='black', linewidths=2)

        # 绘制带箭头的边；arc3 弧度让双向/多条边不至于重叠
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#666666',
                               arrows=True, arrowsize=20, arrowstyle='->',
                               connectionstyle="arc3,rad=0.1", alpha=0.7)

        # 在边上标注关系名称
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8,
                                     font_color='#2c3e50', font_weight='bold')

        # 在节点中心标注实体名称
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold',
                                font_color='black')

        # 设置子图标题并隐藏坐标轴
        ax.set_title("Knowledge Graph Visualization", fontsize=16, fontweight='bold',
                     color='#2c3e50', pad=20)
        ax.axis('off')

    def _add_legend(self, ax, G: nx.DiGraph):
        """绘制图例：说明三种节点颜色代表的相关性等级以及边代表关系。"""
        ax.axis('off')

        # 图例项：三个圆点（低/中/高相关）+ 一个箭头（关系）
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6',
                       markersize=10, label='Low Relevance'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90',
                       markersize=10, label='Medium Relevance'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFB6C1',
                       markersize=10, label='High Relevance'),
            plt.Line2D([0], [0], marker='>', color='#666666', markersize=8,
                       label='Relationship')
        ]

        ax.legend(handles=legend_elements, loc='center', fontsize=10,
                  title='Legend', title_fontsize=12, frameon=True,
                  fancybox=True, shadow=True)

        ax.set_title("Legend", fontsize=14, fontweight='bold', color='#2c3e50')

    def _add_triples_table(self, ax, triples: List[Dict[str, Any]]):
        """在图片底部绘制三元组相关性排名表（最多展示前 5 条）。"""
        # 无数据时显示占位文字
        if not triples:
            ax.text(0.5, 0.5, "No triples to display", ha='center', va='center')
            ax.axis('off')
            return

        # 表格只展示前 5 条，保证可读性
        table_triples = triples[:5]

        # 逐条整理成表格行数据
        table_data = []
        for i, triple in enumerate(table_triples, 1):
            # 防御性检查：非字典元素无法解析，跳过并记录错误
            if not isinstance(triple, dict):
                logger.error(f"Triple {i} is not a dictionary: {type(triple)} - {triple}")
                continue

            src = triple.get('src', 'N/A')
            relation = triple.get('relation', 'N/A')
            tgt = triple.get('tgt', 'N/A')
            score = triple.get('final_score', 0.0)

            # 将 numpy 标量转换为原生 Python 类型，便于格式化输出
            if hasattr(score, 'item'):
                score = score.item()

            table_data.append([
                f"{i}",                 # 排名
                str(src),               # 源实体
                str(relation),          # 关系
                str(tgt),               # 目标实体
                f"{float(score):.4f}"   # 相关性分数，保留 4 位小数
            ])

        # 创建表格：cellText 为数据，colLabels 为表头，bbox 撑满整个子图区域
        headers = ['Rank', 'Source', 'Relation', 'Target', 'Relevance Score']
        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])

        # 关闭自动字号并固定为 9，同时纵向拉伸行高
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # 表头行：深色背景 + 白色粗体文字
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 数据行：奇偶行交替配色（斑马纹），提升可读性
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('white')

        # 子图标题并隐藏坐标轴
        ax.set_title("Triples Ranking by Relevance", fontsize=14, fontweight='bold',
                     color='#2c3e50', pad=20)
        ax.axis('off')
