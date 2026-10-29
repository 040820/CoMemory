"""证据支撑模块参数：算法预算集中定义，运行结果保存实际取值。"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class EvidenceSupportConfig:
    """初始工程参数，未经真实数据调优；排名分数均不是可信度。"""

    structured_k: int = 12             # 每个锚点的精确槽位/事件候选数。
    semantic_k: int = 12               # 每个语义查询独立保留的 Fact 数。
    max_hops: int = 2                  # BFS 的最大关系跳数；0 表示关闭。
    max_graph_nodes: int = 64          # 每次取证允许访问的 (锚点, Fact) 数。
    max_neighbors: int = 12            # 单节点邻接预算，抑制大分组扩散。
    max_question_seeds: int = 2        # 无显式 Fact 时由语义排名选出的图检索种子。
    rrf_k: int = 60                    # RRF 的排名平滑常数。
    reserve_per_route: int = 1         # 融合截断前尽量保留各路独有候选。
    max_candidate_facts: int = 48      # 融合后的 Fact 上限，包含直接锚点。
    max_chunks: int = 12               # 整次审查的不同原文来源上限。
    max_facts_per_chunk: int = 4       # 一个来源给 Judge 的结构化事实数。
    embedding_batch_size: int = 32    # 编码请求批大小，不是最终证据数。
    max_embedding_chars: int = 1000   # 每条语义输入的字符预算，原文不改写。
    min_cosine: float = 0.0           # 余弦候选下限，不是支持度阈值。
    max_semantic_queries: int = 7     # 一次审查中语义查询总数，包含双端查询。
    max_rounds: int = 3               # Judge 轮数上限；每轮一次模型请求。
    max_followups_per_round: int = 2  # 每轮最多接纳两个明确的缺口查询。
    max_prompt_bytes: int = 12000     # 复用 Judge 的 UTF-8 提示词字节预算。
    max_answer_tokens: int | None = None  # None：取消新模块回答的人为输出 Token 上限。

    def __post_init__(self):
        """服务于入口校验：拒绝布尔整数、越界值和非有限阈值，避免隐式降级。"""
        for name, value in asdict(self).items():
            if name == 'max_answer_tokens' and value is None:
                continue
            elif name == 'min_cosine':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError('min_cosine 必须是有限的 [-1,1] 数值')
            elif type(value) is not int or value < (0 if name in ('max_hops', 'reserve_per_route') else 1):
                raise ValueError(f'{name} 必须是有效整数预算')
        if not 2 <= self.max_chunks <= 32 or self.max_candidate_facts < 2:
            raise ValueError('证据预算必须容纳双端来源，max_chunks 最大为 32')
        if not 2 <= self.max_facts_per_chunk <= 4:
            raise ValueError('每 Chunk 展示 2–4 个 Fact，兼容现有 Judge 的结构视图预算')
        if self.max_hops > 3 or self.max_rounds > 5 or self.max_followups_per_round > 2:
            raise ValueError('图扩展最多 3 跳，Judge 最多 5 轮，每轮最多 2 个缺口查询')
        if self.max_question_seeds > 4:
            raise ValueError('问答自动图种子最多为 4 个')
        if not 6000 <= self.max_prompt_bytes <= 12000:
            raise ValueError('Judge 提示词预算必须在 6000–12000 字节之间')

    def to_dict(self):
        """服务于复现：输出全部实际超参数，不混入任何模型估计分数。"""
        return asdict(self)
