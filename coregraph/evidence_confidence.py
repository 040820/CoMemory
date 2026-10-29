"""证据置信度模块：多路一致性、来源可信度与带符号信念的纯函数实现。

这个模块回答的问题不是“这条材料相关吗”，而是“这组材料是否足以支持一个
关系判断”。它把证据分成五层，每层都可以单独审计和消融：

    1. 检索一致性：同一条证据被多少个独立检索路线命中，路线越可靠、排名越靠前，
       贡献越大；多条路线用 Noisy-OR 合并，而不是直接把分数相加。
    2. 来源可信度：原文是否真实存在、引文是否连续、字段是否完整、来源是显式
       原文还是派生断言。这里是确定性规则，不调用模型。
    3. 独立性折扣：同一 Chunk、同一引文派生出的多条事实只算一份证据，避免
       “一句话被抽成三条事实”被误当成三份独立支持。
    4. 带符号信念：支持与反驳分开累计，再计算冲突质量 K 和不确定性 u。
    5. 双侧覆盖：关系判断必须同时看到旧、新两侧的证据，防止只验证新事实就
       判成 SUFFICIENT。

P1 只提供纯函数和默认参数：不读取图、不调用模型、不修改证据或记忆状态。
默认权重是工程初值，不是概率；任何下游决策都必须先经过真实样本校准。
输出中的断言角色来自模型，不等同于独立语义验证。

术语约定：
    route      一条检索路线，例如 structured / bm25 / semantic / graph / summary。
    strength   某条路线对某条证据的命中强度，取值 [0,1]。
    cred       来源可信度，由确定性特征加权得到。
    b / d      支持信念 / 反驳信念，分别由支持证据和反驳证据合并得到。
    K          冲突质量，b 与 d 同时较高时变大。
    u          不确定性，b 与 d 都低时变大。
    bilateral  旧、新两侧覆盖的较小值；只有一侧证据时为 0。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, prod
from typing import Any

# ---------------------------------------------------------------------------
# 角色与路线常量
# ---------------------------------------------------------------------------
# 角色来自 EvidenceJudge 的 evidence_roles；这里只做权重映射，不重新判断语义。
ROLES = ('supports', 'refutes', 'explains_change', 'background', 'uncertain')
# 只有这三类角色参与支持/反驳和双侧覆盖；background 只是背景，uncertain 不投票。
EVIDENCE_ROLES = frozenset(('supports', 'refutes', 'explains_change'))
SIDES = ('old', 'new')

# ---------------------------------------------------------------------------
# 默认参数（工程初值，不是校准后的概率）
# ---------------------------------------------------------------------------
# 路线可靠性：结构化槽位最精确，BM25 补充人名/日期/数字，语义召回噪声较大，
# 图 BFS 只解释“怎么找到”，摘要只是线索。数值由用户确认为起步值。
DEFAULT_ROUTE_WEIGHTS = {
    'structured': 0.35,
    'bm25': 0.30,
    'semantic': 0.20,
    'graph': 0.15,
    'summary': 0.05,
}

# 来源可信度：全部是确定性特征，不依赖模型自评。
# exact_quote 与 inexact_quote 互斥：引文必须存在，只是匹配精度不同。
# llm_self_confidence 默认 0.0，因为当前 LLM 自评分没有做过校准，不能当概率用。
DEFAULT_CREDIBILITY_WEIGHTS = {
    'source_trace': 0.30,
    'exact_quote': 0.20,
    'inexact_quote': 0.08,
    'field_completeness': 0.20,
    'source_type': 0.15,
    'alignment': 0.10,
    'has_time': 0.03,
    'has_quantity': 0.02,
    'llm_self_confidence': 0.00,
}

# 角色权重：supports / refutes / explains_change 都直接参与证据票数；
# background 只说明上下文，给低权重且不进入双侧覆盖；uncertain 不投票。
DEFAULT_ROLE_WEIGHTS = {
    'supports': 1.0,
    'refutes': 1.0,
    'explains_change': 1.0,
    'background': 0.2,
    'uncertain': 0.0,
}

# 决策阈值：只用于生成建议状态，不直接修改记忆。
# 这些值没有经过真实样本校准；P0/P3 之后应重新估计并在报告中标注。
DEFAULT_THRESHOLDS = {
    'sufficient_belief': 0.60,     # b 达到该值才可能 SUFFICIENT
    'refute_ceiling': 0.25,        # d 超过该值不允许 SUFFICIENT
    'uncertainty_ceiling': 0.30,   # u 超过该值不允许 SUFFICIENT
    'conflict_mass': 0.35,         # K 超过该值判为 CONFLICT
    'bilateral_floor': 0.50,       # 关系任务要求旧、新两侧覆盖都达到该值
    'uncertainty_defer': 0.50,     # u 超过该值直接 DEFER，而不是继续补查
}


def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    """把外部输入限制在 [low, high]。

    证据来自模型和检索器，不能假设一定合法：非数字、NaN、Inf 都退化为 low，
    避免一个坏分数污染整条证据链。配置校验用 _unit，不用这个函数。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if not isfinite(number):
        return low
    return max(low, min(high, number))


def _unit(value: Any, label: str) -> float:
    """配置校验：必须是 [0,1] 内的有限数；非法值直接报错，不静默修正。

    配置错误应当在实验开始前暴露，而不是把一个 1.5 的权重悄悄裁成 1.0。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label} 必须是数字') from None
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f'{label} 必须在 [0,1] 内')
    return number


def _round(value: float) -> float:
    """统一保留 6 位小数，让重复运行和审计文件稳定可读。"""
    return round(float(value), 6)


# ---------------------------------------------------------------------------
# 参数对象
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceConfidenceConfig:
    """置信度参数；默认值只是工程起步值，未做真实样本校准。

    四组参数各自独立：
        route_weights       每路检索的可靠性；
        credibility_weights 来源可信度的特征权重；
        role_weights        支持/反驳/背景/不确定的投票权重；
        thresholds          从 b/d/u/K 映射到建议状态的门槛。
    """

    # 使用 default_factory 而不是共享可变 dict，避免多个 Config 实例互相污染。
    route_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_ROUTE_WEIGHTS))
    credibility_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CREDIBILITY_WEIGHTS))
    role_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_ROLE_WEIGHTS))
    thresholds: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    # 来源可信度的裁剪区间：0.05 保留最低可审计值，0.98 避免把规则分当绝对真值。
    credibility_min: float = 0.05
    credibility_max: float = 0.98

    def __post_init__(self):
        """校验所有参数并复制为普通 dict，避免外部修改影响已创建配置。"""
        # frozen dataclass 不能直接赋值，用 object.__setattr__ 写入校验后的副本。
        for name in ('route_weights', 'credibility_weights', 'role_weights', 'thresholds'):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f'{name} 必须是映射')
            normalized = {str(key): _unit(value, f'{name}.{key}')
                          for key, value in values.items()}
            object.__setattr__(self, name, normalized)
        # 角色权重必须覆盖全部五种角色，否则 evaluate_candidate 会在运行时才报错。
        missing = set(ROLES) - set(self.role_weights)
        if missing:
            raise ValueError(f'role_weights 缺少角色：{sorted(missing)}')
        low = _unit(self.credibility_min, 'credibility_min')
        high = _unit(self.credibility_max, 'credibility_max')
        if low > high:
            raise ValueError('credibility_min 不能大于 credibility_max')
        object.__setattr__(self, 'credibility_min', low)
        object.__setattr__(self, 'credibility_max', high)

    def to_dict(self) -> dict[str, Any]:
        """导出普通容器，供 manifest 或实验报告保存。

        返回副本而不是内部映射，调用方修改结果不会影响当前配置对象。
        """
        return {
            'route_weights': dict(self.route_weights),
            'credibility_weights': dict(self.credibility_weights),
            'role_weights': dict(self.role_weights),
            'thresholds': dict(self.thresholds),
            'credibility_min': self.credibility_min,
            'credibility_max': self.credibility_max,
        }


def rank_strength(rank: Any, budget: Any) -> float:
    """排名强度：第 1 名为 1，最后一名为 1/K，超出预算为 0。

    公式：
        strength = (budget - rank + 1) / budget

    这样第 1 名和第 20 名不会拿同一个分数。SodaMem 的参与度是“命中即 1”，
    会让排名信息完全丢失；这里刻意保留排名差异。
    """
    try:
        position, limit = int(rank), int(budget)
    except (TypeError, ValueError):
        return 0.0
    if limit < 1 or position < 1 or position > limit:
        return 0.0
    return (limit - position + 1) / limit


def route_confidence_detail(strengths: Mapping[str, Any] | None,
                            weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Noisy-OR 合并多路命中强度；返回分数、逐路贡献与被忽略路线。

    公式：
        c_route = 1 - Π_h (1 - w_h * s_h)

    含义：每一路都是“独立的一次提醒”，多路同时命中时置信度上升；
    单路命中时最多只到该路权重，不会因为反复命中同一路而无限累加。
    """
    weights = DEFAULT_ROUTE_WEIGHTS if weights is None else weights
    if strengths is None:
        strengths = {}
    if not isinstance(strengths, Mapping):
        raise ValueError('route_strengths 必须是映射')
    contributions: dict[str, float] = {}
    ignored: list[str] = []
    for route, raw_strength in strengths.items():
        # 外部输入先裁剪；权重为 0 或未知的路线不参与计算，但保留在审计里。
        strength = _clip(raw_strength)
        weight = _clip(weights.get(str(route), 0.0))
        if weight <= 0.0:
            ignored.append(str(route))
            continue
        if strength <= 0.0:
            continue
        contributions[str(route)] = weight * strength
    # 没有有效贡献时直接返回 0；prod() 对空序列返回 1，需要单独处理。
    score = 0.0 if not contributions else 1.0 - prod(
        1.0 - value for value in contributions.values())
    return {'score': _round(score), 'contributions': contributions, 'ignored_routes': ignored}


def route_confidence(strengths: Mapping[str, Any] | None,
                     weights: Mapping[str, float] | None = None) -> float:
    """只返回 Noisy-OR 分数；需要逐路贡献时调用 *_detail 版本。"""
    return route_confidence_detail(strengths, weights)['score']


def source_credibility_detail(
        features: Mapping[str, Any] | None,
        weights: Mapping[str, float] | None = None,
        *,
        min_score: float = 0.05,
        max_score: float = 0.98) -> dict[str, Any]:
    """确定性来源可信度；quote_match 取 exact / inexact / none。

    公式：
        cred = clip(Σ β_k * f_k, min_score, max_score)

    这里的每一项都来自可机械检查的事实，不读取 LLM 自评分：
        has_source_trace     事实能否定位到 Chunk/原文；
        quote_match          引文是精确连续、归一化后可定位，还是不存在；
        field_completeness   主体/属性/值/作用域/模态是否齐全；
        source_type          显式原文 vs 派生断言；
        alignment_confidence 抽取器或对齐模块给出的分数（可选）；
        has_time / has_quantity  时间、数量是否明确；
        llm_self_confidence  默认权重 0，未校准前不参与。
    """
    weights = DEFAULT_CREDIBILITY_WEIGHTS if weights is None else weights
    features = {} if features is None else features
    if not isinstance(features, Mapping):
        raise ValueError('credibility_features 必须是映射')
    if not 0.0 <= float(min_score) <= float(max_score) <= 1.0:
        raise ValueError('可信度裁剪区间非法')

    contributions: dict[str, float] = {}

    def add(name: str, value: Any) -> None:
        """只记录正贡献；缺失特征记 0，不用默认值虚构可信度。"""
        weight = _clip(weights.get(name, 0.0))
        measured = _clip(value)
        if weight > 0.0 and measured > 0.0:
            contributions[name] = weight * measured

    add('source_trace', features.get('has_source_trace', 0.0))
    # exact 与 inexact 互斥：有引文才加分，匹配精度决定加多少。
    quote_match = str(features.get('quote_match') or 'none')
    if quote_match == 'exact':
        add('exact_quote', 1.0)
    elif quote_match == 'inexact':
        add('inexact_quote', 1.0)
    add('field_completeness', features.get('field_completeness', 0.0))
    add('source_type', features.get('source_type', 0.0))
    # 可选特征：字段为 None 表示“未评估”，不是 0 分，因此直接跳过。
    for name, key in (('alignment', 'alignment_confidence'),
                      ('llm_self_confidence', 'llm_self_confidence')):
        value = features.get(key)
        if value is not None:
            add(name, value)
    # 布尔特征用 1.0 作为测量值，是否加分由权重决定。
    if features.get('has_time'):
        add('has_time', 1.0)
    if features.get('has_quantity'):
        add('has_quantity', 1.0)

    raw = sum(contributions.values())
    # 加减分之和可能超过 1（默认权重合计约 1.02），也可能低到 0；
    # 用 clip 保留“最低可审计值”和“最高不等于绝对真值”的边界。
    score = max(float(min_score), min(float(max_score), raw))
    return {'score': _round(score), 'raw_score': _round(raw),
            'contributions': contributions, 'quote_match': quote_match}


def source_credibility(features: Mapping[str, Any] | None,
                       weights: Mapping[str, float] | None = None,
                       *, min_score: float = 0.05, max_score: float = 0.98) -> float:
    """只返回来源可信度分数；需要逐项贡献时调用 *_detail 版本。"""
    return source_credibility_detail(features, weights, min_score=min_score,
                                     max_score=max_score)['score']


def combine_independent(grouped: Mapping[str, Sequence[float]]) -> float:
    """同一 independence_group 只取最大值，不同组之间使用 Noisy-OR。

    为什么要先分组：
        同一 Chunk 里的同一句话可能被抽成多条 Observation；它们不是独立证据。
        组内取最大值，表示“这一来源最多贡献一份票”；组间才使用 Noisy-OR。
    示例：
        {'g1': [0.8, 0.9], 'g2': [0.5]} -> 1 - (1-0.9)(1-0.5) = 0.95
    """
    if not isinstance(grouped, Mapping):
        raise ValueError('组合输入必须是映射')
    group_scores: list[float] = []
    for values in grouped.values():
        if isinstance(values, (int, float)):
            values = [values]
        # 先裁剪再过滤，避免 NaN/Inf/负值进入 max 或 Noisy-OR。
        candidates = [_clip(value) for value in values if _clip(value) > 0.0]
        if candidates:
            group_scores.append(max(candidates))
    if not group_scores:
        return 0.0
    return 1.0 - prod(1.0 - value for value in group_scores)


def signed_belief(support_groups: Mapping[str, Sequence[float]],
                  refute_groups: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """支持与反驳分别合并，再计算冲突质量与不确定性。

    b = 1 - Π(1 - s_i)       支持证据的 Noisy-OR
    d = 1 - Π(1 - r_i)       反驳证据的 Noisy-OR
    K = b * d                两边都强时冲突质量高
    u = (1-b)(1-d)           两边都弱时不确定性高

    注意：K 不是经过 Dempster 归一化的冲突概率，只是本模块的审计量；
    它用于把“双方都有证据”与“只有单边证据”区分开。
    """
    belief = combine_independent(support_groups)
    disbelief = combine_independent(refute_groups)
    return {
        'belief': belief,
        'disbelief': disbelief,
        'conflict_mass': belief * disbelief,
        'uncertainty': (1.0 - belief) * (1.0 - disbelief),
    }


def bilateral_coverage(
        side_groups: Mapping[str, Mapping[str, Sequence[float]]],
        sides: Sequence[str] = SIDES) -> dict[str, Any]:
    """分别计算旧/新两侧覆盖；缺失的一侧按 0 计，bilateral 取较小值。

    关系判断必须同时看到旧、新两端的证据。若只给新事实找到材料，
    coverage_new 可能很高，但 coverage_old 为 0，bilateral 也就是 0，
    从而阻止“只验证新事实就判 SUFFICIENT”的任务偏移。
    """
    if not isinstance(side_groups, Mapping):
        raise ValueError('side_groups 必须是映射')
    # 先按期望的两侧初始化；缺失的一侧 combine_independent({}) 返回 0。
    coverage = {str(side): combine_independent(side_groups.get(side, {}))
                for side in sides}
    # 保留调用方额外提供的 side（例如 question 模式），便于审计；
    # 但它们不会改变默认 old/new 的 bilateral 语义。
    for side, groups in side_groups.items():
        if str(side) not in coverage:
            coverage[str(side)] = combine_independent(groups)
    return {'coverage': coverage,
            'bilateral': min(coverage.values()) if coverage else 0.0}


def quote_credibility_match(match: str | None) -> str:
    """把关系判断的 quote_match 映射为来源可信度的三档。

    exact      : 原文精确连续，可信度加满分；
    stripped_* / *+whitespace : 只做了引号或空白归一化，仍算连续原文，但降档；
    None/失败  : 没有可用引文，不贡献引文可信度。
    """
    if match == 'exact':
        return 'exact'
    if match:
        return 'inexact'
    return 'none'


def _evaluate_single_target(
        items: Sequence[Mapping[str, Any]] | None,
        config: EvidenceConfidenceConfig | None = None,
        *,
        old_fact_ids: Sequence[str] = (),
        new_fact_ids: Sequence[str] = (),
        mode: str = 'relation') -> dict[str, Any]:
    """汇总候选证据并给出建议状态；不修改任何记忆或外部状态。

    关系模式要求旧、新两侧 Fact ID；问答模式不要求双侧覆盖。

    执行顺序：
        1. 校验输入；关系模式必须知道哪一侧是旧、哪一侧是新。
        2. 逐条证据计算 route、cred、role_weight 和有效置信度。
        3. 按 independence_group 汇总支持/反驳，按 side 汇总覆盖。
        4. 计算 b/d/K/u 和 bilateral。
        5. 按阈值给出建议状态；只返回建议，不执行更新。
    """
    config = config or EvidenceConfidenceConfig()
    if mode not in ('relation', 'question'):
        raise ValueError("mode 必须是 relation 或 question")
    if items is None:
        items = []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError('items 必须是序列')
    # Fact ID 统一转字符串，避免调用方传 int/str 混用导致 side 判断失效。
    old_ids = {str(value) for value in old_fact_ids}
    new_ids = {str(value) for value in new_fact_ids}
    if mode == 'relation' and (not old_ids or not new_ids):
        raise ValueError('关系模式必须提供旧、新两侧 Fact ID')

    # support/refute 按来源组汇总；side 只记录真正参与判断的证据角色。
    support_groups: dict[str, list[float]] = defaultdict(list)
    refute_groups: dict[str, list[float]] = defaultdict(list)
    side_groups: dict[str, dict[str, list[float]]] = {
        side: defaultdict(list) for side in SIDES}
    details: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError('每条证据必须是映射')
        # 角色必须是已知枚举；未识别角色直接报错，避免静默丢弃证据。
        role = str(item.get('role') or 'uncertain')
        if role not in config.role_weights:
            raise ValueError(f'未知证据角色：{role}')

        # 第 1 层：这条证据被哪些路线命中、排名/相似度如何。
        route = route_confidence_detail(item.get('route_strengths') or {},
                                        config.route_weights)
        # 第 2 层：来源本身是否可核查；与检索相关性相乘，表示“相关且可信”。
        credibility = source_credibility_detail(
            item.get('credibility_features') or {}, config.credibility_weights,
            min_score=config.credibility_min, max_score=config.credibility_max)
        role_weight = config.role_weights[role]
        confidence = route['score'] * credibility['score'] * role_weight

        # side 优先采用调用方显式声明；否则按 target_fact_id 归属推断。
        # 找不到归属时记为 question，既不计入 old/new 覆盖，也不冒充某一侧。
        target = item.get('target_fact_id')
        target_id = None if target is None else str(target)
        declared_side = item.get('side')
        if declared_side in SIDES:
            side = declared_side
        elif target_id in old_ids:
            side = 'old'
        elif target_id in new_ids:
            side = 'new'
        else:
            side = 'question'
        # 同源分组：优先调用方显式给出的 independence_group，其次 Chunk/Fact。
        group = str(item.get('independence_group') or item.get('chunk_id')
                    or item.get('fact_id') or f'item:{index}')

        details.append({
            'fact_id': item.get('fact_id'), 'chunk_id': item.get('chunk_id'),
            'target_fact_id': item.get('target_fact_id'), 'side': side,
            'role': role, 'role_weight': _round(role_weight),
            'independence_group': group,
            'route': route, 'credibility': credibility,
            'confidence': _round(confidence),
        })

        # 第 3 层：只有真正参与判断的角色才进入支持/反驳或双侧覆盖。
        # background 只是上下文，uncertain 不投票；它们仍保留在 details 里供审计。
        if role in EVIDENCE_ROLES:
            if side in SIDES:
                side_groups[side][group].append(confidence)
            if role in ('supports', 'explains_change'):
                support_groups[group].append(confidence)
            elif role == 'refutes':
                refute_groups[group].append(confidence)

    # 第 4 层：支持/反驳分别合并，再算冲突与不确定性。
    belief = signed_belief(support_groups, refute_groups)
    coverage = bilateral_coverage(side_groups)
    thresholds = config.thresholds
    b = belief['belief']
    d = belief['disbelief']
    u = belief['uncertainty']
    k = belief['conflict_mass']
    bilateral = coverage['bilateral']

    # 第 5 层：状态映射。顺序很重要：
    #   先判 CONFLICT，避免“双方都很强”被 SUFFICIENT 抢先；
    #   再判 SUFFICIENT，要求 belief 足够高、反驳足够低、不确定性足够低，
    #   关系任务还要求旧/新两侧都有覆盖。
    reasons: list[str] = []
    if not details:
        status = 'NOT_EVALUATED'
        reasons.append('no_evidence')
    elif k >= thresholds['conflict_mass']:
        status = 'CONFLICT'
        reasons.append('conflict_mass_above_threshold')
    elif (b >= thresholds['sufficient_belief']
          and d <= thresholds['refute_ceiling']
          and u <= thresholds['uncertainty_ceiling']
          and (mode != 'relation' or bilateral >= thresholds['bilateral_floor'])):
        status = 'SUFFICIENT'
    else:
        # 逐项记录未满足的条件，方便 P0 分析“为什么不是 SUFFICIENT”。
        if b < thresholds['sufficient_belief']:
            reasons.append('belief_below_threshold')
        if d > thresholds['refute_ceiling']:
            reasons.append('disbelief_above_ceiling')
        if u > thresholds['uncertainty_ceiling']:
            reasons.append('uncertainty_above_ceiling')
        if mode == 'relation' and bilateral < thresholds['bilateral_floor']:
            reasons.append('bilateral_below_floor')
        # 不确定性太高时直接 DEFER；只是缺一侧证据时用 NEED_MORE 表示可继续补查。
        if u >= thresholds['uncertainty_defer']:
            status = 'DEFER'
        elif mode == 'relation' and bilateral < thresholds['bilateral_floor']:
            status = 'NEED_MORE'
        else:
            status = 'DEFER'

    return {
        'schema': 'evidence_confidence_v1',
        'mode': mode,
        'status': status,
        'belief': _round(b),
        'disbelief': _round(d),
        'uncertainty': _round(u),
        'conflict_mass': _round(k),
        # 问答模式没有旧/新两侧，bilateral 不适用，返回 None 而不是 0 误导。
        'bilateral': _round(bilateral) if mode == 'relation' else None,
        'side_coverage': {side: _round(value) for side, value in coverage['coverage'].items()},
        'evidence': details,
        'thresholds': dict(thresholds),
        'reasons': reasons,
        # 明确标注：这些数值没有经过真实样本校准，不能当概率使用。
        'calibration': 'uncalibrated_engineering_defaults',
        'validation_status': 'model_assessed_not_independently_verified',
        'update_applied': False,
    }


def evaluate_candidate(items, config=None, *, old_fact_ids=(), new_fact_ids=(), mode='relation'):
    """证据置信度入口：按目标断言隔离，禁止把支持两端当作支持更新关系。

    服务于 EvidenceSupport 的 shadow 评估。每个 target_fact_id 单独聚合支持与
    反驳；关系目标使用保留 ID ``relation``，问答整体使用 ``question``。
    没有针对关系本身的角色时，关系分数为 None，不从双端覆盖反推关系可信度。
    """
    if mode not in ('relation', 'question'):
        raise ValueError('mode 必须是 relation 或 question')
    if items is None:
        items = []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError('items 必须是序列')
    old_ids, new_ids = set(map(str, old_fact_ids)), set(map(str, new_fact_ids))
    if mode == 'relation' and (not old_ids or not new_ids or old_ids & new_ids):
        raise ValueError('关系模式需要互不重叠的旧、新 Fact ID')
    config = config or EvidenceConfidenceConfig()
    grouped = defaultdict(list)
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError('每条证据必须是映射')
        target = str(item.get('target_fact_id') or 'question')
        if mode == 'relation' and target not in old_ids | new_ids | {'relation'}:
            raise ValueError('关系证据目标不属于当前事实对或 relation')
        grouped[target].append(item)
    # 单目标函数没有跨断言融合；原有 Noisy-OR/同源折扣只在该目标内部使用。
    targets = {target: _evaluate_single_target(grouped.get(target, []), config, mode='question')
               for target in sorted(set(grouped) | (old_ids | new_ids if mode == 'relation' else set()))}
    target_key = 'relation' if mode == 'relation' else 'question'
    primary = targets.get(target_key)
    output = dict(primary) if primary else dict(status='NEED_MORE' if items else 'NOT_EVALUATED',
        belief=None, disbelief=None, uncertainty=None, conflict_mass=None,
        reasons=['relation_not_evaluated' if items else 'no_evidence'], evidence=[])
    output.update(schema='evidence_confidence_v2', mode=mode, targets=targets,
                  calibration='uncalibrated_engineering_defaults', update_applied=False)
    if mode == 'relation':
        coverage = {}
        for side, ids in (('old', old_ids), ('new', new_ids)):
            votes = defaultdict(list)
            for fid in ids:
                for detail in targets[fid]['evidence']:
                    if detail['role'] in EVIDENCE_ROLES:
                        votes[detail['independence_group']].append(detail['confidence'])
            coverage[side] = combine_independent(votes)
        output.update(side_coverage=coverage, bilateral=min(coverage.values()))
        if output['status'] == 'SUFFICIENT' and output['bilateral'] < config.thresholds['bilateral_floor']:
            output.update(status='NEED_MORE', reasons=['bilateral_below_floor'])
    else:
        output.update(bilateral=None)
    return output


__all__ = [
    'ROLES', 'EVIDENCE_ROLES', 'SIDES',
    'DEFAULT_ROUTE_WEIGHTS', 'DEFAULT_CREDIBILITY_WEIGHTS',
    'DEFAULT_ROLE_WEIGHTS', 'DEFAULT_THRESHOLDS',
    'EvidenceConfidenceConfig',
    'rank_strength', 'route_confidence', 'route_confidence_detail',
    'source_credibility', 'source_credibility_detail',
    'combine_independent', 'signed_belief', 'bilateral_coverage',
    'quote_credibility_match', 'evaluate_candidate',
]
