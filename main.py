"""LiCoMemory 命令行入口：建图 → 查询 → 评测 的主流程编排。

典型用法：
    python main.py -opt config/Memory.yaml -dataset_name group_1 -root locomo_group_1 -query 1

整体步骤：
    1. 解析命令行参数与 YAML 配置；
    2. 准备 ./results/<root>/ 输出目录，并把日志重定向到该目录；
    3. 初始化 GraphRAG（内部持有 LLM、Embedding、分块器与动态记忆图）；
    4. 读取数据集的 Corpus.json / Question.json；
    5. 根据 force / add 配置决定“重建图 / 增量写入 / 直接加载已有图”；
    6. 可选地逐题查询，并在过程中实时打印评测结果；
    7. 查询结束后做一次完整评测，把指标写入 metrics.json。

两种上层记忆模式由配置项 upper_memory_mode 控制：
    - session_summary：默认模式，生成会话摘要并参与检索重排；
    - event_state    ：用事件/状态演化层替代摘要，查询走结构化语义检索。
"""

# ---------------------------------------------------------------------------
# 标准库 / 第三方依赖
# ---------------------------------------------------------------------------
import argparse
import asyncio
import json
import os
import pandas as pd
from pathlib import Path
from shutil import copyfile  # 当前版本未使用，保留导入不影响运行

import sys
import os
# 将本文件所在目录加入模块搜索路径，使得 `python main.py` 可以直接运行，
# 同时兼容从项目根目录之外启动脚本的情况。
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 项目内部模块
# ---------------------------------------------------------------------------
from init.config import Config
from init.graph_rag import GraphRAG
from init.logger import update_logger_path
from evaluation.evaluator import Evaluator
from utils.final_report import FinalReportGenerator
from data.query_dataset import RAGQueryDataset

def parse_args():
    """解析命令行参数。

    参数说明（都可缺省，缺省值依赖 YAML 或后续交互）：
        -opt             YAML 配置文件路径，例如 config/Memory.yaml。
        -dataset_name    数据集/样例目录名，例如 LoCoMo 的 group_1、
                         LongMemEval 的 lm_1；会覆盖 YAML 中的 dataset_name。
        -external_graph  预留给“从外部图文件加载”的参数，当前主流程未使用
                         （实际图路径由 base_dir + index_name 决定）。
        -root            结果子目录名，输出落在 ./results/<root>/；为空则用 ./results。
        -query           是否在加载/建图后执行查询与评测。不传则跳过。
                         注意：主流程按“非空字符串”判断，传入 "0" 也会开启查询。
    """
    parser = argparse.ArgumentParser(description="Dynamic Memory Graph RAG System")
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("-external_graph", type=str, help="Path to external tree file to load from.")
    parser.add_argument("-root", type=str, default="", help="Root directory to prefix result/config/metric paths.")
    parser.add_argument("-query", type=str, default=None, help="Whether to run query and evaluation (1 to enable, 0 to disable). If not specified, will prompt user.")
    return parser.parse_args()

def check_dirs(config: Config, root: str):
    """创建并返回本次运行使用的目录。

    目录布局：
        ./results/<root>/                 根目录，保存图 pkl、session_summaries.json、日志
        ./results/<root>/results/         逐题结果 results.json 与指标 metrics.json

    root 为空时退化为 ./results。把“图/摘要”和“逐题结果/指标”分开放，
    是为了避免一次运行的 results.json 覆盖同一目录下其他实验的产物。

    Returns:
        (base_dir, result_dir) 二元组。
    """
    # Create root folder under results directory
    if root:
        base_dir = os.path.join("./results", root)
    else:
        base_dir = "./results"

    # Only create results subdirectory
    result_dir = os.path.join(base_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    # Also create the base directory for pkl and log files
    os.makedirs(base_dir, exist_ok=True)

    return base_dir, result_dir

async def process_queries_async(query_dataset, graph_rag, dataset_len, config):
    """逐题执行查询，并在过程中实时打印/记录评测结果。

    Args:
        query_dataset: RAGQueryDataset，支持按整数索引取题（返回 dict）。
        graph_rag:     已初始化的 GraphRAG 门面对象。
        dataset_len:   题目数量。
        config:        全局配置，用于决定使用 LLM 判分还是精确匹配。

    Returns:
        与题目顺序一致的列表 all_res。每道题会补充：
            output          模型答案（失败时为占位文本）
            top_session_ids 检索命中的会话 ID（供会话命中率使用）

    说明：
        - 单题异常不会中断整轮评测，异常被捕获后继续处理下一题；
        - 过程中打印的准确率只是即时反馈，最终指标以 metrics.json 为准。
    """
    from evaluation.llm_evaluator import LLMEvaluator
    from evaluation.session_matching_evaluator import SessionMatchingEvaluator
    from evaluation.evaluator import Evaluator
    from init.logger import logger

    all_res = []

    # 选择评测方式：
    #   LLM 判分   —— 适合开放式回答，但每道题会额外调用一次 LLM；
    #   精确匹配   —— 按字符串包含/数字匹配，成本低但更严格。
    enable_llm_eval = config.evaluation.enable_llm_eval if config else False
    if enable_llm_eval:
        # results_path / dataset_name 传空字符串：这里只做单题即时判分，
        # 完整评测稍后由 wrapper_evaluation() 基于落盘文件统一执行。
        llm_evaluator = LLMEvaluator(config, "", "")
        print(f"🔍 Using LLM evaluation with model: {config.evaluation.eval_model}")
        logger.info(f"🔍 Using LLM evaluation with model: {config.evaluation.eval_model}")
    else:
        # 该对象在当前函数内未被使用，仅用于确认精确匹配评测器可用；
        # 精确匹配的最终指标在 wrapper_evaluation() 中计算。
        evaluator_obj = Evaluator("", "", None)
        print("🔍 Using exact match evaluation")
        logger.info("🔍 Using exact match evaluation")

    # 实时累计统计（只用于过程中展示；最终指标以 metrics.json 为准）
    total_correct_llm = 0
    total_correct_exact = 0  # 预留计数：当前函数未更新该变量
    total_evaluated = 0

    # 按数据集顺序逐题处理。该循环是串行的：每题内部再并发调用 LLM/Embedding，
    # 避免不同题之间争抢同一个事件循环与模型并发槽。
    for i in range(dataset_len):
        query = query_dataset[i]
        try:
            # question_time 仅 LongMemEval 提供；LoCoMo 的 Question.json 没有该字段，
            # 此时传入空字符串，时间感知重排会退化为“不按时间衰减”。
            question_time = query.get("question_time", "")
            res = await graph_rag.query(query["question"], question_time=question_time)

            # graph_rag.query() 正常返回 dict；异常路径可能返回字符串（错误信息）。
            # 这里把两种返回统一成相同的结果形状，供后续落盘与评测使用。
            if isinstance(res, dict):
                query["output"] = res.get('answer', '')
                query["top_session_ids"] = res.get('top_session_ids', [])
            else:
                query["output"] = res
                query["top_session_ids"] = []

            all_res.append(query)

            # 实时评测：只对“标准答案和模型输出都非空”的题计分，
            # 避免把空答案题误算成错误或污染分母。
            expected_answer = str(query.get('answer', '')).strip()
            model_output = str(query.get('output', '')).strip()

            if expected_answer and model_output:
                total_evaluated += 1

                # LLM 判分：按 question_type 选择提示词模板。
                # LongMemEval 使用 single-session-user 等类别名；
                # LoCoMo 的 question_type 是数字字符串，会走默认模板。
                if enable_llm_eval:
                    question_type = query.get('question_type', 'default')
                    is_correct_llm = await llm_evaluator.evaluate_with_llm(
                        query.get('question', ''),
                        expected_answer,
                        model_output,
                        question_type
                    )
                    if is_correct_llm:
                        total_correct_llm += 1

                # 取出该题的耗时与 token 统计：
                #   原 QueryProcessor 会填充 query_summary/cost_summary；
                #   event_state 的查询包装器目前不返回这两项，因此可能为空。
                query_stats = res.get('query_summary', {}) if isinstance(res, dict) else {}
                time_stats = query_stats.get('detailed_retrieval_breakdown', {})
                cost_stats = res.get('cost_summary', {}) if isinstance(res, dict) else {}

                # 打印并记录本题结果，便于在长任务中即时观察进度与失败原因。
                separator = "=" * 80
                print(f"\n{separator}")
                logger.info(separator)

                question_info = f"Question {i+1}/{dataset_len}:"
                print(question_info)
                logger.info(question_info)

                # 时间统计：跳过内部合计字段 detailed_total，只展示各检索阶段。
                if time_stats:
                    print("Time Statistics:")
                    logger.info("Time Statistics:")
                    for stage, time_val in time_stats.items():
                        if stage != 'detailed_total':
                            stage_name = stage.replace('_', ' ').title()
                            print(f"  {stage_name}: {time_val:.2f}s")
                            logger.info(f"  {stage_name}: {time_val:.2f}s")

                # Token 统计：区分检索阶段与答案生成阶段的消耗。
                if cost_stats:
                    print("Token Statistics:")
                    logger.info("Token Statistics:")
                    retrieval_tokens = cost_stats.get('retrieval_tokens', 0)
                    answer_tokens = cost_stats.get('answer_generation_tokens', 0)
                    total_tokens = cost_stats.get('total_query_tokens', 0)
                    print(f"  Retrieval Tokens: {retrieval_tokens}")
                    logger.info(f"  Retrieval Tokens: {retrieval_tokens}")
                    print(f"  Answer Generation Tokens: {answer_tokens}")
                    logger.info(f"  Answer Generation Tokens: {answer_tokens}")
                    print(f"  Total Tokens: {total_tokens}")
                    logger.info(f"  Total Tokens: {total_tokens}")

                # 对照打印标准答案与模型答案，便于人工抽查。
                print(f"Expected: {expected_answer}")
                logger.info(f"Expected: {expected_answer}")
                print(f"Got: {model_output}")
                logger.info(f"Got: {model_output}")

                # 实时准确率：分母是“已评测题数”，不是总题数；
                # 精确匹配模式下该分支不累加，因此这里只在 LLM 模式有意义。
                if total_evaluated > 0:
                    accuracy = total_correct_llm / total_evaluated * 100 if enable_llm_eval else 0.0
                    accuracy_text = f"Current Accuracy: {total_correct_llm}/{total_evaluated} ({accuracy:.1f}%)"
                    print(accuracy_text)
                    logger.info(accuracy_text)
            else:
                # 标准答案或模型输出缺失：跳过计分，但该题仍保留在 all_res 中，
                # 以保持 results.json 与数据集的题目数量和顺序一致。
                separator = "=" * 80
                skip_msg = f"Question {i+1}/{dataset_len}: ⚠️  Skipped (missing answer or output)"
                print(f"\n{separator}")
                print(skip_msg)
                logger.warning(skip_msg)

        except Exception as e:
            # 单题失败不终止整轮：记录堆栈、写入占位答案并继续。
            # 这样一次长跑不会因为个别坏样本而丢掉全部已完成结果。
            separator = "=" * 80
            error_msg = f"❌ Error processing query {i+1}: {e}"
            print(f"\n{separator}")
            print(error_msg)
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            query["output"] = "Error processing query"
            query["top_session_ids"] = []
            all_res.append(query)

    # 单题循环结束后的汇总。注意：这里只展示 LLM 评测的准确率；
    # 精确匹配与会话命中的完整结果在 wrapper_evaluation() 中计算。
    separator = "=" * 80
    print(f"\n{separator}")
    logger.info(separator)

    summary_title = "📊 FINAL EVALUATION SUMMARY"
    print(summary_title)
    logger.info(summary_title)

    print(separator)
    logger.info(separator)

    total_q = f"Total Questions: {dataset_len}"
    print(total_q)
    logger.info(total_q)

    evaluated = f"Evaluated: {total_evaluated}"
    print(evaluated)
    logger.info(evaluated)

    if enable_llm_eval:
        if total_evaluated > 0:
            final_accuracy = f"Final Accuracy: {total_correct_llm}/{total_evaluated} ({total_correct_llm/total_evaluated*100:.2f}%)"
        else:
            final_accuracy = "Final Accuracy: N/A"
        print(final_accuracy)
        logger.info(final_accuracy)

    print(f"{separator}\n")
    logger.info(separator)

    return all_res

def wrapper_query(query_dataset, graph_rag, result_dir, config=None):
    """串行处理全部问题，并把逐题结果保存为 results.json。

    使用单个事件循环执行所有异步查询，避免每题重复创建/销毁 loop
    （LLM 与 Embedding 客户端在初始化时绑定到当前 loop）。
    结果以 JSON 数组形式落盘：DataFrame 会把嵌套的 evidence、查询统计等
    字段一并序列化，供后续评测与人工复查使用。

    Returns:
        results.json 的保存路径。
    """
    dataset_len = len(query_dataset)
    print(f"Processing {dataset_len} queries...")

    # Use single event loop for all queries
    all_res = asyncio.run(process_queries_async(query_dataset, graph_rag, dataset_len, config))

    # Save results
    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", indent=2)
    print(f"Results saved to {save_path}")

    return save_path

async def wrapper_evaluation(path, dataset_name, result_dir, config=None):
    """对 results.json 做整体评测，并把指标写入 metrics.json。

    Evaluator 内部按配置选择判分方式：
        - enable_llm_eval=True  → LLMEvaluator 逐题判断答案是否正确；
        - enable_llm_eval=False → 字符串/数字精确匹配；
    两种方式都会额外追加 SessionMatchingEvaluator 的会话命中分数。

    Args:
        path:         逐题结果文件路径（wrapper_query 的返回值）。
        dataset_name: 数据集名称，仅用于指标元数据。
        result_dir:   metrics.json 的输出目录。

    Returns:
        评测指标字典。
    """
    eval = Evaluator(path, dataset_name, config)
    res_dict = await eval.evaluate()

    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(res_dict, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to {save_path}")
    return res_dict

if __name__ == "__main__":
    args = parse_args()

    # 1) 读取 YAML 配置；命令行 -dataset_name 优先于 YAML 中的 dataset_name。
    # Load configuration
    config = Config.parse(Path(args.opt), dataset_name=args.dataset_name)

    # 2) 准备输出目录：图 pkl、摘要、日志都在 base_dir；
    #    逐题结果与指标在 base_dir/results。日志路径在此之后才被重定向。
    # Create directories first
    base_dir, result_dir = check_dirs(config, args.root)

    # Update logger path to use the new base directory
    update_logger_path(base_dir)

    # 3) 初始化门面对象：内部会创建 LLM 管理器、Embedding 管理器、
    #    分块器以及 DynamicMemory（包含实体关系图与事件状态层）。
    # Initialize GraphRAG with base directory
    graph_rag = GraphRAG(config, base_dir)

    # 4) 读取数据集：Corpus.json 是会话/文档，Question.json 是逐题问答。
    #    读取失败时退化为空语料，使“只加载已有图”的模式仍可继续运行。
    # Load dataset
    try:
        query_dataset = RAGQueryDataset(
            data_dir=os.path.join(config.data_root, config.dataset_name)
        )
        corpus = query_dataset.get_corpus()
        print(f"Loaded dataset with {len(corpus)} documents")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        corpus = []

    # 5) 决定本次的记忆写入模式：
    #      force=True, add=False → 清空并重建图；
    #      force=False, add=True → 加载已有图，按会话增量写入；
    #      force=False, add=False → 直接加载已有 pkl。
    #    注意最后一种分支调用 insert([]) 并非“插入空数据”，
    #    而是复用 DynamicMemory 内部“无 force/add 时加载已有图”的分支逻辑。
    # Check if we should insert documents or load existing graph
    force = getattr(config.graph, 'force', False)
    add = getattr(config.graph, 'add', False)

    if not force and not add:
        print("Load existing mode: skipping document insertion, will load from pkl file")
        asyncio.run(graph_rag.insert([]))
    elif corpus:
        print("Inserting documents into graph...")
        asyncio.run(graph_rag.insert(corpus))
        print("Document insertion completed")
    else:
        print("No corpus provided and not in load existing mode")

    # 6) 汇总建图阶段的耗时与 token 统计，供最终报告使用。
    #    注意：当前 GraphRAGCore 只暴露 self.graph（DynamicMemory），
    #    并没有 `dynamic_memory` 属性，因此该分支通常不会执行。
    # Initialize final report generator
    final_report = FinalReportGenerator()

    # Collect graph building statistics
    if hasattr(graph_rag.core, 'dynamic_memory') and graph_rag.core.dynamic_memory:
        dm = graph_rag.core.dynamic_memory
        if hasattr(dm, 'time_manager') and hasattr(dm, 'cost_manager'):
            final_report.set_graph_building_stats(
                dm.time_manager.get_graph_building_summary(),
                dm.cost_manager.get_graph_building_summary()
            )

    # 7) 可选：逐题查询并实时评测，然后再做一次完整评测。
    #    注意 args.query 是字符串，任何非空值（包括 "0"）都会被当作开启。
    # Run queries if requested
    if args.query:
        print("Running query and evaluation...")
        save_path = wrapper_query(query_dataset, graph_rag, result_dir, config)
        evaluation_results = asyncio.run(wrapper_evaluation(save_path, config.dataset_name, result_dir, config))
        final_report.set_evaluation_results(evaluation_results)

        # 8) 收集查询阶段的累计耗时/token 统计。
        #    同样地，GraphRAGCore 并未保存 query_processor 引用（QueryProcessor
        #    是在 GraphRAG.query() 内部临时创建的），因此该分支通常也不会执行；
        #    每题的实际统计已随 res['query_summary']/res['cost_summary'] 返回。
        # Collect query statistics from query processor
        if hasattr(graph_rag.core, 'query_processor') and graph_rag.core.query_processor:
            qp = graph_rag.core.query_processor
            if hasattr(qp, 'time_manager') and hasattr(qp, 'cost_manager'):
                final_report.add_query_stats(
                    qp.time_manager.get_query_summary(),
                    qp.cost_manager.get_query_summary()
                )
    else:
        print("Skipping query and evaluation.")

    # 说明：当前版本只构建 FinalReportGenerator，尚未调用 print_final_report()。
    # 若需要输出完整汇总，可在此处执行 final_report.print_final_report()。
    print("Dynamic Memory Graph RAG System execution completed!")
