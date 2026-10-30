"""session_summary 模式的上层记忆构建：为每个会话生成/更新一段摘要。

摘要的作用：
    查询阶段除了图/三元组检索，还会按摘要相似度找回相关会话，
    作为上下文的一部分参与重排。摘要保存在 base_dir/session_summaries.json，
    独立于 pkl 图文件。

写入时机：
    - 一次性建图且 retriever.enable_summary=True：建图前批量生成；
    - add 增量模式：每处理一个会话，用 ADDITION_PROMPT 更新或新建摘要。
    - event_state 模式：不使用摘要，本类不会被调用。

容错策略：
    LLM 返回的 JSON 可能带 markdown 代码块、解释性文字或格式错误，
    解析失败时退化为“把前 200 字塞进 context.summary_1”的占位摘要，
    保证后续流程拿到的始终是 dict，而不是异常。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Tuple
import json
import asyncio
from init.logger import logger
from prompt.summary_prompt import SUMMARY_PROMPT, ADDITION_PROMPT


class SessionSummarizer:
    """会话摘要的生成、增量更新与持久化。"""

    def __init__(self, llm_manager):
        """注入摘要模型，并初始化 session_id -> 摘要的进程内缓存。"""
        self.llm = llm_manager
        # 以 session_id 为键缓存已加载/已生成的摘要，供增量更新时查找旧摘要。
        self.existing_summaries = {}
        logger.info("Session Summarizer initialized")

    async def summarize_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """为单个会话生成摘要。

        Args:
            session_data: 含 session_id、session_time、context。

        Returns:
            摘要 dict。解析成功时保留 LLM 返回的字段（通常含 keys/context 等）；
            失败时返回 summary_status='failed' 的占位 dict。
        """
        try:
            session_id = session_data.get('session_id', 'unknown')
            session_time = str(session_data.get('session_time', 'unknown'))
            context = session_data.get('context', '')
            session_json = {
                "session_id": session_id,
                "session_time": session_time,
                "context": context
            }

            # 注意：这里没有使用 session_json，而是拼成“键: 值”的纯文本喂给提示词。
            formatted_text = f"Session ID: {session_id}\nSession Time: {session_time}\nContext: {context}"
            prompt = SUMMARY_PROMPT.replace('{text}', formatted_text)
            response = await self.llm.generate(prompt)
            summary = self._parse_summary_response(response, session_id, session_time)

            logger.debug(f"Generated summary for session {session_id}")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for session: {e}")
            return {
                'session_id': session_data.get('session_id', 'unknown'),
                'session_time': str(session_data.get('session_time', 'unknown')),  # Convert to string
                'summary_status': 'failed',
                'error': str(e)
            }

    def _parse_summary_response(self, response: str, session_id: str, session_time: str) -> Dict[str, Any]:
        """从 LLM 回复中提取摘要 JSON，失败则构造占位摘要。

        三种情况：
            1. 回复本身就是 JSON → 直接解析；
            2. 回复里嵌了 ```json ... ``` 代码块 → 用正则抠出 JSON；
            3. 完全不合法 → 用回复前 200 字生成一个最简摘要。

        无论哪条路径，都会补上 session_id/session_time，
        并移除内部状态字段 summary_status/raw_response，避免污染持久化文件。
        """
        try:
            if response.strip().startswith('{'):
                summary_data = json.loads(response)
                if 'session_id' not in summary_data:
                    summary_data['session_id'] = session_id
                if 'session_time' not in summary_data:
                    # 时间统一截断到日期：优先取空格前的日期，其次 T 分隔的日期。
                    summary_data['session_time'] = session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time
                summary_data.pop('summary_status', None)
                summary_data.pop('raw_response', None)
                logger.debug(f"Successfully parsed JSON summary for session {session_id}")
                return summary_data
            else:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
                if json_match:
                    try:
                        summary_data = json.loads(json_match.group(1))
                        if 'session_id' not in summary_data:
                            summary_data['session_id'] = session_id
                        if 'session_time' not in summary_data:
                            summary_data['session_time'] = session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time
                        summary_data.pop('summary_status', None)
                        summary_data.pop('raw_response', None)
                        logger.debug(f"Successfully extracted embedded JSON summary for session {session_id}")
                        return summary_data
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse embedded JSON in response for session {session_id}")
                # 无法解析：退化摘要，只保留截断后的原始回复。
                return {
                    'session_id': session_id,
                    'session_time': session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time,
                    'keys': 'session, conversation',
                    'context': {
                        'theme_1': 'General conversation',
                        'summary_1': response.strip()[:200] + '...' if len(response) > 200 else response.strip()
                    }
                }
        except json.JSONDecodeError:
            # 以 '{' 开头但不是合法 JSON 时走这里，逻辑与上面的“代码块/退化”一致。
            logger.warning(f"Failed to parse summary response as JSON for session {session_id}")
            import re
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
            if json_match:
                try:
                    summary_data = json.loads(json_match.group(1))
                    if 'session_id' not in summary_data:
                        summary_data['session_id'] = session_id
                    if 'session_time' not in summary_data:
                        summary_data['session_time'] = session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time
                    summary_data.pop('summary_status', None)
                    summary_data.pop('raw_response', None)
                    logger.debug(f"Successfully extracted embedded JSON summary for session {session_id}")
                    return summary_data
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse embedded JSON in response for session {session_id}")
            return {
                'session_id': session_id,
                'session_time': session_time.split(' ')[0] if ' ' in session_time else session_time.split('T')[0] if 'T' in session_time else session_time,
                'keys': 'session, conversation',
                'context': {
                    'theme_1': 'General conversation',
                    'summary_1': response.strip()[:200] + '...' if len(response) > 200 else response.strip()
                }
            }

    async def summarize_sessions(self, sessions: List[Dict[str, Any]], progress_bar=None) -> List[Dict[str, Any]]:
        """批量生成摘要。

        Args:
            sessions:     会话列表，每个含 session_id/session_time/context。
            progress_bar: 可选 tqdm；并发时由 batch_generate 更新，串行时每会话更新 1。

        Returns:
            与 sessions 顺序一致的摘要列表。单个会话失败会生成 failed 占位项，
            不会中断整批生成。

        并发/串行差异：并发路径一次提交所有 prompt；串行路径逐个 await。
        """
        if not sessions:
            return []

        if hasattr(self.llm, 'enable_concurrent') and self.llm.enable_concurrent:
            prompts = []
            session_metas = []

            for session in sessions:
                session_id = session.get('session_id', 'unknown')
                session_time = str(session.get('session_time', 'unknown'))
                context = session.get('context', '')
                session_json = {
                    "session_id": session_id,
                    "session_time": session_time,
                    "context": context
                }
                formatted_text = f"Session ID: {session_id}\nSession Time: {session_time}\nContext: {context}"
                prompt = SUMMARY_PROMPT.replace('{text}', formatted_text)
                prompts.append(prompt)
                session_metas.append(session)

            try:
                # Pass progress_bar to batch_generate so it updates as each request completes
                responses = await self.llm.batch_generate(prompts, progress_bar=progress_bar)
            except Exception as e:
                # 整批失败：用空回复占位，解析阶段会生成退化摘要。
                logger.error(f"Failed to batch generate summaries: {e}")
                responses = ["" for _ in prompts]
                # Update progress bar for failed requests
                if progress_bar:
                    progress_bar.update(len(prompts))

            summaries = []
            for i, (response, session) in enumerate(zip(responses, session_metas)):
                try:
                    session_id = session.get('session_id', 'unknown')
                    session_time = str(session.get('session_time', 'unknown'))  # Convert to string
                    summary = self._parse_summary_response(response, session_id, session_time)
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Failed to process summary response for session {i}: {e}")
                    summaries.append({
                        'session_id': session.get('session_id', 'unknown'),
                        'session_time': str(session.get('session_time', 'unknown')),  # Convert to string
                        'summary_status': 'failed',
                        'error': str(e)
                    })
        else:
            summaries = []

            for session in sessions:
                try:
                    summary = await self.summarize_session(session)
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Failed to summarize session {session.get('session_id', 'unknown')}: {e}")
                    summaries.append({
                        'session_id': session.get('session_id', 'unknown'),
                        'session_time': str(session.get('session_time', 'unknown')),
                        'summary_status': 'failed',
                        'error': str(e)
                    })
                finally:
                    if progress_bar:
                        progress_bar.update(1)

        logger.info(f"Generated summaries for {len(summaries)} sessions")
        return summaries

    def save_summaries(self, summaries: List[Dict[str, Any]], output_path: str) -> None:
        """把摘要列表写成 JSON 文件。

        预处理只剩一个目的：把 datetime 之类不可 JSON 序列化的对象转成 isoformat。
        """
        try:
            def convert_timestamps(obj):
                """递归把 datetime 等不可序列化对象转成字符串。"""
                if isinstance(obj, dict):
                    return {k: convert_timestamps(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_timestamps(item) for item in obj]
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return obj

            summaries_serializable = convert_timestamps(summaries)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summaries_serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(summaries)} summaries to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save summaries to {output_path}: {e}")

    def load_summaries(self, input_path: str) -> List[Dict[str, Any]]:
        """读取摘要文件，并刷新内部 session_id → summary 缓存。"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                summaries = json.load(f)
            logger.info(f"Loaded {len(summaries)} summaries from {input_path}")
            self.existing_summaries = {s.get('session_id', ''): s for s in summaries if s.get('session_id')}
            return summaries
        except Exception as e:
            logger.error(f"Failed to load summaries from {input_path}: {e}")
            return []

    async def update_summary_with_chunk(self, chunk: Dict[str, Any], existing_summary: Dict[str, Any] = None) -> Dict[str, Any]:
        """增量模式下更新一个会话的摘要。

        Args:
            chunk:            新会话文本（此处字段名为 chunk，实际承载整个会话的 context）。
            existing_summary: 旧摘要；为空时尝试从内部缓存按 session_id 查找。

        行为：
            - 找到旧摘要 → 调 ADDITION_PROMPT 把新内容并入；
            - 没找到     → 当作新会话，调 SUMMARY_PROMPT 从头生成。
        """
        try:
            session_id = chunk.get('session_id', 'unknown')
            session_time = str(chunk.get('session_time', 'unknown'))
            chunk_text = chunk.get('text', '')

            if existing_summary is None:
                existing_summary = self.existing_summaries.get(session_id)

            if existing_summary:
                logger.info(f"Updating existing summary for session {session_id}")
                summary = await self._update_existing_summary(existing_summary, chunk_text)
            else:
                logger.info(f"Creating new summary for session {session_id}")
                formatted_text = f"Session ID: {session_id}\nSession Time: {session_time}\nContext: {chunk_text}"
                prompt = SUMMARY_PROMPT.replace('{text}', formatted_text)
                response = await self.llm.generate(prompt)
                summary = self._parse_summary_response(response, session_id, session_time)

            self.existing_summaries[session_id] = summary
            logger.debug(f"Summary updated for session {session_id}")
            return summary

        except Exception as e:
            logger.error(f"Failed to update summary for chunk: {e}")
            return {
                'session_id': chunk.get('session_id', 'unknown'),
                'session_time': str(chunk.get('session_time', 'unknown')),
                'summary_status': 'failed',
                'error': str(e)
            }

    async def _update_existing_summary(self, existing_summary: Dict[str, Any], new_chunk_text: str) -> Dict[str, Any]:
        """调用 ADDITION_PROMPT，把新文本合并进旧摘要。

        失败时返回旧摘要本身（宁可少更新，也不要丢已有摘要）。
        """
        try:
            existing_summary_str = json.dumps(existing_summary, ensure_ascii=False, indent=2)
            prompt = ADDITION_PROMPT.format(
                summary=existing_summary_str,
                text=new_chunk_text
            )

            response = await self.llm.generate(prompt)
            session_id = existing_summary.get('session_id', 'unknown')
            session_time = existing_summary.get('session_time', 'unknown')
            updated_summary = self._parse_summary_response(response, session_id, session_time)
            logger.debug(f"Successfully updated summary for session {session_id}")
            return updated_summary

        except Exception as e:
            logger.error(f"Failed to update existing summary: {e}")
            return existing_summary
