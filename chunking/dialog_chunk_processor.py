"""对话数据切块：把一个会话的上下文解析成说话轮次，再按“轮次对”组成 Chunk。

支持两种对话格式（由 data_type 区分）：
    LOCOMO     ：'{"SpeakerName": "内容", ...}'，说话人名字不固定（可能是多个角色）；
    LongmemEval：'{"user": "内容", "assistant": "内容", ...}'，固定 user/assistant。

为什么按“两个轮次”切：
    一问一答构成最小完整语义单元；单独一句用户提问或助手回答往往缺少上下文，
    会让实体/关系抽取和后续检索都不准确。

注意：解析依赖原文字符串的引号结构（rfind/find '"'），
若数据集的 JSON 中含转义引号或跨行格式，可能需要调整。
"""

import json
import re
import hashlib
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import ChunkConfig


class DialogChunkProcessor:
    """对话切块器：解析轮次 → 两两组合成 Chunk。"""

    def __init__(self, config: ChunkConfig, data_type: str = "LongmemEval"):
        """Args:
            config:    切块配置（本类暂未直接使用其 token 参数）。
            data_type: "LOCOMO" 或其它（按 LongmemEval 格式处理）。
        """
        self.config = config
        self.data_type = data_type
        logger.info(f"Dialog Chunk Processor initialized for {data_type} dataset")

    def parse_dialog_turns(self, context_text: str) -> List[Dict[str, str]]:
        """把会话 context 解析为有序的说话轮次列表。

        解析思路（不依赖完整 JSON 解析，而是用正则找每个说话人标签的位置）：
            1. 找出所有 "speaker": " 出现的位置；
            2. 按出现位置排序，得到轮次边界；
            3. 第 i 个说话人的内容 = 从标签结尾到下一个标签开头之间的文本，
               再去掉末尾的引号与填充字符。

        Args:
            context_text: 原始会话字符串。

        Returns:
            [{'speaker': 说话人, 'content': 内容}, ...]，按对话顺序排列。
            内容为空或清洗后为空的轮次会被丢弃。
        """
        dialog_turns = []

        if self.data_type == "LOCOMO":
            # LOCOMO format: "SpeakerName": "content"
            # 说话人名字任意，因此用宽泛的 "([^"]+)" 捕获。
            speaker_pattern = r'"([^"]+)"\s*:\s*"'
            positions = []
            for match in re.finditer(speaker_pattern, context_text):
                speaker_name = match.group(1)
                positions.append((speaker_name, match.end()))
            positions.sort(key=lambda x: x[1])
        else:
            # LongmemEval format: "user"/"assistant"
            # 只认这两个固定角色，避免把正文里的带引号内容误判为说话人。
            user_pattern = r'"user"\s*:\s*"'
            assistant_pattern = r'"assistant"\s*:\s*"'
            positions = []
            for match in re.finditer(user_pattern, context_text):
                positions.append(('user', match.end()))
            for match in re.finditer(assistant_pattern, context_text):
                positions.append(('assistant', match.end()))
            positions.sort(key=lambda x: x[1])
        for i, (speaker, start_pos) in enumerate(positions):
            if i + 1 < len(positions):
                # 当前轮次内容止于“下一个说话人标签的开头”。
                # 标签长度 = '"speaker": "'，需要从下一标签的 end 回退这段长度。
                next_speaker = positions[i + 1][0]
                if self.data_type == "LOCOMO":
                    next_speaker_prefix_len = len(f'"{next_speaker}": "')
                else:
                    next_speaker_prefix_len = len(f'"{next_speaker}": "')
                next_speaker_pos = positions[i + 1][1] - next_speaker_prefix_len
                content_part = context_text[start_pos:next_speaker_pos]
                # 去掉内容结尾的收尾引号。
                quote_pos = content_part.rfind('"')
                if quote_pos >= 0:
                    content = content_part[:quote_pos]
                else:
                    content = content_part.strip()
            else:
                # 最后一个轮次没有下一个标签，止于下一个引号出现处。
                content_part = context_text[start_pos:]
                quote_pos = content_part.find('"')
                if quote_pos >= 0:
                    content = content_part[:quote_pos]
                else:
                    content = content_part.strip()

            content = content.strip()
            # 清理内容中残留的下一个说话人标签碎片（形如  ""speaker": " ）。
            content = re.sub(r'\s*"\s*"[^"]*":\s*', '', content)
            # 合并多余空白，避免切块文本中出现大段空白。
            content = re.sub(r'\s+', ' ', content)

            if content:
                dialog_turns.append({
                    'speaker': speaker,
                    'content': content
                })

        return dialog_turns

    def create_dialog_chunks(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """把一个会话切成若干 Chunk，每个 Chunk 含 1~2 个对话轮次。

        切法：
            LOCOMO      ：每两个轮次组成一个 Chunk（不检查角色是否交替）；
            LongmemEval ：优先把连续的 user + assistant 组成一对；
                          落单的 assistant 也单独成块；落单的 user 不单独成块，
                          会与下一个轮次重新配对（若下一个仍不是 assistant，则该 user 被跳过）。

        每个 Chunk 携带 session_id / session_time，供抽取器和图构建记录来源。

        Args:
            session_data: 含 session_id、session_time、context 的会话字典。

        Returns:
            Chunk 列表；解析不到任何轮次时返回空列表并记录警告。
        """
        session_time = str(session_data.get('session_time', 'unknown'))
        context = session_data.get('context', '')
        session_id = session_data.get('session_id', '')

        dialog_turns = self.parse_dialog_turns(context)
        chunks = []

        if not dialog_turns:
            logger.warning(f"No dialog turns found in session {session_id}")
            return []
        i = 0
        turn_pair_index = 0

        if self.data_type == "LOCOMO":
            # LOCOMO: Group consecutive turns from different speakers (2 turns per chunk)
            while i < len(dialog_turns):
                chunk_content_parts = []
                turn_speakers = []
                if i < len(dialog_turns):
                    turn1 = dialog_turns[i]
                    speaker1 = turn1['speaker']
                    chunk_content_parts.append(f"{speaker1}: {turn1['content']}")
                    turn_speakers.append(speaker1)
                    i += 1
                    if i < len(dialog_turns):
                        turn2 = dialog_turns[i]
                        speaker2 = turn2['speaker']
                        chunk_content_parts.append(f"{speaker2}: {turn2['content']}")
                        turn_speakers.append(speaker2)
                        i += 1

                if chunk_content_parts:
                    chunk_text = "\n".join(chunk_content_parts)

                    chunk = {
                        'text': chunk_text,
                        'session_id': session_id,
                        'session_time': session_time
                    }

                    chunks.append(chunk)
                    turn_pair_index += 1
        else:
            # LongmemEval: Group user-assistant pairs
            while i < len(dialog_turns):
                chunk_content_parts = []
                turn_speakers = []

                if i < len(dialog_turns) and dialog_turns[i]['speaker'] == 'user':
                    # 正常情况：user 后面紧跟 assistant，两者成对。
                    user_turn = dialog_turns[i]
                    chunk_content_parts.append(f"User: {user_turn['content']}")
                    turn_speakers.append('user')
                    i += 1
                    if i < len(dialog_turns) and dialog_turns[i]['speaker'] == 'assistant':
                        assistant_turn = dialog_turns[i]
                        chunk_content_parts.append(f"Assistant: {assistant_turn['content']}")
                        turn_speakers.append('assistant')
                        i += 1
                else:
                    # 兜底：当前是 assistant（例如会话以助手发言开头），单独成块。
                    if i < len(dialog_turns) and dialog_turns[i]['speaker'] == 'assistant':
                        assistant_turn = dialog_turns[i]
                        chunk_content_parts.append(f"Assistant: {assistant_turn['content']}")
                        turn_speakers.append('assistant')
                        i += 1

                if chunk_content_parts:
                    chunk_text = "\n".join(chunk_content_parts)

                    chunk = {
                        'text': chunk_text,
                        'session_id': session_id,
                        'session_time': session_time
                    }
                    chunks.append(chunk)
                    turn_pair_index += 1

        return chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """统计切块结果（当前主流程未调用，保留用于调试）。

        注意：user_only_chunks 等字段依赖 chunk['metadata']，
        而 create_dialog_chunks 产出的 Chunk 并未写入 metadata，
        因此这些计数在现有数据上通常为 0。
        """
        if not chunks:
            return {}

        total_chunks = len(chunks)
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        unique_sessions = len(set(chunk.get('session_id', '') for chunk in chunks))

        user_only_chunks = sum(1 for chunk in chunks
                              if chunk.get('metadata', {}).get('has_user_turn', False)
                              and not chunk.get('metadata', {}).get('has_assistant_turn', False))

        assistant_only_chunks = sum(1 for chunk in chunks
                                   if chunk.get('metadata', {}).get('has_assistant_turn', False)
                                   and not chunk.get('metadata', {}).get('has_user_turn', False))

        complete_pairs = sum(1 for chunk in chunks
                            if chunk.get('metadata', {}).get('has_user_turn', False)
                            and chunk.get('metadata', {}).get('has_assistant_turn', False))

        avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks > 0 else 0

        return {
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'unique_sessions': unique_sessions,
            'avg_tokens_per_chunk': avg_tokens_per_chunk,
            'complete_dialog_pairs': complete_pairs,
            'user_only_chunks': user_only_chunks,
            'assistant_only_chunks': assistant_only_chunks,
            'avg_chunks_per_session': total_chunks / unique_sessions if unique_sessions > 0 else 0
        }

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """校验 Chunk 格式（当前主流程未调用）。

        要求每个 Chunk 含 text/session_id/session_time/chunk_type，
        且 chunk_type == 'dialog_turn'。由于 create_dialog_chunks 并不写 chunk_type，
        现有产出直接调用本方法会校验失败——它更像是为另一套切块实现保留的接口。
        """
        required_fields = ['text', 'session_id', 'session_time', 'chunk_type']

        for i, chunk in enumerate(chunks):
            for field in required_fields:
                if field not in chunk:
                    logger.error(f"Chunk {i} missing required field: {field}")
                    return False

            if chunk.get('chunk_type') != 'dialog_turn':
                logger.error(f"Chunk {i} has invalid chunk_type: {chunk.get('chunk_type')}")
                return False

            if not chunk.get('text', '').strip():
                logger.error(f"Chunk {i} has empty text content")
                return False

        return True
