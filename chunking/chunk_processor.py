"""建图第一步：把原始文档/会话切成 Chunk。

两种切块策略由 config.chunk.dialogue_input 决定：
    True  → 对话模式：委托 DialogChunkProcessor，按“一问一答”切；
    False → 文档模式：按 token（此处以空格分词近似）滑动窗口切。

Chunk 是后续所有步骤的基本单位：
    - 原文保存在 DynamicMemory.chunk_storage[chunk_id] 中；
    - 实体/关系抽取每次处理一个 Chunk；
    - 查询阶段命中的 Chunk 会直接拼进上下文。
因此 chunk_id 必须在整个语料内唯一，过滤后的空 Chunk 不应占用编号逻辑之外的语义。
"""

from typing import List, Dict, Any
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import ChunkConfig
from chunking.dialog_chunk_processor import DialogChunkProcessor


class ChunkProcessor:
    """切块调度器：根据配置选择对话切块或 token 切块。"""

    def __init__(self, config: ChunkConfig, data_type: str = "LongmemEval"):
        """Args:
            config:    切块配置（chunk_token_size / overlap / dialogue_input）。
            data_type: 数据集类型，"LOCOMO" 与其它值对应不同的对话格式解析。
        """
        self.config = config
        self.data_type = data_type
        self.dialog_processor = DialogChunkProcessor(config, data_type=data_type)
        logger.info(f"Chunk Processor initialized for {data_type} dataset")

    def chunk_by_token_size(self, text: str) -> List[Dict[str, Any]]:
        """按固定 token 窗口切分纯文本。

        说明：
            - 这里用 text.split() 近似 token，真正的 tokenizer 未接入；
            - 步长为 chunk_size - overlap，相邻 Chunk 会重叠 overlap 个词，
              避免关键句被切断后语义不完整；
            - start_idx/end_idx 是词序号，不是字符偏移。

        边界情况：若 chunk_size <= overlap，range 步长会变成 0 或负数并抛错，
        配置时需保证 chunk_token_size > chunk_overlap_token_size。
        """
        if not text:
            return []

        # Simple word-based tokenization (for demo)
        words = text.split()
        chunks = []

        chunk_size = self.config.chunk_token_size
        overlap = self.config.chunk_overlap_token_size

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk_text = ' '.join(chunk_words)
                chunk = {
                    'text': chunk_text,
                    'start_idx': i,
                    'end_idx': i + len(chunk_words),
                    'token_count': len(chunk_words)
                }
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

    def chunk_by_dialog_turns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对话模式切块：转交给 DialogChunkProcessor。"""
        return self.dialog_processor.create_dialog_chunks(session_data)

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """切分单个文档/会话，并补上 doc_id 等元数据。

        对话模式下，doc 通常是一个会话（含 session_id/session_time/context）；
        文档模式下，doc 含 title/content。
        """
        # Check dialogue_input first, then fall back to chunk_method
        dialogue_input = getattr(self.config, 'dialogue_input', False)

        if dialogue_input:
            # Dialogue-based chunking
            chunks = self.chunk_by_dialog_turns(doc)
            for chunk in chunks:
                chunk['doc_id'] = doc.get('doc_id', 0)
            return chunks

        else:
            # Default: token-based chunking
            # 标题并入正文，保证切块结果中保留文档主题信息。
            title = doc.get('title', '')
            content = doc.get('content', '')
            full_text = f"{title}\n\n{content}" if title else content
            chunks = self.chunk_by_token_size(full_text)
            for chunk in chunks:
                chunk.update({
                    'doc_id': doc.get('doc_id', 0),
                    'title': title})
            return chunks

    def process_corpus(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理整个语料，并分配全局唯一 chunk_id。

        chunk_id 从 0 开始按“加入 all_chunks 的顺序”递增，
        即先按文档顺序，再按文档内部的切块顺序。
        这个编号是实体/关系回溯原文的主键，必须稳定且唯一。
        """
        all_chunks = []
        global_chunk_id = 0

        for doc in corpus:
            doc_chunks = self.chunk_document(doc)

            for chunk in doc_chunks:
                chunk['chunk_id'] = global_chunk_id
                global_chunk_id += 1

            all_chunks.extend(doc_chunks)

        logger.info(f"Processed {len(corpus)} documents into {len(all_chunks)} chunks with unique chunk IDs")
        return all_chunks
