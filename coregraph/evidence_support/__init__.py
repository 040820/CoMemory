"""证据支撑模块：为关系审查与问答提供共用的只读取证接口。"""

from .config import EvidenceSupportConfig
from .retriever import EvidenceRetriever
from .service import EvidenceSupport

__all__ = ['EvidenceSupportConfig', 'EvidenceRetriever', 'EvidenceSupport']
