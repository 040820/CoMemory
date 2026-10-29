"""兼容旧测试和引用路径：直接复用正式检索模块，不再复制一套算法。"""
from coregraph.structural_semantic_retriever import RetrievalConfig, StructuralSemanticRetriever

__all__ = ['RetrievalConfig', 'StructuralSemanticRetriever']
