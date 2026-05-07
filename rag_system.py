import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict, Any
import numpy as np
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import os

class ScientificRAGSystem:
    """原有系统，仅进行向量检索。此处保留以确保其他代码的兼容性。"""
    def __init__(self, collection_name="scientific_papers", persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        self.collection = self._get_or_create_collection()
        self.multimodal_metadata = {}  # 存储多模态元数据

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(self.collection_name)
        except:
            return self.client.create_collection(self.collection_name)

    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = self.client.create_collection(self.collection_name)
        self.multimodal_metadata = {}

    def add_documents(self, texts, metadatas=None, multimodal_info=None):
        """
        重写方法：支持添加多模态信息
        """
        if metadatas is None:
            metadatas = [{"source": "paper"} for _ in texts]

        # 如果有额外的多模态信息，将其作为特殊文档添加
        if multimodal_info:
            for mm_type, mm_content in multimodal_info.items():
                if mm_content:
                    texts.append(mm_content)
                    metadatas.append({
                        "source": "multimodal",
                        "type": mm_type,
                        "priority": "high"  # 多模态信息优先级高
                    })

        # 修复：使用 texts 而不是 text_chunks
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        ids = [f"chunk_{i}" for i in range(len(texts))]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve(self, query, k=5):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        return results["documents"][0] if results["documents"] else []

    def retrieve_with_metadata(self, query, k=5):
        """
        新增方法：返回带元数据的检索结果
        """
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "metadatas"]
        )
        return results


class EnhancedRAGSystem(ScientificRAGSystem):
    """
    增强版RAG系统：在原有向量检索基础上，增加混合检索（BM25）和重排序。
    继承自ScientificRAGSystem，因此原有方法（add_documents, clear_collection等）均可直接使用。
    """
    def __init__(self,
                 collection_name: str = "scientific_papers",
                 persist_dir: str = "./chroma_db",
                 reranker_model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化增强系统。
        :param reranker_model_name: 重排序模型，推荐使用轻量级的BGE Reranker系列。
        """
        # 调用父类初始化，建立向量数据库连接
        super().__init__(collection_name, persist_dir)

        # 初始化BM25所需的数据结构
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_tokenized_corpus = []

        # 初始化重排序模型
        print(f"[EnhancedRAG] 正在加载重排序模型: {reranker_model_name}")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, trust_remote_code=True)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, trust_remote_code=True)
        self.reranker_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.reranker_model = self.reranker_model.to(self.device)
        print("[EnhancedRAG] 模型加载完毕。")

    def _simple_tokenize(self, text: str) -> List[str]:
        """简单的中英文分词函数，用于BM25。"""
        words = re.split(r'[\s\u3000\.,;!?\(\).\[\]"「」""「」「」：:、/]+', text)
        return [w for w in words if w.strip()]

    def add_documents(self, texts, metadatas=None, multimodal_info=None):
        """
        重写父类方法：在添加文档到向量数据库的同时，也构建BM25索引。
        """
        # 1. 调用父类方法，完成向量数据库的添加（包含多模态信息）
        super().add_documents(texts, metadatas, multimodal_info)

        # 2. 构建BM25索引（只包含文本部分）
        for text in texts:
            self.bm25_documents.append(text)
            tokenized = self._simple_tokenize(text)
            self.bm25_tokenized_corpus.append(tokenized)
        
        if self.bm25_tokenized_corpus:
            self.bm25_index = BM25Okapi(self.bm25_tokenized_corpus)
        print(f"[EnhancedRAG] 已添加 {len(texts)} 个文档。向量与BM25索引就绪。")

    def clear_collection(self):
        """重写父类方法：同时清除BM25索引。"""
        super().clear_collection()
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_tokenized_corpus = []

    def _bm25_retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """使用BM25进行关键词检索，返回(文档, 分数)列表"""
        if self.bm25_index is None or not self.bm25_documents:
            return []
        query_tokens = self._simple_tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(self.bm25_documents[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        return results

    def _rerank(self, query: str, documents: List[str], top_k: int) -> List[str]:
        """使用重排序模型对文档进行精排。"""
        if not documents or len(documents) <= 1:
            return documents[:top_k]

        pairs = [[query, doc] for doc in documents]
        with torch.no_grad():
            inputs = self.reranker_tokenizer(pairs,
                                             padding=True,
                                             truncation=True,
                                             return_tensors='pt',
                                             max_length=512).to(self.device)
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy()

        ranked_indices = np.argsort(scores)[::-1]
        top_indices = ranked_indices[:top_k]
        return [documents[i] for i in top_indices]

    def hybrid_retrieve(self,
                        query: str,
                        vector_k: int = 10,
                        bm25_k: int = 10,
                        rerank_k: int = 5) -> List[str]:
        """
        核心增强方法：执行混合检索（向量+BM25）并进行重排序。
        :param query: 查询文本
        :param vector_k: 向量检索返回的初始文档数
        :param bm25_k: BM25检索返回的初始文档数
        :param rerank_k: 重排序后最终返回的文档数
        :return: 重排序后的最相关文档列表
        """
        all_candidates = {}

        # 1. 向量检索 (使用父类的collection)
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            vector_results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=vector_k
            )
            vector_docs = vector_results["documents"][0] if vector_results["documents"] else []
            for doc in vector_docs:
                all_candidates[doc] = all_candidates.get(doc, 0) + 1
        except Exception as e:
            print(f"[EnhancedRAG] 向量检索失败: {e}")
            vector_docs = []

        # 2. BM25检索
        bm25_results = self._bm25_retrieve(query, k=bm25_k)
        for doc, score in bm25_results:
            all_candidates[doc] = all_candidates.get(doc, 0) + score

        candidate_docs = list(all_candidates.keys())
        if not candidate_docs:
            print("[EnhancedRAG] 未检索到任何相关文档。")
            return []

        print(f"[EnhancedRAG] 混合检索得到 {len(candidate_docs)} 个候选文档。")

        # 3. 重排序
        if len(candidate_docs) > 1:
            print("[EnhancedRAG] 正在进行重排序...")
            final_docs = self._rerank(query, candidate_docs, top_k=rerank_k)
        else:
            final_docs = candidate_docs[:rerank_k]

        print(f"[EnhancedRAG] 重排序完成，返回 {len(final_docs)} 个文档。")
        return final_docs

    def retrieve_with_multimodal_priority(self, query: str, k: int = 5) -> List[str]:
        """
        新增方法：优先返回多模态相关信息
        """
        results = self.retrieve_with_metadata(query, k=k*2)
        
        if not results["documents"]:
            return []
        
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        # 分离多模态和普通文本
        multimodal_docs = []
        text_docs = []
        
        for doc, meta in zip(docs, metadatas):
            if meta.get("source") == "multimodal":
                multimodal_docs.append(doc)
            else:
                text_docs.append(doc)
        
        # 优先返回多模态信息，然后补充普通文本
        final_docs = multimodal_docs[:k] + text_docs[:max(0, k-len(multimodal_docs))]
        return final_docs[:k]