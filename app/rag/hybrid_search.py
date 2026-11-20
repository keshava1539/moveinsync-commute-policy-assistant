import re
from typing import Any, Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi

from app.core.policy_store import policy_store
from app.core.vector_store import vector_store


class TenantIsolatedHybridSearch:
    """Hybrid Search combining dense Chroma vectors and sparse BM25 keyword matching.
    
    CRITICAL SECURITY INVARIANT:
    The BM25 index is strictly partitioned and built per (company_id, active_version).
    It NEVER touches or indexes chunks from any other tenant, mathematically
    eliminating the risk of cross-company data leakage during lexical search.
    """

    def __init__(self):
        # Cache of BM25Okapi instances keyed by (company_id, version)
        self._index_cache: Dict[Tuple[str, int], Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenizer for BM25."""
        return re.findall(r"\w+", text.lower())

    def invalidate_cache(self, company_id: str, version: Optional[int] = None) -> None:
        """Invalidate cached BM25 index for a tenant when a policy is updated."""
        c_id = company_id.strip().lower()
        keys_to_remove = [k for k in self._index_cache if k[0] == c_id and (version is None or k[1] == version)]
        for k in keys_to_remove:
            del self._index_cache[k]

    def _get_or_build_tenant_bm25(
        self,
        company_id: str,
        version: int
    ) -> Optional[Tuple[BM25Okapi, List[Dict[str, Any]]]]:
        """Fetch or lazily build the BM25 index strictly over the tenant's active chunks."""
        c_id = company_id.strip().lower()
        cache_key = (c_id, version)
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]

        # Retrieve ALL chunks belonging strictly to (company_id, version) from Chroma
        where_filter = {
            "$and": [
                {"company_id": {"$eq": c_id}},
                {"policy_version": {"$eq": int(version)}}
            ]
        }
        res = vector_store.collection.get(where=where_filter)
        if not res or not res["ids"]:
            return None

        chunks: List[Dict[str, Any]] = []
        tokenized_corpus: List[List[str]] = []

        for i in range(len(res["ids"])):
            chunk_data = {
                "chunk_id": res["ids"][i],
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
                "distance": 0.0
            }
            chunks.append(chunk_data)
            tokenized_corpus.append(self._tokenize(chunk_data["document"]))

        if not tokenized_corpus:
            return None

        bm25_index = BM25Okapi(tokenized_corpus)
        self._index_cache[cache_key] = (bm25_index, chunks)
        return self._index_cache[cache_key]

    def search(
        self,
        company_id: str,
        query_text: str,
        top_k: int = 3,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Perform Reciprocal Rank Fusion (RRF) between Dense Chroma and Tenant-Scoped BM25.
        
        RRF formula: Score(d) = 1 / (rrf_k + rank_dense) + 1 / (rrf_k + rank_bm25)
        """
        c_id = company_id.strip().lower()
        active_v = policy_store.get_active_version(c_id)
        if active_v is None:
            return []

        # 1. Dense Chroma Search (already tenant-isolated)
        dense_results = vector_store.query_policy(
            company_id=c_id,
            query_text=query_text,
            top_k=top_k * 2
        )

        # 2. Sparse BM25 Search (strictly isolated to tenant's active version)
        bm25_tuple = self._get_or_build_tenant_bm25(c_id, active_v)
        if not bm25_tuple:
            return dense_results[:top_k]

        bm25_index, tenant_chunks = bm25_tuple
        query_tokens = self._tokenize(query_text)
        bm25_scores = bm25_index.get_scores(query_tokens)

        # Rank BM25 chunks
        scored_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranked = [tenant_chunks[i] for i in scored_indices if bm25_scores[i] > 0]

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, item in enumerate(bm25_ranked):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort chunks by RRF score descending
        fused_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        final_results = [chunk_map[cid] for cid in fused_cids[:top_k]]
        return final_results


hybrid_searcher = TenantIsolatedHybridSearch()
