import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from app.config import settings
from app.core.policy_store import policy_store
from app.rag.chunker import PolicyChunk, SectionAwareChunker
from app.schemas.policy import PolicyIngestResponse


class VectorStore:
    """Manages ChromaDB vector collections with atomic versioning, tenant filtering, and rollback."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self.chunker = SectionAwareChunker()
        
        # Initialize default embedding function (ONNX all-MiniLM-L6-v2)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize Chroma client with persistence."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def clean_staged_chunks(self, company_id: str, version: int) -> None:
        """Purge all chunks matching (company_id, version).
        
        Guarantees safe retry on failed ingestion by wiping any orphaned or
        partially written chunks before re-attempting ingestion.
        """
        c_id = company_id.strip().lower()
        where_filter = {
            "$and": [
                {"company_id": {"$eq": c_id}},
                {"policy_version": {"$eq": int(version)}}
            ]
        }
        try:
            self.collection.delete(where=where_filter)
        except Exception:
            # If collection is empty or filter matches nothing, Chroma might pass or raise
            pass

    def ingest_policy(
        self,
        company_id: str,
        markdown_text: str,
        document_name: Optional[str] = None,
        simulate_failure_at_chunk: Optional[int] = None
    ) -> PolicyIngestResponse:
        """Atomic Two-Phase Policy Ingestion:
        1. Determine next version N+1.
        2. Pre-clean any leftover chunks for N+1.
        3. Parse & embed chunks into Chroma under version N+1.
        4. If insertion fails, rollback staged chunks; active version remains unchanged.
        5. Atomic Pointer Swap: Update active version pointer in PolicyStore.
        """
        c_id = company_id.strip().lower()
        target_version = policy_store.get_next_staging_version(c_id)

        # Step 1: Pre-clean target version space
        self.clean_staged_chunks(c_id, target_version)

        # Step 2: Parse markdown into section-aware chunks
        chunks = self.chunker.chunk_markdown(
            markdown_text=markdown_text,
            company_id=c_id,
            version=target_version,
            default_doc_name=document_name
        )

        if not chunks:
            raise ValueError(f"Document contains no valid sections or content to index for {c_id}.")

        effective_doc_name = chunks[0].doc_name

        # Step 3: Insert into Chroma with rollback guard
        try:
            for idx, chk in enumerate(chunks):
                # Fault injection hook for automated failure/recovery tests
                if simulate_failure_at_chunk is not None and idx >= simulate_failure_at_chunk:
                    raise RuntimeError(f"Simulated storage/embedding failure at chunk index {idx}")

                self.collection.add(
                    ids=[chk.chunk_id],
                    documents=[chk.text],
                    metadatas=[chk.to_metadata(company_id=c_id, version=target_version)]
                )
        except Exception as exc:
            # Rollback: Clean up any partially staged chunks so Chroma is never corrupted
            self.clean_staged_chunks(c_id, target_version)
            raise RuntimeError(
                f"Ingestion failed for {c_id} v{target_version}. Staged chunks rolled back cleanly: {str(exc)}"
            ) from exc

        # Step 4: Atomic Commit (Pointer Swap)
        policy_store.commit_version(
            company_id=c_id,
            version=target_version,
            doc_name=effective_doc_name,
            chunks_count=len(chunks)
        )

        # Invalidate any cached BM25 index for this tenant
        try:
            from app.rag.hybrid_search import hybrid_searcher
            hybrid_searcher.invalidate_cache(c_id)
        except Exception:
            pass

        return PolicyIngestResponse(
            company_id=c_id,
            active_version=target_version,
            document_name=effective_doc_name,
            chunks_ingested=len(chunks),
            status="active",
            message=f"Policy version {target_version} successfully indexed and activated for {c_id}."
        )

    def query_policy(
        self,
        company_id: str,
        query_text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks strictly filtered by the verified company_id and active_version.
        
        Zero cross-tenant leakage: Server-enforced Chroma where-clause prevents retrieval
        from other tenants or dead/old policy versions.
        """
        c_id = company_id.strip().lower()
        active_version = policy_store.get_active_version(c_id)
        if active_version is None:
            return []

        where_filter = {
            "$and": [
                {"company_id": {"$eq": c_id}},
                {"policy_version": {"$eq": int(active_version)}}
            ]
        }

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )

        hits: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
            })

        return hits

    def reset_collection(self) -> None:
        """Reset the Chroma collection (used in tests)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._init_chroma()


vector_store = VectorStore()
