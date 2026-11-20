import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_company_id
from app.config import settings
from app.core.policy_store import policy_store
from app.core.vector_store import vector_store
from app.rag.hybrid_search import hybrid_searcher
from app.rag.llm_client import get_llm_client
from app.rag.prompts import (
    FALLBACK_NOT_COVERED,
    FALLBACK_SYSTEM_ERROR,
    build_rag_prompt,
)
from app.schemas.query import Citation, QueryRequest, QueryResponse

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Company Commute Policy",
    description="Answers employee commute policy questions strictly grounded in their company's active policy."
)
def query_policy_assistant(
    payload: QueryRequest,
    company_id: str = Depends(get_company_id)
) -> QueryResponse:
    """Grounded RAG query endpoint with strict server-side tenant isolation."""
    start_time = time.perf_counter()
    c_id = company_id.strip().lower()

    # 1. Look up active policy version
    active_version = policy_store.get_active_version(c_id)
    if active_version is None or active_version == 0:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return QueryResponse(
            answer=FALLBACK_NOT_COVERED,
            citations=[],
            company_id=c_id,
            policy_version=0,
            status="not_covered",
            latency_ms=round(elapsed_ms, 2),
            llm_calls_made=0
        )

    try:
        # 2. Retrieval: Dense Chroma or Tenant-Isolated Hybrid Search (capped at TOP_K_CHUNKS <= 3)
        top_k = min(settings.TOP_K_CHUNKS, 3)
        if settings.ENABLE_HYBRID_SEARCH:
            retrieved_chunks = hybrid_searcher.search(
                company_id=c_id,
                query_text=payload.question,
                top_k=top_k
            )
        else:
            retrieved_chunks = vector_store.query_policy(
                company_id=c_id,
                query_text=payload.question,
                top_k=top_k
            )

        # 3. Check for empty retrieval or excessive semantic distance
        if not retrieved_chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return QueryResponse(
                answer=FALLBACK_NOT_COVERED,
                citations=[],
                company_id=c_id,
                policy_version=active_version,
                status="not_covered",
                latency_ms=round(elapsed_ms, 2),
                llm_calls_made=0
            )

        # If best match distance is beyond threshold, deterministic fallback without LLM call
        top_distance = retrieved_chunks[0].get("distance", 0.0)
        if top_distance > settings.SIMILARITY_DISTANCE_THRESHOLD:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return QueryResponse(
                answer=FALLBACK_NOT_COVERED,
                citations=[],
                company_id=c_id,
                policy_version=active_version,
                status="not_covered",
                latency_ms=round(elapsed_ms, 2),
                llm_calls_made=0
            )

        # 4. Generate Answer via LLM Client (1 call budget)
        prompt = build_rag_prompt(question=payload.question, retrieved_chunks=retrieved_chunks)
        llm = get_llm_client()
        answer = llm.generate_answer(prompt=prompt, retrieved_chunks=retrieved_chunks)
        llm_calls = 1

        # 5. Format and deduplicate citations (maximum 3 chunks)
        citations: List[Citation] = []
        seen_chunks = set()

        if FALLBACK_NOT_COVERED not in answer:
            for hit in retrieved_chunks[:3]:
                cid = hit.get("chunk_id", "")
                if cid and cid not in seen_chunks:
                    seen_chunks.add(cid)
                    meta = hit.get("metadata", {})
                    citations.append(
                        Citation(
                            document=meta.get("doc_name", f"{c_id.capitalize()} Commute Policy"),
                            section=f"{meta.get('section_id', '')}. {meta.get('section_title', '')}".strip(". "),
                            chunk_id=cid
                        )
                    )

        resp_status = "not_covered" if FALLBACK_NOT_COVERED in answer else "answered"
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return QueryResponse(
            answer=answer,
            citations=citations if resp_status == "answered" else [],
            company_id=c_id,
            policy_version=active_version,
            status=resp_status,
            latency_ms=round(elapsed_ms, 2),
            llm_calls_made=llm_calls
        )

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return QueryResponse(
            answer=FALLBACK_SYSTEM_ERROR,
            citations=[],
            company_id=c_id,
            policy_version=active_version or 0,
            status="system_fallback",
            latency_ms=round(elapsed_ms, 2),
            llm_calls_made=0
        )
