import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PolicyChunk:
    """Represents a discrete semantic chunk of a policy document with metadata."""
    chunk_id: str
    text: str
    doc_name: str
    section_id: str
    section_title: str
    chunk_index: int
    char_count: int

    def to_metadata(self, company_id: str, version: int) -> Dict[str, Any]:
        """Convert chunk attributes to Chroma-compatible flat metadata dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "company_id": company_id.strip().lower(),
            "policy_version": int(version),
            "doc_name": self.doc_name,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "chunk_index": int(self.chunk_index),
            "char_count": int(self.char_count)
        }


class SectionAwareChunker:
    """Markdown section-aware document chunker.
    
    Splits policy documents by headings (e.g., '# Title', '## 1. Section'),
    preserving hierarchical metadata on every chunk.
    """

    def __init__(self, max_chunk_chars: int = 1200, chunk_overlap_chars: int = 150):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def chunk_markdown(
        self,
        markdown_text: str,
        company_id: str,
        version: int,
        default_doc_name: Optional[str] = None
    ) -> List[PolicyChunk]:
        """Parse markdown into structured policy chunks with rich metadata."""
        lines = markdown_text.splitlines()
        doc_name = default_doc_name or f"{company_id.capitalize()} Commute Policy v{version}"
        
        # 1. Detect top-level title (# Title) if present
        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith("# ") and not line_strip.startswith("## "):
                doc_name = line_strip.lstrip("#").strip()
                break

        # 2. Split text into sections based on ## headings
        sections = []
        current_section_id = "0"
        current_section_title = "Overview"
        current_lines: List[str] = []

        heading_pattern = re.compile(r"^##\s+(?:(\d+(?:\.\d+)*)\.?\s*)?(.*)$")

        for line in lines:
            # Skip top-level # doc title from body lines
            if line.strip().startswith("# ") and not line.strip().startswith("## "):
                continue

            match = heading_pattern.match(line.strip())
            if match:
                # Save previous section if it has content
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_section_id, current_section_title, body))
                    current_lines = []
                
                sec_num, sec_title = match.groups()
                current_section_id = sec_num if sec_num else str(len(sections) + 1)
                current_section_title = sec_title.strip() if sec_title else "General"
            else:
                current_lines.append(line)

        # Append last section
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_section_id, current_section_title, body))

        # 3. Build chunks from sections
        chunks: List[PolicyChunk] = []
        chunk_counter = 0

        for sec_id, sec_title, sec_body in sections:
            # If section body is small enough, keep as single coherent semantic unit
            if len(sec_body) <= self.max_chunk_chars:
                chunk_id = f"{company_id}_v{version}_sec{sec_id}_{chunk_counter}"
                formatted_text = f"Document: {doc_name}\nSection: {sec_id}. {sec_title}\n\n{sec_body}"
                chunks.append(
                    PolicyChunk(
                        chunk_id=chunk_id,
                        text=formatted_text,
                        doc_name=doc_name,
                        section_id=sec_id,
                        section_title=sec_title,
                        chunk_index=chunk_counter,
                        char_count=len(formatted_text)
                    )
                )
                chunk_counter += 1
            else:
                # Split large section into paragraph-level chunks
                paragraphs = [p.strip() for p in sec_body.split("\n\n") if p.strip()]
                buffer = ""
                for p in paragraphs:
                    if len(buffer) + len(p) + 2 <= self.max_chunk_chars:
                        buffer = f"{buffer}\n\n{p}".strip()
                    else:
                        if buffer:
                            chunk_id = f"{company_id}_v{version}_sec{sec_id}_{chunk_counter}"
                            formatted_text = f"Document: {doc_name}\nSection: {sec_id}. {sec_title}\n\n{buffer}"
                            chunks.append(
                                PolicyChunk(
                                    chunk_id=chunk_id,
                                    text=formatted_text,
                                    doc_name=doc_name,
                                    section_id=sec_id,
                                    section_title=sec_title,
                                    chunk_index=chunk_counter,
                                    char_count=len(formatted_text)
                                )
                            )
                            chunk_counter += 1
                        buffer = p

                if buffer:
                    chunk_id = f"{company_id}_v{version}_sec{sec_id}_{chunk_counter}"
                    formatted_text = f"Document: {doc_name}\nSection: {sec_id}. {sec_title}\n\n{buffer}"
                    chunks.append(
                        PolicyChunk(
                            chunk_id=chunk_id,
                            text=formatted_text,
                            doc_name=doc_name,
                            section_id=sec_id,
                            section_title=sec_title,
                            chunk_index=chunk_counter,
                            char_count=len(formatted_text)
                        )
                    )
                    chunk_counter += 1

        return chunks
