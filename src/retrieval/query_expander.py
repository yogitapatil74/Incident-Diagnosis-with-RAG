"""
src/retrieval/query_expander.py
────────────────────────────────
Hypothetical Document Embedding (HyDE) query expansion.
When initial retrieval returns insufficient results, HyDE generates
a synthetic exemplar document that is then used as the retrieval query.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
Relevance Labels", ACL 2023.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

HYDE_SYSTEM_PROMPT = """You are an expert Site Reliability Engineer.
Given an incident description, write a SHORT hypothetical runbook entry or
incident resolution summary (3-5 sentences) that would appear in a knowledge
base for this type of issue. Focus on technical specifics: services involved,
likely root cause, and typical remediation steps. Do NOT add preamble."""


class HyDEQueryExpander:
    """
    Generates a hypothetical document for the given query using an LLM,
    then returns it as an expanded query string for dense retrieval.
    This broadens coverage for novel or ambiguous incident descriptions.
    """

    def __init__(self, llm_client):
        self._llm = llm_client

    def expand(self, query: str) -> str:
        """
        Generate a hypothetical exemplar document for the query.
        Falls back to the original query on LLM error.
        """
        try:
            hypothetical_doc = self._llm.complete(
                system_prompt=HYDE_SYSTEM_PROMPT,
                user_message=f"Incident: {query}",
                max_tokens=200,
                temperature=0.3,
            )
            expanded = f"{query}\n\n{hypothetical_doc}"
            logger.debug(
                "HyDE expansion generated",
                original_len=len(query),
                expanded_len=len(expanded),
            )
            return expanded
        except Exception as e:
            logger.warning("HyDE expansion failed, using original query", error=str(e))
            return query
