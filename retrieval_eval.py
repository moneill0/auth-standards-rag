"""Evaluate retrieval against a small set of hand-labeled relevant sections."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from statistics import mean
from typing import Any, Callable

from auth_standards_rag import DEFAULT_DB_PATH, search_docs


@dataclass(frozen=True)
class RelevantSection:
    source_name: str
    section_title: str
    section_number: str | None = None


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_sections: tuple[RelevantSection, ...]


EVALUATION_CASES = (
    EvaluationCase(
        "code_verifier code_challenge SHA-256",
        (RelevantSection("Proof Key for Code Exchange (PKCE)", "Server Verifies code_verifier before Returning the Tokens", "4.6"),),
    ),
    EvaluationCase(
        "client creates code verifier PKCE",
        (RelevantSection("Proof Key for Code Exchange (PKCE)", "Client Creates the Code Challenge", "4.2"),),
    ),
    EvaluationCase(
        "OAuth security best practices authorization code PKCE",
        (RelevantSection("Best Current Practice for OAuth 2.0 Security", "Authorization Code Grant", "2.1.1"),),
    ),
    EvaluationCase(
        "OpenID Connect ID Token issuer audience signature validation",
        (RelevantSection("OpenID Connect Core 1.0", "ID Token Validation", "3.1.3.7"),),
    ),
    EvaluationCase(
        "NIST risk assessment assurance levels",
        (
            RelevantSection("NIST SP 800-63-4: Digital Identity Guidelines", "Conduct Initial Impact Assessment"),
            RelevantSection("NIST SP 800-63-4: Digital Identity Guidelines", "Initial Assurance Level Selection"),
            RelevantSection("NIST SP 800-63-4: Digital Identity Guidelines", "Digital Identity Risk Management", "5"),
        ),
    ),
    EvaluationCase(
        "DPoP proof JWT access token binding",
        (RelevantSection("OAuth 2.0 Demonstrating Proof of Possession (DPoP)", "DPoP Proof JWTs", "4"),),
    ),
)


def _section_key(result: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return result.get("source_name"), result.get("section_number"), result.get("section_title")


def _matches(result: dict[str, Any], expected: RelevantSection) -> bool:
    return (
        result.get("source_name") == expected.source_name
        and result.get("section_title") == expected.section_title
        and (expected.section_number is None or result.get("section_number") == expected.section_number)
    )


def evaluate_retrieval(
    top_k: int = 5,
    db_path: str = str(DEFAULT_DB_PATH),
    cases: tuple[EvaluationCase, ...] = EVALUATION_CASES,
    search: Callable[..., list[dict[str, Any]]] = search_docs,
) -> dict[str, Any]:
    """Report hit rate, MRR, precision, and recall for the labeled sections."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10.")
    if not cases:
        raise ValueError("At least one evaluation case is required.")
    if any(not case.relevant_sections for case in cases):
        raise ValueError("Every evaluation case must have at least one relevant section.")

    case_results: list[dict[str, Any]] = []
    for case in cases:
        raw_results = search(case.query, top_k=top_k, db_path=db_path)
        retrieved: list[dict[str, Any]] = []
        seen_sections: set[tuple[str | None, str | None, str | None]] = set()
        for result in raw_results:
            key = _section_key(result)
            if key not in seen_sections:
                seen_sections.add(key)
                retrieved.append(result)

        relevant_ranks: list[dict[str, Any]] = []
        matched_expected: set[RelevantSection] = set()
        for rank, result in enumerate(retrieved, start=1):
            for expected in case.relevant_sections:
                if expected not in matched_expected and _matches(result, expected):
                    matched_expected.add(expected)
                    relevant_ranks.append(
                        {
                            "rank": rank,
                            "source_name": expected.source_name,
                            "section_number": expected.section_number or result.get("section_number"),
                            "section_title": expected.section_title,
                        }
                    )

        first_relevant_rank = min(
            (match["rank"] for match in relevant_ranks),
            default=None,
        )
        case_results.append(
            {
                "query": case.query,
                "hit": first_relevant_rank is not None,
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": 1 / first_relevant_rank if first_relevant_rank else 0.0,
                "recall_at_k": len(matched_expected) / len(case.relevant_sections),
                "relevant_sections_found": relevant_ranks,
            }
        )

    return {
        "top_k": top_k,
        "case_count": len(case_results),
        "metrics": {
            "hit_rate_at_k": mean(float(result["hit"]) for result in case_results),
            "mean_reciprocal_rank_at_k": mean(result["reciprocal_rank"] for result in case_results),
            "mean_recall_at_k": mean(result["recall_at_k"] for result in case_results),
        },
        "results": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    print(json.dumps(evaluate_retrieval(top_k=args.top_k, db_path=args.db), indent=2))


if __name__ == "__main__":
    main()
