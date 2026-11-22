"""Simple grounded answer generation over retrieved authentication standards."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from auth_standards_rag import DEFAULT_DB_PATH, search_docs_with_scores


ABSTAIN_MESSAGE = "not stated in the indexed documents"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _evidence_text(question: str, passages: list[dict[str, Any]]) -> tuple[str, str]:
    evidence = []
    for index, passage in enumerate(passages, start=1):
        section = passage.get("section_number")
        section_title = passage.get("section_title")
        section_label = ".".join(
            part for part in (
                str(section) if section else "",
                str(section_title) if section_title else "",
            )
            if part
        )
        evidence.append(
            f"[{index}] {passage['source_name']}"
            f"{(' — ' + section_label) if section_label else ''}\n"
            f"Source: {passage['source_url']}\n"
            f"Text: {passage['passage']}"
        )
    system = (
        "You answer authentication and security questions using only the supplied evidence. "
        "Do not use outside knowledge. Cite every substantive claim by evidence number, "
        "for example [1]. Use only evidence numbers that appear in the evidence list. "
        "If the evidence is insufficient, answer exactly: " + ABSTAIN_MESSAGE
    )
    user = f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence)
    return system, user


def call_ollama_model(question: str, passages: list[dict[str, Any]]) -> str:
    """Call a local Ollama chat model; no API key or network service is used."""
    system, user = _evidence_text(question, passages)
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0},
    }
    request = Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Unable to reach local Ollama at {host}. Start Ollama and install model '{model}'."
        ) from exc
    try:
        return body["message"]["content"].strip()
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Ollama returned an unexpected response.") from exc


def validate_citations(answer: str, passages: list[dict[str, Any]]) -> tuple[bool, list[int]]:
    """Check that citation numbers refer to retrieved passage evidence."""
    citation_numbers = [int(number) for number in CITATION_PATTERN.findall(answer)]
    unique = list(dict.fromkeys(citation_numbers))
    return bool(unique) and all(1 <= number <= len(passages) for number in unique), unique


def answer_question(
    question: str,
    top_k: int = 5,
    min_score: float = 0.0,
    db_path: str = str(DEFAULT_DB_PATH),
    search: Callable[..., list[dict[str, Any]]] = search_docs_with_scores,
    model_call: Callable[[str, list[dict[str, Any]]], str] | None = None,
) -> dict[str, Any]:
    """Retrieve evidence and produce a cited answer, or abstain first."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10.")

    passages = search(question, top_k=top_k, db_path=db_path)
    qualifying = [passage for passage in passages if float(passage.get("score", 0.0)) > min_score]
    if not qualifying:
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "reason": "No retrieved passage cleared the minimum score threshold.",
            "passages": [],
        }

    if model_call is None:
        model_call = call_ollama_model

    try:
        answer = model_call(question, qualifying)
    except RuntimeError as exc:
        return {
            "answer": str(exc),
            "abstained": True,
            "reason": "Answer generation is unavailable.",
            "passages": qualifying,
        }

    valid_citations, citation_numbers = validate_citations(answer, qualifying)
    if not valid_citations:
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "reason": "The model answer had no valid citations to retrieved passages.",
            "passages": qualifying,
        }

    return {
        "answer": answer,
        "abstained": False,
        "citations": citation_numbers,
        "passages": qualifying,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    print(json.dumps(answer_question(args.question, top_k=args.top_k, min_score=args.min_score, db_path=args.db), indent=2))


if __name__ == "__main__":
    main()
