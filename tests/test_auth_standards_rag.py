import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from auth_standards_rag import (
    FetchedDocument,
    ROOT,
    Section,
    Source,
    _parse_html_sections,
    _parse_rfc_sections,
    build_index,
    chunk_text,
    search_docs,
)
from retrieval_eval import EvaluationCase, RelevantSection, evaluate_retrieval
from answer import ABSTAIN_MESSAGE, answer_question, call_ollama_model, validate_citations


class ChunkTextTests(unittest.TestCase):
    def test_chunks_respect_size_and_preserve_content(self) -> None:
        text = "\n\n".join(["OAuth authorization code flow."] * 30)
        chunks = chunk_text(text, max_chars=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertIn("OAuth authorization code flow.", "\n".join(chunks))

    def test_oversized_paragraph_is_split(self) -> None:
        chunks = chunk_text("x" * 250, max_chars=100, overlap=20)

        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual("x" * 250, chunks[0] + "".join(chunk[20:] for chunk in chunks[1:]))

    def test_rfc_sections_strip_page_footers_and_keep_heading_metadata(self) -> None:
        text = (
            "Table of Contents\n"
            "   1. Introduction ....................................................3\n"
            "   2. Security Considerations ........................................4\n"
            "Abstract\nSummary\n"
            "1.  Introduction\nOAuth overview.\n"
            "1.  This numbered example is not a section.\n"
            "Hardt                        Standards Track                    [Page 1]\n"
            "RFC 6749                        OAuth 2.0                   October 2012\n"
            "2.  Security Considerations\nProtect tokens.\n"
        )

        sections = _parse_rfc_sections(text)

        self.assertEqual([("1", "Introduction"), ("2", "Security Considerations")],
                         [(section.number, section.title) for section in sections])
        self.assertTrue(all("[Page" not in section.text for section in sections))
        self.assertTrue(all("October 2012" not in section.text for section in sections))
        self.assertNotIn("Summary", sections[0].text)
        self.assertNotIn("Table of Contents", sections[0].text)
        self.assertIn("OAuth overview.", sections[0].text)
        self.assertIn("This numbered example is not a section.", sections[0].text)

    def test_html_headings_provide_titles_and_section_numbers(self) -> None:
        html = (
            '<h1 id="sec4">4. Digital Identity</h1><p>Framework overview.</p>'
            '<h2 id="s-4-1">Risk Levels</h2><p>Assess risk before selecting levels.</p>'
        )

        sections = _parse_html_sections(html)

        self.assertEqual([("4", "Digital Identity"), ("4.1", "Risk Levels")],
                         [(section.number, section.title) for section in sections])
        self.assertIn("Assess risk", sections[1].text)

    def test_html_discards_unsectioned_preamble_and_table_of_contents(self) -> None:
        html = (
            "Navigation text"
            "<h1 id=\"abstract\">Abstract</h1><p>Useful abstract.</p>"
            "<h2>Table of Contents</h2><p>1. Introduction 2. Security</p>"
            "<h2 id=\"s-1\">1. Introduction</h2><p>Specification text.</p>"
        )

        sections = _parse_html_sections(html)

        self.assertEqual(["Abstract", "Introduction"], [section.title for section in sections])
        self.assertNotIn("Navigation text", " ".join(section.text for section in sections))
        self.assertNotIn("Table of Contents", " ".join(section.text for section in sections))


class SearchDocsTests(unittest.TestCase):
    def test_returns_matching_passages_with_source_metadata(self) -> None:
        source = Source("OAuth Test Source", "https://example.test/oauth", "Test revision 1")
        document = FetchedDocument(
            (Section("2.1", "PKCE", "PKCE protects OAuth public clients."),),
            "2026-11-21T12:00:00+00:00",
            "a" * 64,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "index.sqlite3"
            with patch("auth_standards_rag.fetch_source", return_value=document):
                build_index(db_path, [source])

            results = search_docs("OAuth PKCE", db_path=db_path)

        self.assertEqual(1, len(results))
        self.assertEqual(source.name, results[0]["source_name"])
        self.assertEqual(source.url, results[0]["source_url"])
        self.assertEqual(source.version, results[0]["source_version"])
        self.assertEqual("2026-11-21T12:00:00+00:00", results[0]["fetched_at"])
        self.assertEqual("a" * 64, results[0]["content_sha256"])
        self.assertEqual("2.1", results[0]["section_number"])
        self.assertEqual("PKCE", results[0]["section_title"])
        self.assertIn("PKCE", results[0]["passage"])

    def test_rejects_out_of_range_top_k(self) -> None:
        with self.assertRaises(ValueError):
            search_docs("OAuth", top_k=11)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_computes_metrics_at_unique_section_level(self) -> None:
        case = EvaluationCase(
            "PKCE",
            (
                RelevantSection("OAuth", "Authorization Code Grant", "2.1"),
                RelevantSection("PKCE", "Token Verification", "4.6"),
            ),
        )

        def fake_search(query: str, top_k: int, db_path: str) -> list[dict[str, str]]:
            return [
                {"source_name": "OAuth", "section_number": "2.1", "section_title": "Authorization Code Grant"},
                {"source_name": "OAuth", "section_number": "2.1", "section_title": "Authorization Code Grant"},
                {"source_name": "PKCE", "section_number": "4.6", "section_title": "Token Verification"},
            ][:top_k]

        report = evaluate_retrieval(top_k=3, cases=(case,), search=fake_search)

        self.assertEqual(1.0, report["metrics"]["hit_rate_at_k"])
        self.assertEqual(1.0, report["metrics"]["mean_reciprocal_rank_at_k"])
        self.assertEqual(1.0, report["metrics"]["mean_recall_at_k"])
        self.assertEqual([1, 2], [item["rank"] for item in report["results"][0]["relevant_sections_found"]])

    def test_reports_misses_and_rejects_empty_benchmark(self) -> None:
        case = EvaluationCase("missing", (RelevantSection("NIST", "Authentication"),))
        report = evaluate_retrieval(
            top_k=2,
            cases=(case,),
            search=lambda query, top_k, db_path: [],
        )

        self.assertEqual(0.0, report["metrics"]["hit_rate_at_k"])
        self.assertEqual(0.0, report["metrics"]["mean_reciprocal_rank_at_k"])
        with self.assertRaises(ValueError):
            evaluate_retrieval(cases=(), search=lambda query, top_k, db_path: [])


class AnswerGenerationTests(unittest.TestCase):
    def test_answer_defaults_to_ollama_when_no_provider_is_set(self) -> None:
        passage = {
            "source_name": "RFC Test", "source_url": "https://example.test",
            "section_number": "2", "section_title": "PKCE",
            "passage": "Use PKCE.", "score": 1.0,
        }
        with patch.dict(os.environ, {}, clear=False):
            with patch("answer.call_ollama_model", return_value="Use PKCE [1].") as ollama:
                result = answer_question(
                    "Why use PKCE?",
                    search=lambda query, top_k, db_path: [passage],
                )

        ollama.assert_called_once()
        self.assertFalse(result["abstained"])
        self.assertEqual([1], result["citations"])

    def test_ollama_request_uses_local_api_and_returns_content(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"message":{"content":"Use PKCE [1]."}}'

        passage = {
            "source_name": "RFC Test", "source_url": "https://example.test",
            "section_number": "2", "section_title": "PKCE", "passage": "Use PKCE.",
        }
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:11434", "OLLAMA_MODEL": "llama3.2:3b"}):
            with patch("answer.urlopen", return_value=Response()) as mocked_urlopen:
                answer = call_ollama_model("Why use PKCE?", [passage])

        self.assertEqual("Use PKCE [1].", answer)
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual("http://localhost:11434/api/chat", request.full_url)

    def test_abstains_below_score_threshold_without_calling_model(self) -> None:
        model_calls = []

        def no_relevant_passages(query: str, top_k: int, db_path: str) -> list[dict[str, object]]:
            return [{"source_name": "NIST", "section_number": "5", "section_title": "Risk", "passage": "text", "score": 0.1}]

        result = answer_question(
            "What is the answer?",
            min_score=0.5,
            search=no_relevant_passages,
            model_call=lambda question, passages: model_calls.append(question) or "should not run",
        )

        self.assertTrue(result["abstained"])
        self.assertEqual(ABSTAIN_MESSAGE, result["answer"])
        self.assertEqual([], model_calls)

    def test_mocked_model_answer_is_cited_to_retrieved_section(self) -> None:
        passage = {
            "source_name": "Proof Key for Code Exchange (PKCE)",
            "source_url": "https://www.rfc-editor.org/rfc/rfc7636.txt",
            "section_number": "4.6",
            "section_title": "Server Verifies code_verifier",
            "passage": "The server verifies the code verifier.",
            "score": 2.5,
        }

        def relevant_search(query: str, top_k: int, db_path: str) -> list[dict[str, object]]:
            return [passage]

        result = answer_question(
            "How does PKCE verification work?",
            search=relevant_search,
            model_call=lambda question, passages: "The server verifies the code verifier [1].",
        )

        self.assertFalse(result["abstained"])
        self.assertEqual([1], result["citations"])
        valid, numbers = validate_citations(result["answer"], result["passages"])
        self.assertTrue(valid)
        self.assertEqual([1], numbers)
        self.assertEqual(passage["source_name"], result["passages"][0]["source_name"])

    def test_rejects_answer_with_out_of_range_citation(self) -> None:
        passage = {
            "source_name": "NIST", "source_url": "https://example.test",
            "section_number": "5", "section_title": "Risk",
            "passage": "Text", "score": 1.0,
        }
        result = answer_question(
            "What is the risk?",
            search=lambda query, top_k, db_path: [passage],
            model_call=lambda question, passages: "The answer is unsupported [2].",
        )

        self.assertTrue(result["abstained"])
        self.assertEqual(ABSTAIN_MESSAGE, result["answer"])


class MCPServerTests(unittest.TestCase):
    def test_stdio_server_exposes_and_runs_search_docs(self) -> None:
        source = Source("MCP Test Source", "https://example.test/mcp", "Test revision")
        document = FetchedDocument(
            (Section("2", "Token Security", "Protect OAuth access tokens from replay."),),
            "2026-11-21T12:00:00+00:00",
            "b" * 64,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "index.sqlite3"
            with patch("auth_standards_rag.fetch_source", return_value=document):
                build_index(db_path, [source])

            async def call_tool() -> tuple[list[str], str]:
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[str(ROOT / "mcp_server.py")],
                    cwd=str(ROOT),
                    env={**os.environ, "AUTH_STANDARDS_DB": str(db_path)},
                )
                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        result = await session.call_tool(
                            "search_docs",
                            {"query": "OAuth access tokens replay", "top_k": 3},
                        )
                        self.assertFalse(result.isError)
                        return [tool.name for tool in tools.tools], result.content[0].text

            tool_names, result_text = asyncio.run(call_tool())

        self.assertEqual(["search_docs"], tool_names)
        self.assertIn("MCP Test Source", result_text)
        self.assertIn("Token Security", result_text)


if __name__ == "__main__":
    unittest.main()
