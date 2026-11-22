import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auth_standards_rag import (
    FetchedDocument,
    Section,
    Source,
    _parse_html_sections,
    _parse_rfc_sections,
    build_index,
    chunk_text,
    search_docs,
)


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


if __name__ == "__main__":
    unittest.main()
