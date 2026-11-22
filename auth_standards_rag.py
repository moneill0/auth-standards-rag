"""Fetch, index, and search a small corpus of authentication standards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "data" / "standards.sqlite3"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    version: str


SOURCES = (
    Source(
        "OAuth 2.0 Authorization Framework",
        "https://www.rfc-editor.org/rfc/rfc6749.txt",
        "RFC 6749 (October 2012)",
    ),
    Source(
        "Best Current Practice for OAuth 2.0 Security",
        "https://www.rfc-editor.org/rfc/rfc9700.txt",
        "RFC 9700 (January 2025)",
    ),
    Source(
        "Proof Key for Code Exchange (PKCE)",
        "https://www.rfc-editor.org/rfc/rfc7636.txt",
        "RFC 7636 (September 2015)",
    ),
    Source(
        "OAuth 2.0 Demonstrating Proof of Possession (DPoP)",
        "https://www.rfc-editor.org/rfc/rfc9449.txt",
        "RFC 9449 (September 2023)",
    ),
    Source(
        "OpenID Connect Core 1.0",
        "https://openid.net/specs/openid-connect-core-1_0.html",
        "OpenID Connect Core 1.0, incorporating errata set 2",
    ),
    Source(
        "NIST SP 800-63-4: Digital Identity Guidelines",
        "https://pages.nist.gov/800-63-4/sp800-63.html",
        "SP 800-63 Revision 4 (July 2025)",
    ),
    Source(
        "NIST SP 800-63A-4: Identity Proofing and Enrollment",
        "https://pages.nist.gov/800-63-4/sp800-63a.html",
        "SP 800-63A Revision 4 (July 2025)",
    ),
    Source(
        "NIST SP 800-63B-4: Authentication and Authenticator Management",
        "https://pages.nist.gov/800-63-4/sp800-63b.html",
        "SP 800-63B Revision 4 (July 2025)",
    ),
    Source(
        "NIST SP 800-63C-4: Federation and Assertions",
        "https://pages.nist.gov/800-63-4/sp800-63c.html",
        "SP 800-63C Revision 4 (July 2025)",
    ),
)

SEARCH_DOCS_TOOL: dict[str, Any] = {
    "name": "search_docs",
    "description": "Search indexed OAuth, OpenID Connect, and NIST digital identity standards. "
    "Return passages with source names, section context, source versions, fetch dates, and links.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question or terms to search for."},
            "top_k": {
                "type": "integer",
                "description": "Number of passages to return (1-10).",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Section:
    number: str | None
    title: str | None
    text: str


@dataclass(frozen=True)
class FetchedDocument:
    sections: tuple[Section, ...]
    fetched_at: str
    content_sha256: str


class _HTMLSectionExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "article", "br", "dd", "div", "dl", "dt", "h1", "h2", "h3", "h4",
        "h5", "h6", "li", "ol", "p", "pre", "section", "table", "td", "th",
        "tr", "ul",
    }
    _SKIP_TAGS = {"aside", "footer", "header", "nav", "noscript", "script", "style"}
    _VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self.parts: list[str] = []
        self.skip_depth = 0
        self.heading_level: int | None = None
        self.heading_parts: list[str] = []
        self.heading_id: str | None = None
        self.current_number: str | None = None
        self.current_title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skip_depth:
            if tag not in self._VOID_TAGS:
                self.skip_depth += 1
            return
        if tag in self._SKIP_TAGS:
            self.skip_depth = 1
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_section()
            self.heading_level = int(tag[1])
            self.heading_parts = []
            self.heading_id = dict(attrs).get("id")
        elif tag == "a" and self.heading_level is not None and not self.heading_id:
            self.heading_id = dict(attrs).get("id")
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if tag not in self._VOID_TAGS:
                self.skip_depth -= 1
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.heading_level is not None:
            heading = _normalize_plain_text("".join(self.heading_parts))
            self.current_number, self.current_title = _heading_metadata(heading, self.heading_id)
            self.heading_level = None
            self.heading_parts = []
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            if self.heading_level is not None:
                self.heading_parts.append(data)
            else:
                self.parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush_section()

    def _flush_section(self) -> None:
        text = _normalize_plain_text("".join(self.parts))
        if text and self.current_title and self.current_title.casefold() != "table of contents":
            self.sections.append(Section(self.current_number, self.current_title, text))
        self.parts = []


def _normalize_plain_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _heading_metadata(heading: str, anchor: str | None = None) -> tuple[str | None, str | None]:
    heading = re.sub(r"\s+", " ", heading).strip()
    numbered_heading = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", heading)
    if numbered_heading:
        return numbered_heading.group(1), numbered_heading.group(2).strip()
    numbered_anchor = re.match(r"^(?:sec|s-)(\d+(?:-\d+)*)$", anchor or "", re.IGNORECASE)
    number = numbered_anchor.group(1).replace("-", ".") if numbered_anchor else None
    return number, heading or None


def _strip_rfc_page_artifacts(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    page_header = re.compile(
        r"^\s*RFC\s+\d+\s+.+\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\s*$",
        re.IGNORECASE,
    )
    lines = [
        line for line in lines
        if not re.search(r"\[Page\s+\d+\]\s*$", line) and not page_header.match(line)
    ]
    return "\n".join(lines).replace("\f", "\n")


def _parse_rfc_sections(text: str) -> list[Section]:
    lines = _strip_rfc_page_artifacts(text).splitlines()
    introductions = [
        index for index, line in enumerate(lines)
        if re.match(r"^\s*1\.\s+Introduction\b", line)
    ]
    body_start = introductions[-1] if introductions else 0
    table_of_contents = [
        index for index, line in enumerate(lines[:body_start])
        if line.strip().casefold() == "table of contents"
    ]
    toc_entries: dict[str, str] = {}
    if table_of_contents:
        toc_start = table_of_contents[-1]
        pending_number: str | None = None
        pending_title = ""
        pending_indent = 0

        def save_toc_entry() -> None:
            if pending_number and pending_title:
                toc_entries[pending_number] = pending_title

        for line in lines[toc_start + 1:body_start]:
            entry = re.match(r"^(\s*)(\d+(?:\.\d+)*)\.\s+(.+?)\s*$", line)
            if entry:
                save_toc_entry()
                pending_number = entry.group(2)
                pending_title = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", entry.group(3)).strip()
                pending_indent = len(entry.group(1))
                continue
            indent = len(line) - len(line.lstrip())
            continuation = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", line.strip()).strip()
            if pending_number and continuation and indent > pending_indent + 2:
                pending_title = f"{pending_title} {continuation}"
            else:
                save_toc_entry()
                pending_number = None
                pending_title = ""
        save_toc_entry()
        toc_entries = {
            number: re.sub(r"\s+", " ", title).strip()
            for number, title in toc_entries.items()
        }

    sections: list[Section] = []
    current_number: str | None = None
    current_title: str | None = None
    current_parts: list[str] = []

    for line in lines[body_start:]:
        heading = re.match(r"^\s*(\d+(?:\.\d+)*)\.\s{2,}(.+?)\s*$", line)
        if heading:
            title = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", heading.group(2)).strip()
            title = re.sub(r"\s+", " ", title)
            toc_title = toc_entries.get(heading.group(1))
            matches_toc = (
                not toc_entries
                or (
                    toc_title is not None
                    and (
                        toc_title == title
                        or toc_title.startswith(f"{title} ")
                    )
                )
            )
            if title and matches_toc:
                section_text = _normalize_plain_text("\n".join(current_parts))
                if section_text:
                    sections.append(Section(current_number, current_title, section_text))
                current_number = heading.group(1)
                current_title = toc_title or title
                current_parts = []
                continue
        current_parts.append(line)

    section_text = _normalize_plain_text("\n".join(current_parts))
    if section_text:
        sections.append(Section(current_number, current_title, section_text))
    return sections


def _parse_html_sections(text: str) -> list[Section]:
    parser = _HTMLSectionExtractor()
    parser.feed(text)
    parser.close()
    return parser.sections


def fetch_source(source: Source) -> FetchedDocument:
    request = Request(source.url, headers={"User-Agent": "auth-standards-rag/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            raw_body = response.read()
            body = raw_body.decode(charset, errors="replace")
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Unable to fetch {source.name} from {source.url}: {exc}") from exc
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sections = (
        _parse_html_sections(body)
        if content_type in {"text/html", "application/xhtml+xml"}
        else _parse_rfc_sections(body)
    )
    return FetchedDocument(tuple(sections), fetched_at, sha256(raw_body).hexdigest())


def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0.")

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        next_start = max(start + 1, end - overlap)
        next_boundary = normalized.find(" ", next_start, end)
        start = next_boundary + 1 if next_boundary >= 0 else next_start
    return chunks


def _chunk_sections(sections: list[Section]) -> list[tuple[Section, str]]:
    return [
        (section, passage)
        for section in sections
        for passage in chunk_text(section.text)
    ]


def build_index(
    db_path: str | Path = DEFAULT_DB_PATH,
    sources: tuple[Source, ...] | list[Source] = SOURCES,
) -> dict[str, int]:
    documents = [(source, fetch_source(source)) for source in sources]
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS passages")
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS passages "
            "USING fts5(source_name UNINDEXED, source_url UNINDEXED, source_version UNINDEXED, "
            "fetched_at UNINDEXED, content_sha256 UNINDEXED, section_number UNINDEXED, "
            "section_title UNINDEXED, passage, tokenize='unicode61')"
        )
        counts: dict[str, int] = {}
        for source, document in documents:
            passages = _chunk_sections(list(document.sections))
            connection.executemany(
                "INSERT INTO passages(source_name, source_url, source_version, fetched_at, "
                "content_sha256, section_number, section_title, passage) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        source.name,
                        source.url,
                        source.version,
                        document.fetched_at,
                        document.content_sha256,
                        section.number,
                        section.title,
                        passage,
                    )
                    for section, passage in passages
                ],
            )
            counts[source.name] = len(passages)
    return counts


def search_docs(
    query: str,
    top_k: int = 5,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, str | None]]:
    """Return matching passages with source, version, fetch, and section metadata."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10.")

    terms = re.findall(r"[A-Za-z0-9]+", query)
    if not terms:
        raise ValueError("query must contain at least one searchable word or number.")
    all_terms_query = " AND ".join(f'"{term}"' for term in terms)
    any_terms_query = " OR ".join(f'"{term}"' for term in terms)

    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Search index not found at {path}. Run `python -m auth_standards_rag index` first.")
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT source_name, source_url, source_version, fetched_at, content_sha256, "
            "section_number, section_title, passage "
            "FROM passages WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT ?",
            (all_terms_query, top_k),
        ).fetchall()
        if not rows and all_terms_query != any_terms_query:
            rows = connection.execute(
                "SELECT source_name, source_url, source_version, fetched_at, content_sha256, "
                "section_number, section_title, passage "
                "FROM passages WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT ?",
                (any_terms_query, top_k),
            ).fetchall()
    return [
        {
            "source_name": source_name,
            "source_url": source_url,
            "source_version": source_version,
            "fetched_at": fetched_at,
            "content_sha256": content_sha256,
            "section_number": section_number,
            "section_title": section_title,
            "passage": passage,
        }
        for (
            source_name,
            source_url,
            source_version,
            fetched_at,
            content_sha256,
            section_number,
            section_title,
            passage,
        ) in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Fetch the configured standards and rebuild the local index.")
    search_parser = subparsers.add_parser("search", help="Search the local index.")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "index":
        print(json.dumps(build_index(), indent=2))
    else:
        print(json.dumps(search_docs(args.query, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
