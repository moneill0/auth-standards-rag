"""MCP stdio server exposing the authentication standards search tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from auth_standards_rag import DEFAULT_DB_PATH, search_docs as search_index


mcp = FastMCP("auth-standards-rag")


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search OAuth, OpenID Connect, and NIST standards for relevant passages.

    Returns passages with source names, section metadata, source versions,
    fetch timestamps, and official links.
    """
    db_path = Path(os.environ.get("AUTH_STANDARDS_DB", DEFAULT_DB_PATH))
    return search_index(query, top_k=top_k, db_path=db_path)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
