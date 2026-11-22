# auth-standards-rag

A small, local retrieval starter for OAuth 2.0, OpenID Connect, and NIST digital
identity standards. It fetches nine public sources, chunks their text, and
indexes the passages with SQLite FTS5; no embedding service or API key is
required.

## Index the sources

From this directory, run:

```bash
python -m auth_standards_rag index
```

The script downloads the sources in `SOURCES` and writes the local index to
`data/standards.sqlite3`. Run the command again to refresh the corpus. If a
source cannot be fetched, indexing fails with the source name and URL rather
than silently creating an incomplete index.

The corpus includes RFC 6749, RFC 9700, RFC 7636, RFC 9449, OpenID Connect Core
1.0, and all four NIST SP 800-63 Revision 4 volumes: SP 800-63, 63A, 63B, and
63C. OpenID Connect Core is an OpenID Foundation specification, not an RFC.

RFC page footer lines are removed, RFC section headings are parsed from the
plain-text documents, and HTML headings provide section titles and available
section numbers. Each indexed passage records its source version, UTC fetch
timestamp, SHA-256 hash of the fetched response, and section number/title where
available. Rebuilding the index replaces its passage table and refreshes this
metadata. The hash identifies the exact fetched response, but the project does
not archive fetched source files; retrieving the same upstream revision is
still necessary to reproduce the complete corpus later.

## Search

```bash
python -m auth_standards_rag search "when should clients use PKCE?" --top-k 5
```

The `search_docs(query, top_k=5)` function is the single retrieval tool. It
returns a list of passages with `source_name`, official `source_url`,
`source_version`, `fetched_at`, `content_sha256`, `section_number`,
`section_title`, and `passage`. Section numbers may be absent for headings
that have no explicit number in the source.
`SEARCH_DOCS_TOOL` provides a JSON-schema-style tool description for an agent
host to register; the search function itself remains usable without choosing a
particular LLM or agent SDK.

This starter uses keyword/full-text ranking, not embeddings. Treat results as
retrieval evidence rather than a complete compliance answer, and check the
linked standard text for normative requirements.

## Generate an answer

The default is a fully local Ollama model:

```bash
ollama pull llama3.2:3b
ollama serve
python -m auth_standards_rag answer "Why should clients use PKCE?" --top-k 5
```

Set `OLLAMA_MODEL` to use another installed Ollama model or `OLLAMA_HOST` for
another local Ollama endpoint:

```bash
export OLLAMA_MODEL="qwen2.5:7b"
export OLLAMA_HOST="http://localhost:11434"
```

The answer command first retrieves passages using the existing local index. If
no passage exceeds `--min-score` (default `0.0`), it prints the exact abstention
`not stated in the indexed documents` and does not call a model. Otherwise, it
sends the question and numbered passages to the selected model and requires
citations such as `[1]`. It rejects answers containing no valid citation or
citations outside the retrieved passage list and abstains.

No API key is needed with the default Ollama provider. Ollama and the selected
model must be installed and running. Search and the test suite work even when
no local or hosted model is available. No answer-generation SDK is required.

## Use search from an AI assistant with MCP

Install the MCP server dependency in the Python environment that the assistant
will use to launch the server:

```bash
python3 -m pip install -r requirements.txt
```

In VS Code, open this folder as the workspace and start the
`auth-standards-rag` MCP server from the MCP servers UI. The
`.vscode/mcp.json` configuration launches `mcp_server.py` over stdio using
`python3` from `PATH`; make sure that interpreter has the requirements
installed. The server exposes one MCP tool, `search_docs`, which calls the
existing local search function and returns its citation metadata. The index
must already exist; build it with the index command above.

For another MCP-compatible assistant, configure a stdio server with command
`python3` and argument `/absolute/path/to/auth-standards-rag/mcp_server.py`,
using the project directory as its working directory. Install `requirements.txt`
in the Python environment used by that command.

## Evaluate retrieval

```bash
python retrieval_eval.py --top-k 5
```

The evaluator runs a small hand-labeled benchmark in `retrieval_eval.py`.
Expected results are source sections, not exact chunks, so multiple chunks
from one section do not inflate scores. It reports hit rate@k, mean reciprocal
rank@k, and mean recall@k, plus per-query results. Precision is omitted because
the benchmark labels known relevant sections but does not claim to label every
relevant section in the corpus. The labels are an initial, human-curated
sample; expand and review them as the corpus and intended questions grow. These
retrieval metrics do not measure answer correctness or citation faithfulness.

Hit rate measures how often a query retrieves any expected section; reciprocal
rank rewards finding one near the top; recall measures how many labeled
sections were retrieved.

## Tests

```bash
python -m unittest discover -s tests -v
```
