"""Client for the FineWiki BM25 index server, plus the tool schemas shown to the model.

The tool set, result count and page window are fixed here so every model sees the same environment.
"""
import json

import httpx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Keyword search (BM25): use specific keywords and names rather than full sentences. "
                "Returns the top passages, each with its page title and section."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keyword search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_page",
            "description": (
                "Read a page by its exact title, as shown in search results. Long pages come back in "
                "windows; pass the returned `start` value to keep reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Exact page title."},
                    "start": {"type": "integer", "description": "Chunk index to start from (default 0)."},
                },
                "required": ["title"],
            },
        },
    },
]


class SearchEnv:
    def __init__(self, base_url: str, k: int = 5, page_words: int = 1200, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.k = k
        self.page_words = page_words
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def health(self) -> dict:
        r = await self.client.get("/health")
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self.client.aclose()

    async def call(self, name: str, arguments: str) -> tuple[str, dict]:
        """Run one tool call. Returns (text shown to the model, metadata for logging)."""
        try:
            args = json.loads(arguments) if arguments else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return f"Error: could not parse tool arguments as JSON ({e}).", {"error": "bad_arguments"}
        try:
            if name == "search":
                return await self._search(args)
            if name == "open_page":
                return await self._open_page(args)
            return f"Error: unknown tool '{name}'. Available tools: search, open_page.", {"error": "unknown_tool"}
        except httpx.HTTPError as e:
            return f"Error: the search service failed ({type(e).__name__}).", {"error": "service"}

    async def _search(self, args: dict):
        q = str(args.get("query", "")).strip()
        if not q:
            return "Error: `query` is required.", {"error": "empty_query"}
        r = await self.client.get("/search", params={"q": q, "k": self.k})
        r.raise_for_status()
        results = r.json()["results"]
        if not results:
            return f"No results for: {q}", {"query": q, "titles": []}
        lines = []
        for i, h in enumerate(results, 1):
            head = h["title"] + (f" (section: {h['section']})" if h["section"] else "")
            lines.append(f"[{i}] {head}\n{h['text']}")
        return "\n\n".join(lines), {"query": q, "titles": [h["title"] for h in results]}

    async def _open_page(self, args: dict):
        title = str(args.get("title", "")).strip()
        if not title:
            return "Error: `title` is required.", {"error": "empty_title"}
        try:
            start = int(args.get("start", 0) or 0)
        except (TypeError, ValueError):
            start = 0
        params = {"title": title, "start": start, "max_words": self.page_words}
        r = await self.client.get("/open_page", params=params)
        # A title copied with its section attached ("Page > Section" or "Page (section: ...)"): retry with the page part.
        if r.status_code == 404:
            base = title.split(" (section:")[0].split(" > ")[0].strip()
            if base and base != title:
                r = await self.client.get("/open_page", params={**params, "title": base})
        if r.status_code == 404:
            detail = r.json().get("detail", {})
            sugg = detail.get("did_you_mean", []) if isinstance(detail, dict) else []
            msg = f"No article titled '{title}'."
            if sugg:
                msg += " Similar titles: " + "; ".join(sugg)
            return msg, {"title": title, "found": False}
        r.raise_for_status()
        p = r.json()
        nxt = p["next_start"]
        footer = (f"\n\n[Showing chunks {p['start']}-{nxt - 1} of {p['total_chunks']}. "
                  f"Call open_page with start={nxt} to continue.]") if nxt is not None else ""
        return f"# {p['title']}\n\n{p['text']}{footer}", {"title": p["title"], "found": True, "start": start}
