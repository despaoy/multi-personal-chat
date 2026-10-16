"""Unverified character search summaries; search failure never means no matches."""

import asyncio

import httpx
from bs4 import BeautifulSoup


class CharacterResearchError(RuntimeError):
    """Safe search failure details, without the query or upstream response body."""

    def __init__(self, source, reason, *, status_code=None, timed_out=False):
        self.source = source
        self.reason = reason
        self.status_code = status_code
        self.timed_out = timed_out
        status = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"Character research failed: {source}/{reason}{status}")


async def _get(client, source, url, **kwargs):
    try:
        response = await client.get(url, **kwargs)
    except httpx.TimeoutException as exc:
        raise CharacterResearchError(source, "timeout", timed_out=True) from exc
    except httpx.RequestError as exc:
        raise CharacterResearchError(source, "request_failed") from exc
    if response.status_code != 200:
        raise CharacterResearchError(source, "http_status", status_code=response.status_code)
    return response


async def _duckduckgo(client, description, max_results):
    response = await _get(
        client,
        "duckduckgo",
        "https://html.duckduckgo.com/html/",
        params={"q": f"{description} 角色 人物介绍"},
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )
    soup = BeautifulSoup(response.text, "html.parser")
    no_results = ".no-results, .no-results__message"
    rows = [row for row in soup.select(".result") if row.select_one(no_results) is None]
    if not rows:
        if soup.select_one(no_results) is None:
            raise CharacterResearchError("duckduckgo", "invalid_response")
        return ""

    lines = []
    for row in rows[:max_results]:
        title = row.select_one(".result__title")
        snippet = row.select_one(".result__snippet")
        if title is None or snippet is None:
            raise CharacterResearchError("duckduckgo", "invalid_result")
        title_text = title.get_text(" ", strip=True)
        snippet_text = snippet.get_text(" ", strip=True)
        if not title_text or not snippet_text:
            raise CharacterResearchError("duckduckgo", "invalid_result")
        lines.append(f"- {title_text}\n  {snippet_text}")
    return "DuckDuckGo 搜索摘要（未经独立核验）：\n" + "\n".join(lines)


async def _wikipedia(client, description, max_results):
    response = await _get(
        client,
        "wikipedia",
        "https://zh.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": description,
            "format": "json",
            "srlimit": max_results,
        },
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise CharacterResearchError("wikipedia", "invalid_json") from exc
    if not isinstance(payload, dict) or "error" in payload:
        raise CharacterResearchError("wikipedia", "invalid_response")
    query = payload.get("query")
    if not isinstance(query, dict) or not isinstance(query.get("search"), list):
        raise CharacterResearchError("wikipedia", "invalid_response")
    pages = query["search"]
    if not pages:
        return ""

    lines = []
    for page in pages[:max_results]:
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("title"), str)
            or not isinstance(page.get("snippet"), str)
        ):
            raise CharacterResearchError("wikipedia", "invalid_result")
        title = page["title"].strip()
        snippet = BeautifulSoup(page["snippet"], "html.parser").get_text(" ", strip=True)
        if not title or not snippet:
            raise CharacterResearchError("wikipedia", "invalid_result")
        lines.append(f"- {title}\n  {snippet}")
    return "Wikipedia 搜索摘要（未经独立核验）：\n" + "\n".join(lines)


async def search_character_info(description: str, max_results: int = 3) -> str:
    """Search both sources within one deadline; only valid empty results return empty."""
    async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
        try:
            async with asyncio.timeout(6):
                results = await asyncio.gather(
                    _duckduckgo(client, description, max_results),
                    _wikipedia(client, description, max_results),
                    return_exceptions=True,
                )
        except TimeoutError as exc:
            raise CharacterResearchError("search", "timeout", timed_out=True) from exc
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return "\n\n".join(result for result in results if result)
