"""Render verified source packets without asking a model to invent a quote."""
from __future__ import annotations

import re

MAX_SOURCE_REPLY_CHARS = 4000


def render_source_response(retrieval):
    """A related passage is not certification of previous generated claims.

    Only dedicated extractor packets qualify. Never reconstruct a quote from
    summaries, document titles, chat history, or model-written evidence text.
    """
    if not retrieval.source_lookup:
        return None
    if retrieval.status == 'ok':
        for packet in retrieval.source_excerpts:
            text, path = packet.get('text'), packet.get('source_path')
            start, end = packet.get('line_start'), packet.get('line_end')
            if (not isinstance(text, str) or not text.strip() or len(text) > MAX_SOURCE_REPLY_CHARS
                    or not isinstance(path, str) or not path or re.search(r'[\r\n`]', path)
                    or type(start) is not int or type(end) is not int or start < 1 or end < start
                    or len(text.split('\n')) != end - start + 1
                    or packet.get('truncated') is not False):
                continue
            fence = '`' * max(3, max((len(m.group()) + 1 for m in re.finditer(r'`+', text)), default=3))
            reply = ('能核对的相关原文如下。它不代表前面每一项判断都已得到证明。\n\n'
                     f'出处：`{path}`，第 {start}–{end} 行。\n\n{fence}text\n{text}\n{fence}')
            citation = {'id': packet.get('parent_id') or packet.get('evidence_id') or '',
                        'source_path': path, 'line_start': start, 'line_end': end,
                        'card_id': packet.get('parent_id') or ''}
            return reply, 'source_excerpt', (citation,)
    return '目前没有可完整核对的原文片段，不能把前面的说法当作已经证实。', 'source_unavailable', ()
