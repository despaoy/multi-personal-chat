"""Incremental removal of explicit inline reasoning protocol markers.

Only a possible marker suffix is buffered (at most seven characters), never
the scratchpad. Native reasoning_content deltas remain outside this filter.
"""
from __future__ import annotations

_OPEN = '<think>'
_CLOSE = '</think>'


class ReasoningTextFilter:
    def __init__(self):
        self._pending = ''
        self._depth = 0

    def feed(self, content: str) -> str:
        self._pending += content
        output = []
        while self._pending:
            lowered = self._pending.lower()
            markers = (_OPEN, _CLOSE) if self._depth else (_OPEN,)
            hits = [(lowered.find(marker), marker) for marker in markers if marker in lowered]
            if hits:
                index, marker = min(hits)
                if not self._depth:
                    output.append(self._pending[:index])
                self._pending = self._pending[index + len(marker):]
                self._depth += 1 if marker == _OPEN else -1
                continue
            # Hold split delimiters only; emit final text or discard reasoning.
            keep = max((size for marker in markers for size in range(1, len(marker))
                        if lowered.endswith(marker[:size])), default=0)
            length = len(self._pending) - keep
            if not self._depth:
                output.append(self._pending[:length])
            self._pending = self._pending[length:]
            break
        return ''.join(output)

    def finish(self) -> str:
        # A partial opener alone is literal text; once an opener was completed,
        # an interrupted generation must never flush the hidden scratchpad.
        result = '' if self._depth else self._pending
        self._pending = ''
        self._depth = 0
        return result


def strip_reasoning_text(content: str) -> str:
    lower = content.lower()
    opener, closer = lower.find(_OPEN), lower.find(_CLOSE)
    has_protocol = opener >= 0 or closer >= 0
    if closer >= 0 and (opener < 0 or closer < opener):
        # Non-streaming compatibility with templates that supply the opener.
        content = content[closer + len(_CLOSE):]
    stream = ReasoningTextFilter()
    result = stream.feed(content) + stream.finish()
    return result.strip() if has_protocol else result
