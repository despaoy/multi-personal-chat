"""Measure complete source text on the real wire, including JSON quote transport."""

import json
from html import unescape


def complete_source_in_transport(text, source):
    """Require byte-identical decoded speech; never normalize or accept fragments.

    Temporal read views intentionally transmit source speech as JSON strings.
    Escaped line breaks/tabs preserve the speech but defeat a raw substring
    check. This presence measurement does not establish speaker or validity;
    native scope, source IDs/clocks and temporal authority are separate gates.
    """
    if not isinstance(text, str) or not isinstance(source, str) or not source.strip():
        raise ValueError("Expected nonempty complete source and text transport")
    transport = unescape(text)
    return source in transport or json.dumps(source, ensure_ascii=False) in transport
