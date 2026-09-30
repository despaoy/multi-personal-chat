"""Model-adapter experiment, not a default runtime or storage policy."""
from __future__ import annotations

from copy import deepcopy
from html import escape


def separate_source_packet(messages: list[dict], source: str) -> list[dict]:
    """Move the identical untrusted source block; preserve every history turn.

    Do not turn retrieved speech into synthetic conversation turns, infer its
    factual subject, or remove assistant history needed for references.
    """
    result = deepcopy(messages)
    if not source:
        return result
    if not result or result[-1].get('role') != 'user':
        raise ValueError('Expected a final user query')
    packet = ('<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
              + escape(source, quote=False) + '\n</dialogue_evidence>')
    if result[-1]['content'].count(packet) != 1:
        raise ValueError('The exact source packet must reach the model once')
    result[-1]['content'] = result[-1]['content'].replace(packet, '')
    result.insert(len(result)-1, {'role': 'user', 'content': packet})
    return result


def separate_reference_packet(messages: list[dict], source: str) -> list[dict]:
    """Keep memory and RAG together while giving the query its own message.

    Diagnostic alternative to privileging current-message RAG over a moved
    source packet. Exact escaped reference text and query markup stay intact.
    """
    if not source:
        return deepcopy(messages)
    # Reuse the source-presence assertion; never manufacture missing evidence.
    separate_source_packet(messages, source)
    prefix, marker, query = messages[-1]['content'].rpartition('\n\n<user_query>\n')
    if not prefix or not marker or not query.endswith('\n</user_query>'):
        raise ValueError('Expected exactly rendered reference data followed by the current query')
    return [*deepcopy(messages[:-1]), {'role': 'user', 'content': prefix},
            {'role': 'user', 'content': '<user_query>\n' + query}]
