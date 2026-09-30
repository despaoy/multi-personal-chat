"""Recorded independent writer proposals for controlled admission experiments."""
import json

from character.memory_llm import _extract_json


class FrozenMemoryWriter:
    def __init__(self, rows, current):
        self.rows = {(row['case'], row['index']): row for row in rows}
        self.current = current
        self.calls = []
        self.used = {}

    async def complete(self, messages):
        key = (self.current['case'], self.current['index'])
        index = self.used.get(key, 0)
        call = self.rows[key]['writer_calls'][index]
        actual = json.loads(messages[-1]['content'])
        recorded = json.loads(call['messages'][-1]['content'])
        if actual['current_user_message'] != recorded['current_user_message']:
            raise ValueError('Frozen writer source mismatch')
        candidates = _extract_json(call['output'])['memories']
        if any(c.get('operation', 'ADD') not in {'ADD', 'NOOP'}
               or c.get('target_memory_id') or c.get('target_memory_key') for c in candidates):
            raise ValueError('Do not replay target-dependent proposals against a changed database')
        self.used[key] = index + 1
        self.calls.append(dict(messages=messages, output=call['output'], recorded=True, seconds=0))
        return call['output']

    async def close(self):
        pass
