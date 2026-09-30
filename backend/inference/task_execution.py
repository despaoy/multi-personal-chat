"""Execute validated independent memory slots before one identity generation.

Unknown or dependent tasks are not split. No extra model/reviewer call, no
generated memory values, no source lookup, no mutation of stored conversation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from inference.memory_response import render_absent_memory_response, render_memory_response
from knowledge.task_plan import plan_turn_tasks


def prepare_independent_tasks(request):
    if not request.independent_tasks_enabled:
        return None
    binding = request.retrieval.identity_subtask
    if (not isinstance(binding, Mapping) or binding.get('query') != request.message
            or not isinstance(binding.get('subject'), str) or not binding['subject'].strip()
            or not request.retrieval.has_evidence or request.retrieval.source_lookup
            or getattr(request.character_context, 'branch_context', '')
            or any(getattr(request.reply_guard, key, False) for key in
                   ('require_gentle_safety_check', 'require_urgent_safety_check', 'third_party_safety'))
            or request.reply_guard_mode != 'lightweight'):
        return None
    tasks = plan_turn_tasks(request.message)
    content = [task for task in tasks if task.kind == 'content']
    memory = [(index, task) for index, task in enumerate(tasks) if task.kind == 'memory']
    if len(content) != 1 or not memory:
        return None
    completed = {}
    for index, task in memory:
        reply = render_memory_response(task.query, request.character_context, history=request.history)
        mode = 'memory_lookup'
        # A saved-memory absence does not negate facts in visible history.
        # Keep such requests intact until their additional dependencies can be
        # resolved; never let absence override an unpersisted user assertion.
        if reply is None and not request.history:
            reply = render_absent_memory_response(task.query, request.character_context)
            mode = 'memory_not_found'
        if reply is None:
            return None
        completed[index] = (reply, mode)
    # Preserve all current response controls and the original content clause.
    # Only tasks already executed from admitted memory are omitted here.
    residual = '，'.join(task.original.strip() for task in tasks if task.kind != 'memory')
    retrieval = replace(request.retrieval, identity_subtask={},
                        identity_task={'query': residual, 'subject': binding['subject']})
    return replace(request, message=residual, retrieval=retrieval), tasks, completed
