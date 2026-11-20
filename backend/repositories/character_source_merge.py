"""Conservative document-level three-way merge; never infer source identity from text."""
import json


def blocked(reason):
    return {'status': 'blocked', 'reason': reason, 'actions': [], 'conflicts': [], 'merged_sources': []}


def plan_source_merge(baseline, local, incoming, choices=None):
    from repositories.character_sources import OriginalSource

    def validate(sources):
        if not isinstance(sources, list) or len(sources) > 100:
            raise ValueError('Invalid source collection')
        values = [OriginalSource.model_validate(s).model_dump() for s in sources]
        if len({s['id'] for s in values}) != len(values):
            raise ValueError('Duplicate source identity')
        return {s['id']: s for s in values}

    try:
        base, ours, theirs = (validate(s) for s in (baseline, local, incoming))
    except ValueError:
        return blocked('invalid_source_identity_or_content')
    merged, actions, conflicts = [], [], []
    # Keep local ordering; append genuinely new author documents in author order.
    order = list(dict.fromkeys([*ours, *theirs, *base]))
    for identity in order:
        b, left, right = base.get(identity), ours.get(identity), theirs.get(identity)
        chosen = None
        if b is None:
            if left is not None and right is not None:
                # Equal text does not establish shared provenance for independent additions.
                conflicts.append({'id': identity, 'kind': 'addition_id_collision'})
                continue
            chosen = left if left is not None else right
            action = 'keep_local_addition' if left is not None else 'take_author_addition'
        elif left == b:
            chosen = right
            action = 'unchanged' if right == b else 'take_author_deletion' if right is None else 'take_author_modification'
        elif right == b:
            chosen = left
            action = 'keep_local_deletion' if left is None else 'keep_local_modification'
        elif left == right:
            chosen = left
            action = 'both_deleted' if left is None else 'same_modification'
        else:
            kind = 'local_delete_author_modify' if left is None else 'local_modify_author_delete' if right is None else 'both_modified'
            conflicts.append({'id': identity, 'kind': kind})
            continue
        actions.append({'id': identity, 'action': action})
        if chosen is not None:
            merged.append(chosen)
    resolved = []
    if choices is not None:
        if not isinstance(choices, dict) or len(choices) > 100 or any(value not in ('local', 'author') for value in choices.values()):
            raise ValueError('每篇冲突必须明确选择 local 或 author')
        if any(c['kind'] == 'addition_id_collision' for c in conflicts):
            return blocked('source_identity_collision_requires_explicit_handling')
        if set(choices) != {c['id'] for c in conflicts}:
            raise ValueError('必须逐篇选择全部冲突；不能提交缺漏、未知或非冲突资料的选择')
        selected = {s['id']: s for s in merged}
        for conflict in conflicts:
            identity = conflict['id']
            side = choices[identity]
            value = (ours if side == 'local' else theirs).get(identity)
            if value is not None:
                selected[identity] = value
            actions.append({'id': identity, 'action': f'choose_{side}_' + ('deletion' if value is None else 'version')})
            resolved.append({**conflict, 'choice': side})
        merged = [selected[identity] for identity in order if identity in selected]
        by_id = {a['id']: a for a in actions}
        actions = [by_id[identity] for identity in order]
        conflicts = []
    if conflicts:
        return {'status': 'conflict', 'reason': 'explicit_resolution_required', 'actions': actions,
                'conflicts': conflicts, 'merged_sources': []}
    payload = json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if len(merged) > 100 or len(payload.encode('utf-8')) > 5_000_000:
        result = blocked('merged_source_capacity_exceeded')
        result['actions'] = actions
        return result
    result = {'status': 'ready', 'reason': None, 'actions': actions, 'conflicts': [], 'merged_sources': merged}
    if choices is not None:
        result['resolved_conflicts'] = resolved
    return result
