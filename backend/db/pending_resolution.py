"""Validate internal candidate-resolution requests shared by database adapters."""


def pending_resolution_ids(metadata, relation, status):
    if not isinstance(metadata, dict) or 'resolved_pending_ids' not in metadata:
        return ()
    ids = metadata['resolved_pending_ids']
    if (metadata.get('origin') != 'rule_v2' or metadata.get('operation') not in {'append', 'replace'}
            or relation not in {'ADD', 'SUPERSEDE'} or status != 'active'
            or not isinstance(ids, list) or not ids
            or any(type(value) is not int or value <= 0 for value in ids)
            or len(ids) != len(set(ids))):
        raise ValueError('invalid pending memory resolution')
    return tuple(ids)
