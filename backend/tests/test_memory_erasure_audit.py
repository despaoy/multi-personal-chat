"""Archive retention is distinct from permission-filtered history visibility."""

from evaluation.memory_erasure_audit import audit_erasure_archive


def test_revoked_quote_can_remain_in_archive_without_history_access():
    result = audit_erasure_archive({'archive_quote_row_count': 1},
                                   [{'role': 'user', 'content': '本人回执PB-681-Q已作废。'}], 'FJ-394-L')
    assert all(result.values())


def test_missing_archive_evidence_does_not_prove_archive_retention():
    assert not audit_erasure_archive({}, [], 'FJ-394-L')['chat_archive_quote_physically_preserved']


def test_physical_archive_deletion_is_not_a_successful_retention_check():
    assert not audit_erasure_archive({'archive_quote_row_count': 0}, [], 'FJ-394-L')['chat_archive_quote_physically_preserved']


def test_erased_quote_reappearing_in_regular_user_history_is_rejected():
    result = audit_erasure_archive({'archive_quote_row_count': 1},
                                   [{'role': 'user', 'content': '朋友回执FJ-394-L。'}], 'FJ-394-L')
    assert not result['revoked_receipt_filtered_from_regular_history']


def test_erased_detail_in_assistant_history_is_also_rejected():
    result = audit_erasure_archive({'archive_quote_row_count': 1},
                                   [{'role': 'assistant', 'content': '朋友的回执是FJ-394-L。'}], 'FJ-394-L')
    assert not result['revoked_receipt_filtered_from_regular_history']
