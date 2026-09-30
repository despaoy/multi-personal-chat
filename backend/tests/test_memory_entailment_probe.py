import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.memory_entailment_probe import validate_cases


def test_independent_fixture_has_balanced_labels_and_unique_cases():
    path = Path(__file__).resolve().parents[1] / 'evaluation/fixtures/memory_entailment_transfer_20260927.json'
    cases = json.loads(path.read_text(encoding='utf-8'))
    validate_cases(cases)
    assert Counter(row['label'] for row in cases) == dict(entailment=8, neutral=8, contradiction=8)
    assert len({r['group'] for r in cases}) == 4


@pytest.mark.parametrize('change', ['duplicate', 'bad_label', 'empty_source'])
def test_invalid_experiments_rejected(change):
    case = dict(id='one', group='control', premise='甲在读书。', hypothesis='甲在读书。', label='entailment')
    cases = [case]
    if change == 'duplicate':
        cases.append(dict(case))
    elif change == 'bad_label':
        case['label'] = 'true'
    else:
        case['premise'] = ' '
    with pytest.raises(ValueError):
        validate_cases(cases)
