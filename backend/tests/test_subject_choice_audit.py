import json

import pytest

from evaluation.subject_choice_audit import CHOICES, choice_tokens, parse_choice, request_body


def test_all_known_role_combinations_and_separate_unknown_exist():
    assert len(CHOICES) == len(set(CHOICES.values())) == 16
    assert all('unknown' not in roles or len(roles) == 1 for roles in CHOICES.values())
    assert ('user', 'character') in CHOICES.values()
    assert ('shared',) in CHOICES.values()


def test_restriction_only_changes_token_mask_and_never_exposes_gold():
    case = dict(kind='query', text='我妹妹住哪？', expected_subjects=['secret'])
    tokens = {label: ord(label) for label in CHOICES}
    masked = request_body(case, 'model', tokens)
    plain = request_body(case, 'model', tokens, restricted=False)
    assert masked.pop('allowed_token_ids') == list(tokens.values())
    assert masked == plain
    assert 'secret' not in json.dumps(masked)


def test_exact_single_token_label_accepts_intentional_length_finish():
    assert parse_choice({'choices': [{'finish_reason': 'length', 'message': {'content': 'C'}}]}) == ('character',)


def test_input_view_ablation_keeps_system_sampling_and_text_exact():
    case = dict(kind='query', text='你忘了我原来的安排吗？')
    metadata = request_body(case, 'm', {'A': 1})
    plain = request_body(case, 'm', {'A': 1}, input_view='text')
    old_messages, new_messages = metadata.pop('messages'), plain.pop('messages')
    assert metadata == plain
    assert old_messages[0] == new_messages[0]
    assert json.loads(old_messages[1]['content'])['text'] == new_messages[1]['content'] == case['text']


@pytest.mark.parametrize('label', [' C', 'C\n', 'CC', '角色', '', 'Z'])
def test_invalid_label_never_silently_maps_to_user(label):
    with pytest.raises(ValueError):
        parse_choice({'choices': [{'finish_reason': 'length', 'message': {'content': label}}]})


def test_tokenizer_contract_is_validated():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(text)]
        def decode(self, tokens):
            return chr(tokens[0])
    assert choice_tokens(Tokenizer())['A'] == 65

    class SplitTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return [1, 2]
    with pytest.raises(ValueError):
        choice_tokens(SplitTokenizer())
