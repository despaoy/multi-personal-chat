import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from character.context_builder import build_user_scope
from db.database import SQLiteDB
from evaluation.episodic_chain_probe import attach_evidence, seed_cases, seed_saved_claims
from evaluation.sqlite_episode_source import read_scoped_episodes


@dataclass(frozen=True)
class Compiled:
    reference_context: str = '原有参考'
    memory_status: str = 'absent'
    episodic_reference_context: str = ''


@dataclass(frozen=True)
class Prepared:
    compiled: Compiled
    history: tuple = ()


def test_storage_bridge_preserves_history_and_never_promotes_raw_fact(tmp_path):
    db = SQLiteDB(tmp_path / 'isolated.sqlite')
    suite = json.loads((Path(__file__).parents[1] / 'evaluation/fixtures/episodic_evidence_20260926.json')
                       .read_text(encoding='utf-8'))
    cases, fixtures = seed_cases(db, suite)
    for case, fixture in zip(cases, fixtures):
        scope = build_user_scope(platform='web', adapter='dialogue-audit', sender_id=case['id'],
                                 conversation_type='private', conversation_id=case['id'])
        source = read_scoped_episodes(db, scope, 'tsukiyashiro_kisaki')
        assert len(source.episodes) == len(fixture['rows'])
        assert all('段落编号' in r.message for r in source.episodes[-24:])
        prior = Prepared(Compiled(), ({'role': 'assistant', 'content': '正常历史保留'},))
        enriched, details = attach_evidence(prior, source, case['turns'][0]['message'], max_chars=16000)
        assert enriched.history == prior.history
        assert enriched.compiled.memory_status == 'absent'
        assert prior.compiled.reference_context == '原有参考'
        assert len(details['source_ids']) == len(source.episodes)
        assert all('reply' not in row for row in json.loads(details['injected_packet']))
        assert all(r.message in enriched.compiled.reference_context for r in source.episodes)
        separate, _ = attach_evidence(prior, source, case['turns'][0]['message'],
                                      max_chars=16000, separate_channel=True)
        assert separate.compiled.reference_context == prior.compiled.reference_context
        assert separate.compiled.episodic_reference_context == details['injected_packet']
        assert separate.compiled.memory_status == 'absent'
        rejected, details = attach_evidence(prior, source, case['turns'][0]['message'], max_chars=1500)
        assert rejected is prior
        assert details['budget_rejected'] == ['complete_suffix']
        assert details['source_ids'] == []


def test_owner_transfer_has_identical_evidence_and_separate_review_rubrics(tmp_path):
    suite = json.loads((Path(__file__).parents[1] / 'evaluation/fixtures/episodic_owner_transfer_20260926.json')
                       .read_text(encoding='utf-8'))
    assert len(suite) == 6
    for scenario in {'revised', 'ambiguous'}:
        group = [case for case in suite if case['scenario'] == scenario]
        assert {case['query_owner'] for case in group} == {'user', 'character', 'implicit'}
        assert all(case['episodes'] == group[0]['episodes'] for case in group)
    db = SQLiteDB(tmp_path / 'owner.sqlite')
    cases, _ = seed_cases(db, suite)
    assert [case['turns'][0]['rubric'] for case in cases] == [case['rubric'] for case in suite]


@pytest.mark.asyncio
async def test_dependency_fixture_seeds_only_explicit_old_fact(tmp_path):
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    suite = json.loads((Path(__file__).parents[1] / 'evaluation/fixtures/episodic_dependency_20260926.json')
                       .read_text(encoding='utf-8'))
    db = SQLiteDB(tmp_path / 'dependency.sqlite')
    cases, _ = seed_cases(db, suite)
    await seed_saved_claims(db, suite)
    repo = DatabaseCharacterMemoryRepository(db)
    for case, specification in zip(cases, suite):
        scope = build_user_scope(platform='web', adapter='dialogue-audit', sender_id=case['id'],
                                 conversation_type='private', conversation_id=case['id'])
        records = await repo.list_memory_records('tsukiyashiro_kisaki', scope, limit=None)
        assert len(records) == len(specification['stored_episode_indices'])
        if records:
            assert '厦门' in records[0]['content'] and '海口' not in records[0]['content']
