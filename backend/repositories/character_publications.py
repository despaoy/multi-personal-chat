"""Explicit, immutable sharing; only allowlisted author content crosses accounts."""
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timezone

from character.library import CharacterSettings
from db.errors import RevisionConflictError
from repositories.character_library import CharacterNotFoundError, SourceMergeBlockedError
from repositories.character_source_merge import blocked, plan_source_merge


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def now():
    return datetime.now(timezone.utc).isoformat()


class CharacterPublicationRepository:
    def __init__(self, characters):
        self.characters = characters
        self.db = characters.database

    def preview(self, owner, character_id, revision, corpus_id=None, source_ids=None):
        from repositories.character_sources import CharacterSourceRepository

        character = self.characters.get(owner, character_id)
        if character['revision'] != revision:
            raise RevisionConflictError()
        source_ids = source_ids or []
        if len(source_ids) != len(set(source_ids)) or len(source_ids) > 100:
            raise ValueError('资料选择重复或超过100篇')
        if bool(corpus_id) != bool(source_ids):
            raise ValueError('选择资料版本后须明确勾选原文；不携带资料时清空选择')
        sources = []
        if corpus_id:
            corpus = CharacterSourceRepository(self.db).get(owner, corpus_id)
            if corpus['character_id'] != character_id or corpus['character_revision'] != revision:
                raise CharacterNotFoundError('资料不属于当前角色版本')
            sources = [s for s in corpus['sources'] if s['id'] in source_ids]
            if len(sources) != len(source_ids):
                raise CharacterNotFoundError('所选原文不存在')
        payload = {'character_id': character_id, 'revision': revision, 'corpus_id': corpus_id,
                   'settings': character['settings'], 'sources': sources}
        return {**payload, 'preview_id': hashlib.sha256(encode(payload).encode()).hexdigest(),
                'previous_publication_id': character['publication_id']}

    def publish(self, owner, character_id, revision, enabled, corpus_id=None, source_ids=None,
                preview_id=None, previous_publication_id=None):
        current = self.characters.get(owner, character_id)
        if not enabled:
            self.db.revision_transaction([(
                'UPDATE character_library SET share_token=NULL,shared_revision=NULL,publication_id=NULL '
                'WHERE id=:id AND owner_user_id=:owner AND revision=:revision',
                {'id': character_id, 'owner': owner, 'revision': revision}, 1)])
            return {'share_token': None, 'shared_revision': None, 'publication_id': None}
        preview = self.preview(owner, character_id, revision, corpus_id, source_ids)
        if preview_id is not None and preview['preview_id'] != preview_id:
            raise RevisionConflictError()
        # Exact content identity makes a retried publication idempotent.
        publication_id = preview['preview_id']
        expected = previous_publication_id if preview_id is not None else current['publication_id']
        token = current['share_token'] or secrets.token_urlsafe(32)
        params = {'id': character_id, 'owner': owner, 'revision': revision, 'token': token,
                  'publication': publication_id, 'previous': expected, 'corpus': corpus_id,
                  'settings': encode(preview['settings']), 'sources': encode(preview['sources']), 'now': now()}
        statements = [(
            'UPDATE character_library SET share_token=COALESCE(share_token,:token),shared_revision=:revision,publication_id=:publication '
            'WHERE id=:id AND owner_user_id=:owner AND revision=:revision '
            'AND (publication_id=:previous OR (publication_id IS NULL AND :previous IS NULL) OR publication_id=:publication)', params, 1)]
        if corpus_id:
            statements.append(self._corpus_guard(corpus_id, character_id, owner))
        statements.append((
            'INSERT INTO character_publications (id,character_id,character_revision,corpus_id,settings_json,sources_json,created_at) '
            'SELECT :publication,:id,:revision,:corpus,:settings,:sources,:now '
            'WHERE NOT EXISTS (SELECT 1 FROM character_publications WHERE id=:publication)', params, None))
        self.db.revision_transaction(statements)
        published = self.characters.get(owner, character_id)
        if published['publication_id'] != publication_id or not published['share_token']:
            raise RevisionConflictError()
        return {'share_token': published['share_token'], 'shared_revision': revision, 'publication_id': publication_id}

    @staticmethod
    def _corpus_guard(corpus_id, character_id, owner):
        return ('UPDATE character_source_corpora SET revoked=revoked WHERE id=:corpus '
                'AND character_id=:character AND owner_user_id=:owner AND revoked=0',
                {'corpus': corpus_id, 'character': character_id, 'owner': owner}, 1)

    def _publication(self, *, token=None, character_id=None):
        rows = self.db.execute_sql(
            'SELECT p.*,c.owner_user_id,c.share_token FROM character_library c '
            'JOIN character_publications p ON p.id=c.publication_id '
            'WHERE c.share_token IS NOT NULL AND ' + ('c.share_token=:key' if token else 'c.id=:key'),
            {'key': token or character_id})
        if not rows:
            raise CharacterNotFoundError('分享不存在或已关闭')
        value = rows[0]
        if value['corpus_id']:
            active = self.db.execute_sql(
                'SELECT id FROM character_source_corpora WHERE id=:id AND owner_user_id=:owner '
                'AND character_id=:character AND revoked=0',
                {'id': value['corpus_id'], 'owner': value['owner_user_id'], 'character': value['character_id']})
            if not active:
                raise CharacterNotFoundError('分享包含已撤销资料，请等待作者重新发布')
        return value

    @staticmethod
    def _public(value):
        from repositories.character_sources import OriginalSource

        return {'id': value['character_id'], 'settings_revision': value['character_revision'],
                'publication_id': value['id'], 'settings': CharacterSettings.model_validate_json(value['settings_json']).model_dump(),
                'sources': [OriginalSource.model_validate(s).model_dump() for s in json.loads(value['sources_json'])]}

    def shared(self, token):
        return self._public(self._publication(token=token))

    def _guards(self, publication, token):
        statements = [(
            'UPDATE character_library SET revision=revision WHERE id=:id '
            'AND share_token=:token AND publication_id=:publication AND shared_revision=:revision',
            {'id': publication['character_id'], 'token': token, 'publication': publication['id'],
             'revision': publication['character_revision']}, 1)]
        if publication['corpus_id']:
            statements.append(self._corpus_guard(publication['corpus_id'], publication['character_id'], publication['owner_user_id']))
        return statements

    @staticmethod
    def _copy_sources(owner, character_id, revision, sources):
        if not sources:
            return []
        payload = encode(sources)
        return [(
            'INSERT INTO character_source_corpora (id,owner_user_id,character_id,character_revision,content_digest,'
            'sources_json,revoked,created_at,index_status,index_attempt,edition_number) '
            "VALUES (:id,:owner,:character,:revision,:digest,:payload,0,:now,'pending',1,1)",
            {'id': uuid.uuid4().hex, 'owner': owner, 'character': character_id, 'revision': revision,
             'digest': hashlib.sha256(payload.encode()).hexdigest(), 'payload': payload, 'now': now()}, 1)]

    def _retry(self, owner, request_id, publication_id):
        rows = self.db.execute_sql('SELECT * FROM character_publication_copies WHERE owner_user_id=:owner AND request_id=:request',
                                   {'owner': owner, 'request': request_id})
        if not rows:
            return None
        if rows[0]['publication_id'] != publication_id:
            raise RevisionConflictError()
        return self.characters.get(owner, rows[0]['character_id'])

    def copy(self, owner, token, publication_id=None, request_id=None):
        if request_id and publication_id:
            retried = self._retry(owner, request_id, publication_id)
            if retried:
                return retried
        # Use the normal read entry point so tests can interleave a revocation.
        source = self.characters.shared(token)
        if publication_id is not None and source['publication_id'] != publication_id:
            raise RevisionConflictError()
        try:
            publication = self._publication(token=token)
        except CharacterNotFoundError as exc:
            raise RevisionConflictError() from exc
        if publication['id'] != source['publication_id']:
            raise RevisionConflictError()
        character_id = uuid.uuid4().hex
        params = {'id': character_id, 'owner': owner, 'source': source['id'], 'source_revision': source['settings_revision'],
                  'publication': source['publication_id'], 'revision': 1, 'settings': encode(source['settings']), 'now': now(),
                  'request': request_id or uuid.uuid4().hex}
        statements = self._guards(publication, token) + [(
            'INSERT INTO character_library (id,owner_user_id,revision,source_character_id,source_revision,source_publication_id,created_at,updated_at) '
            'VALUES (:id,:owner,1,:source,:source_revision,:publication,:now,:now)', params, 1),
            self.characters._insert_version(params)]
        statements += self._copy_sources(owner, character_id, 1, source['sources'])
        statements.append((
            'INSERT INTO character_publication_copies (owner_user_id,request_id,publication_id,character_id) '
            'VALUES (:owner,:request,:publication,:id)', params, 1))
        try:
            self.db.revision_transaction(statements)
        except Exception:
            # The transaction rolled back. A concurrent identical retry may have won.
            if request_id:
                retried = self._retry(owner, request_id, source['publication_id'])
                if retried:
                    return retried
            raise
        return self.characters.get(owner, character_id)

    def _local_sources(self, owner, character):
        from repositories.character_sources import CharacterSourceRepository

        repo = CharacterSourceRepository(self.db)
        editions = repo.list(owner, character['id'], character['revision'], 1)
        if not editions:
            return None, []
        corpus = repo.get(owner, editions[0]['id'])
        return corpus['id'], corpus['sources']

    def carry_local_sources(self, owner, current, next_revision):
        """Editing a greeting must not silently detach the character's original sources."""
        corpus_id, sources = self._local_sources(owner, current)
        params = {'id': current['id'], 'owner': owner, 'previous': current['revision'], 'corpus': corpus_id}
        statements = [self._local_guard(params)]
        if corpus_id:
            statements.append(self._corpus_guard(corpus_id, current['id'], owner))
        return statements + self._copy_sources(owner, current['id'], next_revision, sources)

    @staticmethod
    def _local_guard(params):
        return (
            'UPDATE character_library SET revision=revision WHERE id=:id AND owner_user_id=:owner '
            'AND ((CAST(:corpus AS TEXT) IS NULL AND NOT EXISTS (SELECT 1 FROM character_source_corpora '
            'WHERE owner_user_id=:owner AND character_id=:id AND character_revision=:previous AND revoked=0)) '
            'OR :corpus=(SELECT id FROM character_source_corpora WHERE owner_user_id=:owner AND character_id=:id '
            'AND character_revision=:previous AND revoked=0 '
            'ORDER BY edition_number DESC,created_at DESC,id DESC LIMIT 1))', params, 1,
        )

    def _source_baseline(self, current):
        """Only the imported author's still-authorized publication can be a baseline.

        Never substitute the current author version, the recipient's edited corpus,
        or a snapshot belonging to another character. Do not expose baseline text.
        """
        from repositories.character_sources import CharacterSourceRepository

        if not current['source_publication_id']:
            return None
        rows = self.db.execute_sql(
            'SELECT p.*,c.owner_user_id FROM character_publications p '
            'JOIN character_library c ON c.id=p.character_id '
            'WHERE p.id=:publication AND p.character_id=:source AND p.character_revision=:revision '
            'AND c.share_token IS NOT NULL',
            {'publication': current['source_publication_id'], 'source': current['source_character_id'],
             'revision': current['source_revision']})
        if not rows:
            return None
        value = rows[0]
        try:
            if value['corpus_id']:
                corpus = CharacterSourceRepository(self.db).get(value['owner_user_id'], value['corpus_id'])
                if corpus['character_id'] != value['character_id'] or corpus['character_revision'] != value['character_revision']:
                    return None
            source = self._public(value)
            # A settings-only publication cannot hide an unbound source snapshot.
            if not value['corpus_id'] and source['sources']:
                return None
            if value['corpus_id']:
                originals = {s['id']: s for s in corpus['sources']}
                if any(originals.get(s['id']) != s for s in source['sources']):
                    return None
            return value
        except (CharacterNotFoundError, ValueError, TypeError, KeyError):
            return None

    def _baseline_guards(self, baseline):
        statements = [(
            'UPDATE character_publications SET id=id WHERE id=:id AND character_id=:character '
            'AND character_revision=:revision AND sources_json=:sources',
            {'id': baseline['id'], 'character': baseline['character_id'],
             'revision': baseline['character_revision'], 'sources': baseline['sources_json']}, 1)]
        if baseline['corpus_id']:
            statements.append(self._corpus_guard(baseline['corpus_id'], baseline['character_id'], baseline['owner_user_id']))
        return statements

    def source_update(self, owner, character_id):
        current = self.characters.get(owner, character_id)
        empty = {'available': False, 'revision': current['revision'], 'source_revision': None,
                 'publication_id': None, 'changes': {}, 'sources': [], 'current_sources': [], 'local_corpus_id': None,
                 'incoming_sources': [], 'baseline_publication_id': None,
                 'source_merge': blocked('incoming_publication_unavailable')}
        if not current['source_character_id']:
            return empty
        try:
            publication = self._publication(character_id=current['source_character_id'])
        except CharacterNotFoundError:
            return empty
        source = self._public(publication)
        corpus_id, sources = self._local_sources(owner, current)
        # Legacy copies retain their settings-only source revision until their author republishes.
        available = (source['publication_id'] != current['source_publication_id'] if current['source_publication_id']
                     else source['settings_revision'] > (current['source_revision'] or 0) or not source['publication_id'].startswith('legacy-'))
        baseline = self._source_baseline(current)
        plan = (plan_source_merge(self._public(baseline)['sources'], sources, source['sources'])
                if baseline else blocked('baseline_unavailable'))
        return {**empty, 'available': available, 'source_revision': source['settings_revision'],
                'publication_id': source['publication_id'], 'local_corpus_id': corpus_id,
                'sources': plan['merged_sources'] if plan['status'] == 'ready' else source['sources'],
                'incoming_sources': source['sources'], 'current_sources': sources, 'source_merge': plan,
                'baseline_publication_id': baseline['id'] if baseline else None,
                'changes': {k: {'current': current['settings'][k], 'incoming': v}
                            for k, v in source['settings'].items() if v != current['settings'][k]}}

    def _prepare_update(self, owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices):
        current = self.characters.get(owner, character_id)
        preview = self.source_update(owner, character_id)
        if (current['revision'] != revision or not preview['available'] or preview['source_revision'] != source_revision
                or (publication_id is not None and (publication_id != preview['publication_id'] or local_corpus_id != preview['local_corpus_id']))):
            raise RevisionConflictError()
        publication = self._publication(character_id=current['source_character_id'])
        if publication['id'] != preview['publication_id']:
            raise RevisionConflictError()
        if preview['source_merge']['status'] == 'blocked':
            raise SourceMergeBlockedError('资料基线不可用或合并超限；需明确处理，不能自动覆盖')
        baseline = self._source_baseline(current)
        if not baseline:
            raise SourceMergeBlockedError('原导入资料基线已不可用；需明确处理，不能自动覆盖')
        source = self._public(publication)
        originals = self._public(baseline)['sources']
        unchosen = plan_source_merge(originals, preview['current_sources'], source['sources'])
        if baseline['id'] != preview['baseline_publication_id'] or unchosen != preview['source_merge']:
            raise RevisionConflictError()
        plan = plan_source_merge(originals, preview['current_sources'], source['sources'], choices)
        if plan['status'] != 'ready':
            raise SourceMergeBlockedError('资料存在未解决冲突、来源ID碰撞或合并超限；不能自动覆盖')
        return current, preview, publication, baseline, source, plan

    @staticmethod
    def _update_request(owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices):
        return {'owner': owner, 'character_id': character_id, 'revision': revision, 'source_revision': source_revision,
                'publication_id': publication_id, 'local_corpus_id': local_corpus_id, 'choices': choices}

    @staticmethod
    def _final_preview(request, prepared):
        current, preview, publication, baseline, source, plan = prepared
        # The digest is a content/version precondition, not permission to read a source.
        # Authority and all transaction guards still apply at confirmation time.
        binding = {'format': 'source-update-preview-v1', 'request': request,
                   'baseline_id': baseline['id'], 'baseline_sources': json.loads(baseline['sources_json']),
                   'current_settings': current['settings'], 'local_sources': preview['current_sources'],
                   'author_settings': source['settings'], 'author_sources': source['sources'],
                   'share_token': publication['share_token'], 'plan': plan}
        return {'revision': current['revision'], 'source_revision': source['settings_revision'],
                'publication_id': source['publication_id'], 'local_corpus_id': preview['local_corpus_id'],
                'baseline_publication_id': baseline['id'], 'choices': request['choices'],
                'preview_id': hashlib.sha256(encode(binding).encode('utf-8')).hexdigest(),
                'settings': source['settings'], 'sources': plan['merged_sources'], 'source_merge': plan}

    def preview_update(self, owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices):
        if choices is None:
            raise ValueError('最终预览必须提供逐篇选择；无冲突时提供空选择')
        prepared = self._prepare_update(owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices)
        request = self._update_request(owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices)
        return self._final_preview(request, prepared)

    def _completed_update(self, owner, character_id, preview_id, request):
        current = self.characters.get(owner, character_id)
        rows = self.db.execute_sql(
            'SELECT request_json,result_revision FROM character_source_update_receipts '
            'WHERE owner_user_id=:owner AND character_id=:character AND preview_id=:preview',
            {'owner': owner, 'character': character_id, 'preview': preview_id})
        if not rows:
            return None
        if rows[0]['request_json'] != encode(request) or rows[0]['result_revision'] != current['revision']:
            raise RevisionConflictError()
        return current

    def import_update(self, owner, character_id, revision, source_revision, publication_id=None, local_corpus_id=None,
                      choices=None, preview_id=None):
        current = self.characters.get(owner, character_id)
        if (choices is None) != (preview_id is None):
            raise ValueError('逐篇选择与最终预览编号必须一起提交')
        request = self._update_request(owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices)
        if preview_id is not None:
            completed = self._completed_update(owner, character_id, preview_id, request)
            if completed:
                return completed
        elif (publication_id and current['revision'] == revision + 1
                and current['source_publication_id'] == publication_id and current['source_revision'] == source_revision):
            # Compatibility for the existing conflict-free confirmation protocol.
            return current
        try:
            prepared = self._prepare_update(owner, character_id, revision, source_revision, publication_id, local_corpus_id, choices)
        except RevisionConflictError:
            # An identical confirmation may have committed after the initial receipt lookup.
            if preview_id is not None:
                completed = self._completed_update(owner, character_id, preview_id, request)
                if completed:
                    return completed
            raise
        current, preview, publication, baseline, source, plan = prepared
        if preview_id is not None and self._final_preview(request, prepared)['preview_id'] != preview_id:
            raise RevisionConflictError()
        params = {'id': character_id, 'owner': owner, 'previous': revision, 'revision': revision + 1,
                  'source_revision': source_revision, 'publication': publication['id'], 'settings': encode(source['settings']),
                  'now': now(), 'corpus': preview['local_corpus_id']}
        statements = self._guards(publication, publication['share_token']) + self._baseline_guards(baseline) + [(
            'UPDATE character_library SET revision=:revision,source_revision=:source_revision,source_publication_id=:publication,updated_at=:now '
            'WHERE id=:id AND owner_user_id=:owner AND revision=:previous', params, 1),
            self._local_guard(params)]
        if preview['local_corpus_id']:
            statements.append(self._corpus_guard(preview['local_corpus_id'], character_id, owner))
        statements += [self.characters._insert_version(params)] + self._copy_sources(owner, character_id, revision + 1, plan['merged_sources'])
        if preview_id is not None:
            statements.append((
                'INSERT INTO character_source_update_receipts (owner_user_id,character_id,preview_id,request_json,result_revision) '
                'VALUES (:owner,:id,:preview,:request,:revision)',
                {**params, 'preview': preview_id, 'request': encode(request)}, 1))
        try:
            self.db.revision_transaction(statements)
        except RevisionConflictError:
            if preview_id is not None:
                completed = self._completed_update(owner, character_id, preview_id, request)
                if completed:
                    return completed
            raise
        return self.characters.get(owner, character_id, revision + 1)
