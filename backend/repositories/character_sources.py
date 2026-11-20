"""Immutable primary source editions and verification of untrusted search candidates."""

import hashlib
import json
import uuid

from pydantic import BaseModel, ConfigDict, Field

from db.character_source_schema import INSERT_SOURCE_REVOCATION
from db.errors import RevisionConflictError
from repositories.character_chat import timestamp
from repositories.character_library import CharacterLibraryRepository, CharacterNotFoundError


class SourceIndexNotReadyError(ValueError):
    """An otherwise authorized source version has no completed index."""


class OriginalSource(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=200000)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class CharacterSourceRepository:
    def __init__(self, database):
        self.database = database
        self.characters = CharacterLibraryRepository(database)

    def list(self, owner, character_id, character_revision, limit=50, offset=0):
        self.characters.get(owner, character_id, character_revision)
        return self.database.execute_sql(
            'SELECT id,character_id,character_revision,content_digest,created_at,index_status,index_attempt FROM character_source_corpora '
            'WHERE owner_user_id=:owner AND character_id=:character AND character_revision=:revision AND revoked=0 '
            'ORDER BY edition_number DESC,created_at DESC,id DESC LIMIT :limit OFFSET :offset',
            {'owner': owner, 'character': character_id, 'revision': character_revision, 'limit': limit, 'offset': offset},
        )

    def has_edition(self, owner, character_id, character_revision, corpus_id, *, revoked=False):
        return bool(self.database.execute_sql(
            'SELECT 1 FROM character_source_corpora WHERE id=:id AND owner_user_id=:owner '
            'AND character_id=:character AND character_revision=:revision AND revoked=:revoked',
            {'id': corpus_id, 'owner': owner, 'character': character_id, 'revision': character_revision,
             'revoked': int(revoked)},
        ))

    def publish(self, owner, character_id, character_revision, sources):
        self.characters.get(owner, character_id, character_revision)
        params = self._edition_params(owner, character_id, character_revision, sources)
        self.database.revision_transaction([
            ('UPDATE character_library SET revision=revision WHERE id=:character AND owner_user_id=:owner AND revision=:revision', params, 1),
            self._insert_edition(params),
        ])
        return self.get(owner, params['id'])

    def install_preset(self, owner, key, settings, sources):
        character_id = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(["character-library-preset", key, owner])).hex
        params = self._edition_params(owner, character_id, 1, sources)
        try:
            self.database.revision_transaction([
                *self.characters.creation_statements(owner, settings, character_id),
                self._insert_edition(params),
            ])
        except RevisionConflictError as exc:
            try:
                return self.characters.get(owner, character_id)
            except CharacterNotFoundError:
                raise exc from None
        return self.characters.get(owner, character_id)

    @staticmethod
    def _edition_params(owner, character_id, character_revision, sources):
        if not 1 <= len(sources) <= 100:
            raise ValueError('A corpus requires 1 to 100 sources')
        validated = [OriginalSource.model_validate(source).model_dump() for source in sources]
        if len({s['id'] for s in validated}) != len(validated):
            raise ValueError('Duplicate original source identity')
        payload = json.dumps(validated, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if len(payload.encode('utf-8')) > 5_000_000:
            raise ValueError('Source edition exceeds 5 MB')
        return {'id': uuid.uuid4().hex, 'owner': owner, 'character': character_id, 'revision': character_revision,
                'digest': digest(payload), 'payload': payload, 'now': timestamp()}

    @staticmethod
    def _insert_edition(params):
        return ('INSERT INTO character_source_corpora (id,owner_user_id,character_id,character_revision,content_digest,'
             'sources_json,revoked,created_at,edition_number) '
             'SELECT :id,:owner,:character,:revision,:digest,:payload,0,:now,COALESCE(MAX(edition_number),0)+1 '
             'FROM character_source_corpora WHERE owner_user_id=:owner AND character_id=:character '
             'AND character_revision=:revision', params, 1)

    def get(self, owner, corpus_id):
        rows = self.database.execute_sql('SELECT * FROM character_source_corpora WHERE id=:id AND owner_user_id=:owner AND revoked=0',
                                         {'id': corpus_id, 'owner': owner})
        if not rows:
            raise CharacterNotFoundError('资料版本不存在或已撤销')
        result = dict(rows[0])
        result.pop('edition_number')
        payload = result.pop('sources_json')
        if digest(payload) != result['content_digest']:
            raise ValueError('Original corpus integrity check failed')
        result['sources'] = json.loads(payload)
        return result

    def revoke(self, owner, corpus_id):
        self.get(owner, corpus_id)
        self.database.revision_transaction([
            ('UPDATE character_source_corpora SET revoked=1 WHERE id=:id AND owner_user_id=:owner AND revoked=0',
             {'id': corpus_id, 'owner': owner}, 1),
            (INSERT_SOURCE_REVOCATION, {'id': corpus_id}, None),
        ])

    def pending_index_jobs(self, limit=100):
        return self.database.execute_sql(
            "SELECT id AS corpus_id,owner_user_id AS owner,index_attempt AS attempt FROM character_source_corpora "
            "WHERE revoked=0 AND index_status='pending' ORDER BY created_at,id LIMIT :limit", {'limit': limit},
        )

    def set_index_job_status(self, owner, corpus_id, attempt, status):
        if status not in ('submitted', 'failed'):
            raise ValueError('Invalid index job transition')
        self.get(owner, corpus_id)
        allowed = "('pending')" if status == 'submitted' else "('pending','submitted')"
        self.database.revision_transaction([
            (f'UPDATE character_source_corpora SET index_status=:status WHERE id=:id AND owner_user_id=:owner '
             f'AND revoked=0 AND index_attempt=:attempt AND index_status IN {allowed}',
             {'id': corpus_id, 'owner': owner, 'attempt': attempt, 'status': status}, 1),
        ])

    def retry_index_job(self, owner, corpus_id, attempt):
        self.get(owner, corpus_id)
        self.database.revision_transaction([
            ("UPDATE character_source_corpora SET index_status='pending',index_attempt=index_attempt+1,index_receipt_json=NULL "
             "WHERE id=:id AND owner_user_id=:owner AND revoked=0 AND index_attempt=:attempt AND index_status='failed'",
             {'id': corpus_id, 'owner': owner, 'attempt': attempt}, 1),
        ])
        return self.get(owner, corpus_id)

    def mark_indexed(self, owner, corpus_id, receipt, *, expected_attempt=None):
        corpus = self.get(owner, corpus_id)
        if receipt.get('corpus_digest') != corpus['content_digest'] or receipt.get('chunk_count', 0) < 1:
            raise ValueError('Invalid index publication receipt')
        self.database.revision_transaction([
            ("UPDATE character_source_corpora SET index_receipt_json=:receipt,index_status='completed' "
             "WHERE id=:id AND owner_user_id=:owner AND revoked=0 AND content_digest=:digest "
             "AND index_attempt=:attempt AND index_status IN ('pending','submitted','completed')",
             {'id': corpus_id, 'owner': owner, 'digest': corpus['content_digest'],
              'attempt': corpus['index_attempt'] if expected_attempt is None else expected_attempt,
              'receipt': json.dumps(receipt, sort_keys=True)}, 1),
        ])

    def require_indexed(self, owner, corpus_id, index_name, embedding_identity):
        corpus = self.get(owner, corpus_id)
        if corpus['index_status'] != 'completed':
            raise SourceIndexNotReadyError('资料索引尚未准备完成')
        receipt = json.loads(corpus.get('index_receipt_json') or '{}')
        if (receipt.get('index_name') != index_name or receipt.get('embedding_identity') != embedding_identity
                or receipt.get('corpus_digest') != corpus['content_digest'] or receipt.get('chunk_count', 0) < 1):
            raise ValueError('资料索引尚未准备完成')
        return corpus

    def verify_candidate(self, owner, character_id, character_revision, corpus_id, candidate):
        return self.verify_candidates(owner, character_id, character_revision, corpus_id, [candidate])[0]

    def verify_candidates(self, owner, character_id, character_revision, corpus_id, candidates):
        corpus = self.get(owner, corpus_id)
        if corpus['character_id'] != character_id or corpus['character_revision'] != character_revision:
            raise CharacterNotFoundError('资料版本不属于当前角色版本')
        required = {'owner_id': owner, 'character_id': character_id, 'character_revision': character_revision, 'corpus_revision': corpus_id}
        sources = {item['id']: item for item in corpus['sources']}
        return [self._verify_candidate(required, sources, candidate) for candidate in candidates]

    @staticmethod
    def _verify_candidate(required, sources, candidate):
        if any(candidate.get(key) != value for key, value in required.items()):
            raise ValueError('Source candidate scope changed')
        source = sources.get(candidate.get('source_id'))
        if source is None:
            raise CharacterNotFoundError('原文不存在')
        lines = source['content'].splitlines()
        start, end = candidate.get('line_start'), candidate.get('line_end')
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ValueError('Source candidate line range changed')
        content = '\n'.join(lines[start - 1:end])
        if digest(content) != candidate.get('content_sha256'):
            raise ValueError('Source candidate content changed')
        corpus_id = required['corpus_revision']
        identity = f'{corpus_id}:{source["id"]}:{start}:{end}'
        if candidate.get('chunk_id') != identity:
            raise ValueError('Source candidate identity changed')
        return {'id': identity, 'title': source['title'], 'content': content, 'line_start': start, 'line_end': end,
                'source_id': source['id'], 'corpus_revision': corpus_id, 'content_sha256': digest(content)}

    def chunks(self, owner, corpus_id, *, max_chars=1500):
        if type(max_chars) is not int or not 1 <= max_chars <= 12000:
            raise ValueError('Invalid source chunk size')
        corpus = self.get(owner, corpus_id)
        chunks = []
        for source in corpus['sources']:
            lines = source['content'].splitlines()
            start, selected, size = 1, [], 0
            for position, line in enumerate(lines, 1):
                if len(line) > max_chars:
                    raise ValueError('Source line exceeds chunk limit; split the source into shorter lines')
                if selected and size + len(line) + 1 > max_chars:
                    chunks.append(self._chunk(corpus, source, start, position - 1, selected))
                    start, selected, size = position, [], 0
                size += len(line) + (1 if selected else 0)
                selected.append(line)
            if selected:
                chunks.append(self._chunk(corpus, source, start, len(lines), selected))
        return chunks

    @staticmethod
    def _chunk(corpus, source, start, end, lines):
        content = '\n'.join(lines)
        return {'owner_id': corpus['owner_user_id'], 'character_id': corpus['character_id'],
                'character_revision': corpus['character_revision'], 'corpus_revision': corpus['id'],
                'chunk_id': f'{corpus["id"]}:{source["id"]}:{start}:{end}', 'source_id': source['id'],
                'title': source['title'], 'content': content, 'content_sha256': digest(content),
                'line_start': start, 'line_end': end}
