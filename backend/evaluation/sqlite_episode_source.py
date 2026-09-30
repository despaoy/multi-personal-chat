"""Read-only SQLite source for isolated episodic recall experiments.

Not registered with the runtime and not a PostgreSQL adapter. The supplied
scope must already be authenticated by its caller; this is not an API.
"""
from dataclasses import dataclass

from character.models import UserScope
from db.database import SQLiteDB
from evaluation.episodic_recall_audit import Episode


@dataclass(frozen=True)
class EpisodeSource:
    episodes: tuple[Episode, ...]
    scope: tuple[str, ...]
    older_rows_omitted: bool


def read_scoped_episodes(db: SQLiteDB, scope: UserScope, character_id: str, *, max_rows: int = 200) -> EpisodeSource:
    if not isinstance(db, SQLiteDB):
        raise TypeError('Only isolated SQLite experiments are supported')
    page = db.list_scoped_conversation_turns(scope, character_id, limit=max_rows)
    episodes = []
    for turn in page.turns:
        texts = {utterance.role: utterance.text for utterance in turn.utterances}
        episodes.append(Episode(str(turn.source_id).zfill(20), page.scope, turn.session_id,
            turn.timestamp, texts.get('user', ''), texts.get('assistant', '')))
    return EpisodeSource(tuple(episodes), page.scope, page.older_rows_omitted)
