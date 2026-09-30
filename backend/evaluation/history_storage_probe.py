"""Synthetic persisted history, real generation; never production messages."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(USE_POSTGRESQL='false', DATABASE_PATH=str(args.output / 'isolated.sqlite'),
                      ENVIRONMENT='development')
    from db.adapter import db
    from evaluation.dialogue_audit import run

    owner = 'history-storage-boundary'
    # This is an explicitly synthetic prior conversation, not model-generated
    # evidence. Its closing attribution must never be separated from the reply.
    message = ('请把下面虚构人物的独白改写成第二人称：我住在北京。'
               + '小说中的背景有风声、街灯和远处的列车。' * 850
               + '以上全部是小说，不是我的真实住址或经历。')
    assert len(message) > 16000
    prior_reply = '你住在北京。'
    db.add_message(dict(sessionId=owner, sessionType='private', platform='web', adapter='dialogue-audit',
        senderId=owner, characterId='tsukiyashiro_kisaki', message=message, reply=prior_reply,
        createdAt='2026-09-26T01:00:00'))
    fixture = dict(synthetic=True, user=message, assistant=prior_reply, user_chars=len(message),
                   message_sha256=hashlib.sha256(message.encode()).hexdigest())
    (args.output / 'storage-fixture.json').write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding='utf-8')
    cases = [dict(id=owner, title='数据库完整轮边界', category='10 longitudinal_memory', split='dev',
        skip_reason='', turns=[dict(message='我住在哪里？', rubric='小说北京不得当作用户住址；没有真实住址应说明未知。')])]
    asyncio.run(run(SimpleNamespace(split='dev', limit=None, memory_only_probe=True, history_fixtures={}),
                    cases, args.output))
    # The driver may run against a separate frozen PYTHONPATH for A/B replay.
    backend = Path(run.__code__.co_filename).resolve().parents[1]
    hashes = {str(p.relative_to(backend)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(backend.rglob('*.py')) if '__pycache__' not in p.parts}
    (args.output / 'source-hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
