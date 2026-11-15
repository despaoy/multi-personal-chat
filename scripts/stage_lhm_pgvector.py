"""Stage PostgreSQL and pgvector in a separate project-local tools tree."""

from probe_lhm_runtime import connect
from run_lhm_postgres_tests import remote_python

SOURCE = r'''
import pathlib, shutil
base = pathlib.Path('/home/boot/lhm/multipersonal-deploy/tools')
target = base/'postgres-vector-14'
original = pathlib.Path('/home/boot/lhm/multipersonal-runtime/tools/pgsql')
source = base/'pgvector-build/archive-source/pgvector-0.8.7'
assert (source/'vector.so').is_file()
assert target.resolve().is_relative_to(pathlib.Path('/home/boot/lhm'))
marker = target/'STAGED'
if not marker.exists():
    assert not target.exists(), 'Incomplete staging directory requires inspection'
    for relative in ['usr/lib/postgresql/14', 'usr/share/postgresql/14']:
        shutil.copytree(original/relative, target/relative, symlinks=False)
    extension = target/'usr/share/postgresql/14/extension'
    extension.mkdir(exist_ok=True)
    shutil.copy2(source/'vector.so', target/'usr/lib/postgresql/14/lib/vector.so')
    shutil.copy2(source/'vector.control', extension/'vector.control')
    for sql in (source/'sql').glob('vector--*.sql'):
        shutil.copy2(sql, extension/sql.name)
    marker.write_text('PostgreSQL 14.24 with pgvector 0.8.7; isolated tools copy\n')
print('Staged:',target)
'''

if __name__ == "__main__":
    with connect() as client:
        print(remote_python(client, SOURCE))
