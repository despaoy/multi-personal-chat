"""Build pinned pgvector in the project tree; never run make install."""

import hashlib
from pathlib import Path

import httpx
from probe_lhm_runtime import connect
from run_lhm_postgres_tests import remote_python

SOURCE = r'''
import hashlib, os, pathlib, subprocess, tarfile
root = pathlib.Path('/home/boot/lhm/multipersonal-deploy/tools/pgvector-build').resolve()
assert root.is_relative_to(pathlib.Path('/home/boot/lhm'))
env = dict(os.environ, TMPDIR=str(root/'tmp'), HOME=str(root))
archive = root/'downloads/pgvector-v0.8.7.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA
source = root/'archive-source/pgvector-0.8.7'
if not source.exists():
    destination = root/'archive-source'
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        for entry in bundle.getmembers():
            target = (destination/entry.name).resolve()
            assert target.is_relative_to(destination) and (entry.isfile() or entry.isdir())
        bundle.extractall(destination)
pgroot = pathlib.Path('/home/boot/lhm/multipersonal-runtime/tools/pgsql/usr')
args = ['make','-j2','PG_CONFIG='+str(pgroot/'lib/postgresql/14/bin/pg_config'),
        'PGXS='+str(pgroot/'lib/postgresql/14/lib/pgxs/src/makefiles/pgxs.mk'),
        'pgxsdir='+str(pgroot/'lib/postgresql/14/lib/pgxs'),
        'includedir_server='+str(root/'sdk/usr/include/postgresql/14/server'),
        'includedir_internal='+str(pgroot/'include/postgresql/internal'),
        'includedir='+str(pgroot/'include/postgresql'), 'with_llvm=no','OPTFLAGS=']
with (root/'build.log').open('w') as log:
    result = subprocess.run(args,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
if result.returncode:
    raise RuntimeError((root/'build.log').read_text()[-6000:])
assert (source/'vector.so').is_file()
print('Built pgvector v0.8.7 archive SHA256:', EXPECTED_SHA)
print('Artifact:',source/'vector.so')
'''

if __name__ == "__main__":
    cache = Path(__file__).resolve().parents[1]/'.uv-cache/pgvector'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache/'pgvector-v0.8.7.tar.gz'
    response = httpx.get('https://codeload.github.com/pgvector/pgvector/tar.gz/refs/tags/v0.8.7', timeout=45, follow_redirects=True)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != 'cac0b10c360f05b2d521200105ba3697e773d4cd3731f5a915a7e37ebe0bea85':
        raise RuntimeError('Pinned pgvector source archive digest changed')
    archive.write_bytes(response.content)
    with connect() as client:
        with client.open_sftp() as sftp:
            sftp.put(str(archive), '/home/boot/lhm/multipersonal-deploy/tools/pgvector-build/downloads/pgvector-v0.8.7.tar.gz')
        print(remote_python(client, 'EXPECTED_SHA='+repr(digest)+'\n'+SOURCE))
