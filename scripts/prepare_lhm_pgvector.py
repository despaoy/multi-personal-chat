"""Prepare PostgreSQL development files inside lhm, without system installation."""

from probe_lhm_runtime import connect
from run_lhm_postgres_tests import remote_python

SOURCE = r'''
import os, pathlib, subprocess
root = pathlib.Path('/home/boot/lhm/multipersonal-deploy/tools/pgvector-build')
root.mkdir(parents=True, exist_ok=True)
root = root.resolve()
assert root.is_relative_to(pathlib.Path('/home/boot/lhm'))
cache = root / 'downloads'
cache.mkdir(exist_ok=True)
temporary = root / 'tmp'
temporary.mkdir(exist_ok=True)
env = dict(os.environ, TMPDIR=str(temporary), HOME=str(root))
subprocess.run(['apt-get', 'download', 'postgresql-server-dev-14=14.24-0ubuntu0.22.04.1'],
               cwd=cache, env=env, check=True)
packages = list(cache.glob('postgresql-server-dev-14_14.24-0ubuntu0.22.04.1_amd64.deb'))
assert len(packages) == 1
subprocess.run(['dpkg-deb', '-x', str(packages[0]), str(root / 'sdk')], env=env, check=True)
header = root / 'sdk/usr/include/postgresql/14/server/postgres.h'
assert header.is_file()
print('Project-local PostgreSQL headers ready:', header)
'''

if __name__ == "__main__":
    with connect() as client:
        print(remote_python(client, SOURCE))
