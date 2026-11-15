"""Prepare pinned OpenSearch inside lhm; asynchronous download has a durable status."""

import json
from pathlib import Path

from probe_lhm_runtime import connect
from run_lhm_postgres_tests import remote_python

WORKER = r'''
import hashlib, json, pathlib, tarfile, urllib.request
root = pathlib.Path(__file__).resolve().parent
assert root.parent == pathlib.Path('/home/boot/lhm/multipersonal-deploy/tools')
status = root / 'prepare-status.json'
def report(phase, **values):
    status.write_text(json.dumps(dict(phase=phase, **values)))
try:
    url = 'https://artifacts.opensearch.org/releases/bundle/opensearch/3.9.0/opensearch-3.9.0-linux-x64.tar.gz'
    expected = 'a70e52a93ad66d9497dd734f5cc48eb4a8a8c4a92c47491a7ad0f1a898383d11c2948d4ba4159b2361d961bf221efa0fd02084e6389532df960b1ebe0c0a2099'
    archive = root / 'opensearch.tar.gz'
    digest, size = hashlib.sha512(), 0
    if archive.exists():
        report('verifying')
        with archive.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
                size += len(chunk)
    else:
        report('downloading', bytes=0)
        with urllib.request.urlopen(url, timeout=45) as response, archive.open('wb') as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size % (16 * 1024 * 1024) == 0:
                    report('downloading', bytes=size)
    if digest.hexdigest() != expected:
        raise ValueError('Official archive SHA512 mismatch')
    report('extracting', bytes=size)
    with tarfile.open(archive) as package:
        for member in package.getmembers():
            destination = (root / member.name).resolve()
            if not destination.is_relative_to(root) or not (member.name.startswith('opensearch-3.9.0/') or (member.name == 'opensearch-3.9.0' and member.isdir())):
                raise ValueError('Archive member escapes installation directory')
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError('Archive contains unsupported special member')
            if member.issym() or member.islnk():
                target = ((destination.parent if member.issym() else root) / member.linkname).resolve()
                if pathlib.Path(member.linkname).is_absolute() or not target.is_relative_to(root):
                    raise ValueError('Archive link escapes installation directory')
        package.extractall(root)
    report('ready', home=str(root / 'opensearch-3.9.0'), sha512=expected, bytes=size)
except Exception as exc:
    report('failed', error=type(exc).__name__ + ': ' + str(exc))
    raise
'''


def main():
    with connect() as ssh:
        source = "import pathlib,tempfile,subprocess,json\n"
        source += "base=pathlib.Path('/home/boot/lhm/multipersonal-deploy/tools')\nbase.mkdir(exist_ok=True)\n"
        source += "root=pathlib.Path(tempfile.mkdtemp(prefix='opensearch-prepare-',dir=base))\n"
        source += 'worker=' + repr(WORKER) + '\n'
        source += "(root/'prepare.py').write_text(worker)\n"
        source += "with (root/'prepare.log').open('wb') as log:\n    process=subprocess.Popen(['python3',str(root/'prepare.py')],cwd=root,stdout=log,stderr=log,start_new_session=True)\n"
        source += "print(json.dumps({'root':str(root),'pid':process.pid}))\n"
        state = json.loads(remote_python(ssh, source))
        target = Path(__file__).resolve().parents[1] / '.codex-test-runtime-opensearch'
        target.mkdir(exist_ok=True)
        (target / 'prepare.json').write_text(json.dumps(state), encoding='utf-8')
        print(json.dumps(state))


if __name__ == '__main__':
    main()
