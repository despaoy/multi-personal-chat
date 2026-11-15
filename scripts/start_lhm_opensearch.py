"""Start only a fresh loopback validation node, with every writable path in lhm."""

import json
from pathlib import Path

from probe_lhm_runtime import connect
from run_lhm_postgres_tests import remote_python

START = r'''
import json, os, pathlib, shutil, socket, subprocess, tempfile
prepared = pathlib.Path(prepare_root).resolve()
base = pathlib.Path('/home/boot/lhm/multipersonal-deploy').resolve()
assert prepared.parent == base / 'tools'
state = json.loads((prepared / 'prepare-status.json').read_text())
if state['phase'] != 'ready':
    raise RuntimeError('Verified OpenSearch extraction is not ready')
home = pathlib.Path(state['home']).resolve()
assert home.parent == prepared
root = pathlib.Path(tempfile.mkdtemp(prefix='opensearch-validation-', dir=base)).resolve()
assert root.parent == base
for name in ('data', 'logs', 'tmp', 'cache'):
    (root / name).mkdir()
shutil.copytree(home / 'config', root / 'config')
# This optional plugin defaults to shared-memory/extra-listener paths. Keep it out of the isolated node.
plugin = home / 'plugins' / 'opensearch-performance-analyzer'
if plugin.exists():
    disabled = prepared / 'disabled-plugins'
    disabled.mkdir(exist_ok=True)
    plugin.rename(disabled / plugin.name)
def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]
http_port, transport_port = port(), port()
settings = {
    'cluster.name': 'lhm-source-validation', 'node.name': root.name,
    'network.host': '127.0.0.1', 'http.port': http_port, 'transport.port': transport_port,
    'discovery.type': 'single-node', 'plugins.security.disabled': True,
    'node.store.allow_mmap': False, 'bootstrap.memory_lock': False,
    'path.data': str(root / 'data'), 'path.logs': str(root / 'logs'),
}
(root / 'config/opensearch.yml').write_text('\n'.join(key + ': ' + json.dumps(value) for key, value in settings.items()) + '\n')
env = dict(os.environ, HOME=str(root), TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
           OPENSEARCH_PATH_CONF=str(root / 'config'), OPENSEARCH_TMPDIR=str(root / 'tmp'),
           OPENSEARCH_JAVA_HOME=str(home / 'jdk'),
           OPENSEARCH_JAVA_OPTS='-Xms1g -Xmx1g -XX:ActiveProcessorCount=2 -XX:-UsePerfData -Duser.home=' + str(root))
with (root / 'console.log').open('wb') as log:
    process = subprocess.Popen([str(home / 'bin/opensearch')], cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
info = {'root': str(root), 'home': str(home), 'pid': process.pid, 'port': http_port, 'transport_port': transport_port}
(root / 'instance.json').write_text(json.dumps(info))
print(json.dumps(info))
'''


def main():
    runtime = Path(__file__).resolve().parents[1] / '.codex-test-runtime-opensearch'
    if (runtime / 'instance.json').exists():
        raise RuntimeError('A validation instance record exists; inspect that instance before starting another')
    state = json.loads((runtime / 'local-transfer.json').read_text())
    with connect() as ssh:
        instance = json.loads(remote_python(ssh, 'prepare_root=' + repr(state['root']) + '\n' + START))
    (runtime / 'instance.json').write_text(json.dumps(instance), encoding='utf-8')
    print(json.dumps(instance))


if __name__ == '__main__':
    main()
